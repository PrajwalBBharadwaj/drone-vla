"""Persistent visual goals over the existing DJI one-frame /plan protocol.

The controller is a bounded reactive approach behavior, not a global navigator.
Only the DJI process owns the SDK; this module can only return expiring axes.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import logging
import math
import time
from typing import Literal, Optional

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from navigation_perception import FeatureTracker, GroundedTarget, MetricDepth, OllamaGrounder, TrackingLost
from bridge_protocol import MAX_COMMAND_LEASE_SECONDS, MAX_FRAME_AGE_SECONDS

PREFIX = "@nav1:"
RESULT_PREFIX = "navigation:"
LOGGER = logging.getLogger("drone_vla.navigation")


class NavigationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    # Intrinsics for the exact uncropped, rectified DJI frame, before resizing.
    image_width: int = Field(gt=0, le=4096)
    image_height: int = Field(gt=0, le=2160)
    fx: float = Field(gt=0)
    fy: float = Field(gt=0)
    cx: float = Field(gt=0)
    cy: float = Field(gt=0)
    camera_forward_aligned: Literal[True]
    processing_width: int = Field(default=448, ge=224, le=672)
    max_speed_mps: float = Field(default=0.5, gt=0, le=1)
    max_yaw_deg_s: float = Field(default=15, gt=0, le=30)
    step_seconds: float = Field(default=0.2, ge=0.1, le=0.5)
    vehicle_radius_m: float = Field(gt=0, le=3)
    clearance_margin_m: float = Field(default=0.5, ge=0.2, le=3)
    braking_deceleration_mps2: float = Field(gt=0, le=5)
    minimum_clearance_m: float = Field(default=3, ge=2, le=10)
    standoff_m: float = Field(default=5, ge=3, le=30)
    depth_scale: float = Field(default=0.65, gt=0, le=1)
    max_processing_seconds: float = Field(default=0.35, gt=0, le=1)
    max_frame_gap_seconds: float = Field(default=1, ge=0.3, le=2)
    max_goal_seconds: float = Field(default=120, gt=0, le=300)
    grounding_timeout_seconds: float = Field(default=60, gt=0, le=120)
    min_grounding_confidence: float = Field(default=0.8, ge=0.7, le=1)

    @model_validator(mode="after")
    def geometry(self):
        if not 0 < self.cx < self.image_width or not 0 < self.cy < self.image_height:
            raise ValueError("Principal point must be inside the calibrated image")
        if self.standoff_m <= self.vehicle_radius_m + self.clearance_margin_m:
            raise ValueError("Stand-off must exceed inflated vehicle radius")
        if round(self.processing_width*self.image_height/self.image_width) > 672:
            raise ValueError("Processing height exceeds the bounded image size")
        return self

    def frame(self, image):
        if image.size != (self.image_width, self.image_height):
            raise ValueError("Live frame dimensions differ from camera calibration")
        height = round(self.processing_width * self.image_height / self.image_width)
        return image.resize((self.processing_width, height), Image.Resampling.BILINEAR)


class NavigationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    op: Literal["start", "step"]
    goal_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    owner: str = Field(min_length=1, max_length=128)
    sequence: int = Field(ge=0)
    text: Optional[str] = Field(default=None, min_length=1, max_length=600)

    @model_validator(mode="after")
    def start_text(self):
        if (self.op == "start") != (self.text is not None):
            raise ValueError("Only start must carry the original instruction")
        if self.op == "start" and (self.sequence != 0 or not self.text.strip()):
            raise ValueError("Start requires sequence zero and nonblank text")
        return self


def encode_message(**values):
    message = NavigationMessage(**values)
    text = PREFIX + message.model_dump_json(exclude_none=True)
    if len(text.encode()) > 1000:
        raise ValueError("Navigation instruction envelope exceeds DJI's 1000-byte limit")
    return text


def zero_axes(duration):
    return dict(forward=0.0, right=0.0, up=0.0, yaw=0.0, duration_seconds=duration)


def local_command(config, depth, box, elapsed):
    """Score observed forward headings using an inflated vehicle and stopping distance.

    No backward, lateral or vertical movement into unobserved space. No fallback
    that crosses a known obstacle. A turn is performed before forward motion.
    """
    axes = zero_axes(config.step_seconds)
    h, w = depth.shape
    expected_h = round(config.processing_width * config.image_height / config.image_width)
    if (w, h) != (config.processing_width, expected_h):
        raise ValueError("Depth must be aligned with the processed camera frame")
    if not np.all(np.isfinite(depth)) or np.any(depth <= 0):
        return axes, "blocked", "Depth contains unknown or invalid regions"
    z = depth.astype(np.float32) * config.depth_scale
    scale = w / config.image_width
    fx, fy, cx, cy = config.fx*scale, config.fy*(h/config.image_height), config.cx*scale, config.cy*(h/config.image_height)
    radius = config.vehicle_radius_m + config.clearance_margin_m
    if float(z.min()) <= radius:
        return axes, "blocked", "Obstacle inside the inflated vehicle radius"
    # Include the bridge's maximum input-frame age, update delay, movement lease
    # and conservative stopping distance. No actual-velocity telemetry is available.
    speed = config.max_speed_mps
    lookahead = max(config.minimum_clearance_m,
                    radius + speed*(MAX_FRAME_AGE_SECONDS + elapsed + config.step_seconds + config.max_frame_gap_seconds)
                    + speed*speed/(2*config.braking_deceleration_mps2))
    x1, y1, x2, y2 = box
    target = z[max(0, int(y1*h)):min(h, math.ceil(y2*h)),
               max(0, int(x1*w)):min(w, math.ceil(x2*w))]
    if target.size == 0:
        return axes, "lost", "Target is outside the image"
    if float(np.median(target)) <= config.standoff_m:
        return axes, "arrived", "Estimated target stand-off reached; holding"
    target_heading = math.atan2((x1+x2)*w/2 - cx, fx)
    # This behavior holds altitude. A target far above/below the optical axis
    # requires another behavior rather than silently inventing vertical motion.
    if not y1*h - h*0.1 <= cy <= y2*h + h*0.1:
        return axes, "blocked", "Target requires an unsupported altitude change"
    yy, xx = np.mgrid[:h, :w]
    px = (xx-cx)/fx*z
    py = (yy-cy)/fy*z
    headings = np.radians(np.arange(-25, 26, 5))
    free = []
    for heading in headings:
        # The endpoint footprint must fit in the visible FOV.
        u = cx + fx*math.tan(heading)
        rx, ry = fx*radius/lookahead, fy*radius/lookahead
        if u-rx < 0 or u+rx >= w or cy-ry < 0 or cy+ry >= h:
            continue
        along = z*math.cos(heading) + px*math.sin(heading)
        across = px*math.cos(heading) - z*math.sin(heading)
        collision = (along > 0) & (along <= lookahead+radius) & (abs(across) <= radius) & (abs(py) <= radius)
        if not np.any(collision):
            free.append(float(heading))
    if not free:
        return axes, "blocked", "No observed heading passes the stopping-distance check"
    heading = min(free, key=lambda angle: abs(angle-target_heading) + 0.15*abs(angle))
    # An in-place turn allows the next camera frame to check the new forward path.
    # Follow the selected free heading, including straight ahead when the target
    # bearing is obstructed. Turning straight back toward an obstructed target
    # would undo the detour on every subsequent frame.
    error = heading
    if abs(error) >= math.radians(5):
        axes["yaw"] = math.copysign(min(config.max_yaw_deg_s, max(5, abs(math.degrees(error))*0.8)), error)
        return axes, "aligning", "Turning toward an observed free heading"
    # Never translate on a candidate heading different from the actual nose direction.
    if 0.0 not in free:
        return axes, "blocked", "Current forward corridor is blocked"
    axes["forward"] = speed
    return axes, "approaching", "Forward corridor passes the local depth check"


@dataclass
class Goal:
    message: NavigationMessage
    started: float
    future: object
    frames: deque = field(default_factory=lambda: deque(maxlen=256))
    tracker: object = None
    state: str = "grounding"
    reason: str = "Grounding the destination; holding"
    sequence: int = 0
    last_frame_time: float = 0
    last_frame_id: int = 0
    last_source_ns: int = 0


class NavigationEngine:
    """One active goal, one background grounding job, no queued old goals."""
    def __init__(self, config, grounder=None, depth=None, tracker_factory=FeatureTracker,
                 clock=time.monotonic, executor=None):
        self.config = config
        self.grounder = grounder or OllamaGrounder(timeout=config.grounding_timeout_seconds+5)
        self.depth = depth or MetricDepth()
        self.tracker_factory = tracker_factory
        self.clock = clock
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="goal-grounding")
        self.goal = None
        self.pending = None

    def close(self):
        self.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def cancel(self):
        self.goal = None
        if self.pending is not None:
            self.pending.cancel()  # Running network calls finish, but their results cannot actuate.

    def preload(self):
        h = round(self.config.processing_width*self.config.image_height/self.config.image_width)
        self.depth(Image.new("RGB", (self.config.processing_width, h)))

    def response(self, metadata, message, state, reason, started, command=None, **timings):
        timings["service_ms"] = round((self.clock()-started)*1000, 2)
        info = dict(goal_id=message.goal_id, sequence=message.sequence, state=state,
                    reason=reason[:180], timings_ms=timings)
        LOGGER.info("goal=%s sequence=%d state=%s timings_ms=%s reason=%s",
                    message.goal_id, message.sequence, state, timings, reason)
        duration = (min(MAX_COMMAND_LEASE_SECONDS, metadata["default_duration"]) if metadata.get("bridge_protocol") == 2
                    else min(self.config.step_seconds, metadata["default_duration"]))
        return dict(ok=True, request_id=metadata["request_id"],
                    command=command or zero_axes(duration),
                    override_reason=RESULT_PREFIX + json.dumps(info, separators=(",", ":")), navigation=info)

    def handle(self, metadata, image, received_at=None):
        started = self.clock() if received_at is None else received_at
        message = NavigationMessage.model_validate_json(metadata["text"][len(PREFIX):])
        continuous = metadata.get("bridge_protocol") == 2
        if continuous and (metadata["session"], metadata["goal_id"]) != (message.owner, message.goal_id):
            raise ValueError("Navigation envelope does not match native goal/session")
        # Preparation is included in the response freshness budget.
        frame = self.config.frame(image)
        if message.op == "start":
            if self.pending is not None and not self.pending.done():
                return self.response(metadata, message, "error", "Previous grounding job is still finishing", started)
            self.cancel()
            self.pending = self.executor.submit(self.grounder, message.text, frame.copy())
            # Sample grounding history at 5 Hz. Capacity is tied to the deadline
            # and never silently evicts the original grounding reference frame.
            capacity = math.ceil(self.config.grounding_timeout_seconds/0.2)+4
            self.goal = Goal(message, started, self.pending, frames=deque(maxlen=capacity), last_frame_time=started,
                             last_frame_id=metadata.get("frame_id", 0), last_source_ns=metadata.get("frame_monotonic_ns", 0))
            self.goal.frames.append((started, frame))
            return self.response(metadata, message, "grounding", self.goal.reason, started)
        goal = self.goal
        if goal is None or (goal.message.goal_id, goal.message.owner) != (message.goal_id, message.owner):
            return self.response(metadata, message, "error", "Goal owner changed or goal was cancelled", started)
        sequence_invalid = message.sequence <= goal.sequence if continuous else message.sequence != goal.sequence+1
        if sequence_invalid:
            goal.state, goal.reason = "error", "Repeated or out-of-order navigation step"
        goal.sequence = message.sequence
        if continuous:
            if metadata["frame_id"] <= goal.last_frame_id or metadata["frame_monotonic_ns"] <= goal.last_source_ns:
                goal.state, goal.reason = "error", "Repeated or out-of-order native camera frame"
            elif (metadata["frame_monotonic_ns"]-goal.last_source_ns)/1e9 > self.config.max_frame_gap_seconds:
                goal.state, goal.reason = "lost", "Camera decode stream interrupted"
            goal.last_frame_id, goal.last_source_ns = metadata["frame_id"], metadata["frame_monotonic_ns"]
        if started-goal.started > self.config.max_goal_seconds:
            goal.state, goal.reason = "error", "Goal deadline exceeded"
        if started-goal.last_frame_time > self.config.max_frame_gap_seconds:
            goal.state, goal.reason = "lost", "Fresh-frame stream interrupted"
        goal.last_frame_time = started
        if goal.state in {"lost", "blocked", "arrived", "error"}:
            return self.response(metadata, message, goal.state, goal.reason, started)
        try:
            if goal.state in {"grounding", "acquiring"}:
                if len(goal.frames) == goal.frames.maxlen:
                    raise ValueError("Grounding frame buffer exhausted")
                if not goal.frames or started-goal.frames[-1][0] >= 0.2-1e-9:
                    goal.frames.append((started, frame))
                if goal.state == "grounding":
                    if started-goal.started > self.config.grounding_timeout_seconds:
                        raise ValueError("Grounding deadline exceeded")
                    if not goal.future.done():
                        return self.response(metadata, message, "grounding", goal.reason, started)
                    result = GroundedTarget.model_validate(goal.future.result())
                    if not result.found or result.confidence < self.config.min_grounding_confidence:
                        raise ValueError("Destination absent or ambiguous: " + result.reason)
                    goal.tracker = self.tracker_factory()
                    goal.tracker.start(goal.frames.popleft()[1], result.box)
                    goal.state, goal.reason = "acquiring", "Tracking buffered frames; holding"
                # Drain in bounded batches while continuing to accept current
                # frames. A long grounding job must not create one large replay
                # stall that itself interrupts the fresh-frame stream.
                while goal.frames and self.clock()-started < self.config.max_processing_seconds*0.5:
                    _, buffered = goal.frames.popleft()
                    goal.tracker.update(buffered)
                if goal.frames:
                    return self.response(metadata, message, goal.state,
                                         f"{goal.reason} ({len(goal.frames)} frames remaining)", started)
                goal.state, goal.reason = "ready", "Target acquired; waiting for next fresh frame"
                # Grounding/replay never returns movement based on an aged scene.
                return self.response(metadata, message, goal.state, goal.reason, started)
            tracking_started = self.clock()
            box = goal.tracker.update(frame)
            tracking_ms = (self.clock()-tracking_started)*1000
            depth_started = self.clock()
            depth = self.depth(frame)
            depth_ms = (self.clock()-depth_started)*1000
            planning_started = self.clock()
            planning_config = self.config
            if continuous:
                planning_config = self.config.model_copy(update={"step_seconds": min(MAX_COMMAND_LEASE_SECONDS, metadata["default_duration"])})
            command, state, reason = local_command(planning_config, np.asarray(depth), box, self.clock()-started)
            elapsed = self.clock()-started
            if elapsed > self.config.max_processing_seconds:
                # Keep the tracker up to date, but ask for a new frame before motion.
                command, state, reason = zero_axes(planning_config.step_seconds), "stale", "Processing exceeded the fresh-frame budget"
            command["duration_seconds"] = min(command["duration_seconds"], metadata["default_duration"])
            goal.state, goal.reason = state, reason
            return self.response(metadata, message, state, reason, started, command,
                                 tracking_ms=round(tracking_ms, 2), depth_ms=round(depth_ms, 2),
                                 planning_ms=round((self.clock()-planning_started)*1000, 2))
        except TrackingLost as error:
            LOGGER.warning("Target lost; holding: %s", error)
            goal.state, goal.reason = "lost", str(error)
            return self.response(metadata, message, goal.state, goal.reason, started)
        except Exception as error:
            LOGGER.warning("Goal held after perception/planning failure: %s", error, exc_info=not isinstance(error, ValueError))
            goal.state, goal.reason = "error", str(error)
            return self.response(metadata, message, goal.state, goal.reason, started)
