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
    yaw_deadband_deg: float = Field(default=3, gt=0, le=10)
    target_recovery_seconds: float = Field(default=5, gt=0, le=30)
    # Temporary hybrid-HITL diagnostic: simulated yaw cannot turn the real feed.
    # Object destinations retain normal tracking and destination steering.
    sim_forward_stop_test: bool = False
    # Diagnostic output only; the independent real-camera corridor keeps its
    # low-speed planning profile. The deployed native bridge accepts <=2 m/s.
    sim_forward_speed_mps: Optional[float] = Field(default=None, gt=0, le=2, strict=True)

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


def simulation_forward_stop_config(config, forward_speed_mps=None):
    """Derive a temporary test profile without changing saved calibration."""
    output_speed = config.sim_forward_speed_mps if forward_speed_mps is None else forward_speed_mps
    if output_speed is not None:
        if (type(output_speed) not in (int, float) or not math.isfinite(output_speed)
                or not 0 < output_speed <= 2):
            raise ValueError("Simulation forward speed must be finite and in (0, 2] m/s")
        output_speed = float(output_speed)
    return config.model_copy(update=dict(sim_forward_stop_test=True,
                                         max_speed_mps=min(config.max_speed_mps, 0.2),
                                         sim_forward_speed_mps=output_speed,
                                         clearance_margin_m=0.2,
                                         minimum_clearance_m=2.0))


def is_sim_forward_instruction(text):
    """Recognize only unambiguous forward intents for the hybrid diagnostic."""
    if not isinstance(text, str):
        return False
    normalized = " ".join(text.strip().casefold().rstrip(".!?").split())
    return normalized in {
        "forward", "move forward", "go forward", "fly forward",
        "move straight", "go straight", "fly straight",
        "move straight ahead", "go straight ahead", "fly straight ahead",
    }


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
    """Visually servo toward the destination; stop for an obstructed forward corridor.

    A destination outside the alignment deadband causes an in-place turn. Small
    bearing errors are corrected during forward approach. There is no blind
    lateral/backward/vertical motion or automatic detour around an obstruction.
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
    if config.sim_forward_stop_test:
        # This diagnostic follows the real camera's current forward corridor,
        # rather than the grounded object's bearing or arrival stand-off. It
        # isolates forward motion/obstacle holds when SDK simulation and video
        # do not share the same pose. Native motion authority, freshness and
        # expiry checks still apply; object grounding/tracking are intentionally
        # absent from this literal-forward diagnostic.
        rx, ry = fx*radius/lookahead, fy*radius/lookahead
        if cx-rx < 0 or cx+rx >= w or cy-ry < 0 or cy+ry >= h:
            return axes, "blocked", "SIM forward-stop: inflated footprint does not fit the camera view"
        yy, xx = np.mgrid[:h, :w]
        px, py = (xx-cx)/fx*z, (yy-cy)/fy*z
        collision = (z <= lookahead+radius) & (abs(px) <= radius) & (abs(py) <= radius)
        if np.any(collision):
            return axes, "blocked", "SIM forward-stop: obstacle in the current forward corridor; holding"
        axes["forward"] = config.sim_forward_speed_mps if config.sim_forward_speed_mps is not None else speed
        return axes, "approaching", "SIM forward-stop: current corridor clear; destination steering disabled"
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
    rx, ry = fx*radius/lookahead, fy*radius/lookahead
    if cx-rx < 0 or cx+rx >= w or cy-ry < 0 or cy+ry >= h:
        return axes, "blocked", "Inflated forward footprint does not fit the camera view"
    collision = (z <= lookahead+radius) & (abs(px) <= radius) & (abs(py) <= radius)
    if np.any(collision):
        return axes, "blocked", "Immediate obstacle in the forward corridor; holding"
    bearing_deg = math.degrees(target_heading)
    axes["yaw"] = float(np.clip(0.8*bearing_deg, -config.max_yaw_deg_s, config.max_yaw_deg_s))
    if abs(bearing_deg) > config.yaw_deadband_deg:
        return axes, "aligning", "Turning toward the tracked destination"
    axes["forward"] = speed
    return axes, "approaching", "Approaching the tracked destination; forward corridor clear"


def target_observation(config, instruction, metadata, sequence, box=None, depth=None,
                       visibility="pending", confidence=None, description=None):
    """Report the current observation in source pixels and forward-aligned body axes.

    Metric depth is optical-axis Z. Euclidean distance and right/up coordinates
    are derived from the target-box center ray, not independent range sensing.
    Missing observations deliberately have no current pixel or distance values.
    """
    target = dict(instruction=instruction, visibility=visibility,
                  image_size_px=[config.image_width, config.image_height],
                  box_normalized=None, box_px=None, center_px=None, bearing_deg=None,
                  distance_m=None, forward_distance_m=None, position_body_m=None,
                  distance_source="monocular_metric_estimate", fresh=False,
                  observed_sequence=sequence, frame_id=metadata.get("frame_id"),
                  frame_monotonic_ns=metadata.get("frame_monotonic_ns"),
                  grounding_confidence=confidence, description=description)
    if box is None:
        return target
    x1, y1, x2, y2 = (float(value) for value in box)
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("Tracked destination box must be inside the image")
    u, v = (x1+x2)*config.image_width/2, (y1+y2)*config.image_height/2
    right_ray, down_ray = (u-config.cx)/config.fx, (v-config.cy)/config.fy
    target.update(box_normalized=[x1, y1, x2, y2],
                  box_px=[round(x1*config.image_width, 2), round(y1*config.image_height, 2),
                          round(x2*config.image_width, 2), round(y2*config.image_height, 2)],
                  center_px=[round(u, 2), round(v, 2)],
                  bearing_deg=round(math.degrees(math.atan(right_ray)), 3), fresh=True)
    if depth is not None:
        z = np.asarray(depth)
        h, w = z.shape
        expected_h = round(config.processing_width*config.image_height/config.image_width)
        if (w, h) != (config.processing_width, expected_h):
            raise ValueError("Depth must be aligned with the processed camera frame")
        roi = z[max(0, int(y1*h)):min(h, math.ceil(y2*h)),
                max(0, int(x1*w)):min(w, math.ceil(x2*w))]
        if roi.size and np.all(np.isfinite(roi)) and np.all(roi > 0):
            forward = float(np.median(roi))*config.depth_scale
            right, up = right_ray*forward, -down_ray*forward
            target.update(distance_m=round(math.sqrt(forward**2+right**2+up**2), 3),
                          forward_distance_m=round(forward, 3),
                          position_body_m=dict(forward=round(forward, 3), right=round(right, 3), up=round(up, 3)))
    return target


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
    forward_test: bool = False
    tracker_ready: bool = False
    tracking_missing_since: Optional[float] = None
    target: dict = field(default_factory=dict)
    grounding_confidence: Optional[float] = None
    description: Optional[str] = None


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
        goal = self.goal
        if goal is not None and (goal.message.goal_id, goal.message.owner) == (message.goal_id, message.owner):
            if goal.target:
                info["target"] = dict(goal.target)
                # A terminal/repeated update must not present a previous frame's
                # coordinates or range as a current observation.
                info["target"]["fresh"] = (goal.target.get("observed_sequence") == message.sequence
                                             and state not in {"stale", "lost", "error"}
                                             and goal.target.get("fresh", False))
            if goal.forward_test:
                # Only literal-forward diagnostics can request the optional
                # native HITL speed override. Object goals never carry it.
                info["behavior"] = "sim-forward-stop-test"
        if info.get("target") is not None:
            LOGGER.debug("goal=%s sequence=%d target=%s", message.goal_id, message.sequence,
                        json.dumps(info["target"], separators=(",", ":"), allow_nan=False))
        LOGGER.debug("goal=%s sequence=%d state=%s timings_ms=%s reason=%s",
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
            if self.config.sim_forward_stop_test and is_sim_forward_instruction(message.text):
                self.cancel()
                self.goal = Goal(message, started, None, state="ready",
                                 reason="SIM forward-stop: forward instruction accepted; waiting for next fresh frame",
                                 last_frame_time=started, last_frame_id=metadata.get("frame_id", 0),
                                 last_source_ns=metadata.get("frame_monotonic_ns", 0), forward_test=True)
                # A directional intent has no object to ground or follow. The
                # initial response holds; a new frame and depth check must pass
                # before any movement can be returned.
                return self.response(metadata, message, self.goal.state, self.goal.reason, started)
            if self.pending is not None and not self.pending.done():
                return self.response(metadata, message, "error", "Previous grounding job is still finishing", started)
            self.cancel()
            self.pending = self.executor.submit(self.grounder, message.text, frame.copy())
            # Object destinations always track and steer, even if the optional
            # literal-forward diagnostic is enabled for other instructions.
            capacity = math.ceil(self.config.grounding_timeout_seconds/0.2)+4
            self.goal = Goal(message, started, self.pending, frames=deque(maxlen=capacity), last_frame_time=started,
                             last_frame_id=metadata.get("frame_id", 0), last_source_ns=metadata.get("frame_monotonic_ns", 0))
            self.goal.target = target_observation(self.config, message.text, metadata, message.sequence)
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
                    LOGGER.debug("goal=%s grounded_target found=%s box=%s confidence=%.3f reason=%s",
                                message.goal_id, result.found, result.box, result.confidence, result.reason)
                    if not result.found or result.confidence < self.config.min_grounding_confidence:
                        raise ValueError("Destination absent or ambiguous: " + result.reason)
                    LOGGER.info("Destination selected: %s (confidence=%.2f)", result.reason, result.confidence)
                    goal.grounding_confidence, goal.description = result.confidence, result.reason
                    goal.tracker = self.tracker_factory()
                    goal.tracker.start(goal.frames.popleft()[1], result.box)
                    goal.tracker_ready = True
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
            if goal.forward_test:
                box, tracking_ms = None, 0.0
            else:
                if (goal.tracking_missing_since is not None
                        and started-goal.tracking_missing_since >= self.config.target_recovery_seconds):
                    goal.state, goal.reason = "lost", "Destination tracking recovery timed out; manual takeover or a new goal is required"
                    return self.response(metadata, message, goal.state, goal.reason, started)
                box = goal.tracker.update(frame)
                tracking_ms = (self.clock()-tracking_started)*1000
                goal.target = target_observation(self.config, goal.message.text, metadata, message.sequence, box,
                                                 visibility="visible", confidence=goal.grounding_confidence,
                                                 description=goal.description)
                tracking_status = getattr(goal.tracker, "last_status", None)
                tracking_confidence = getattr(goal.tracker, "confidence", None)
            depth_started = self.clock()
            depth = self.depth(frame)
            depth_ms = (self.clock()-depth_started)*1000
            planning_started = self.clock()
            planning_config = self.config.model_copy(update={"sim_forward_stop_test": goal.forward_test,
                                                             "sim_forward_speed_mps": self.config.sim_forward_speed_mps if goal.forward_test else None})
            if continuous:
                planning_config = planning_config.model_copy(update={"step_seconds": min(MAX_COMMAND_LEASE_SECONDS, metadata["default_duration"])})
            if not goal.forward_test:
                goal.target = target_observation(self.config, goal.message.text, metadata, message.sequence, box, depth,
                                                 visibility="visible", confidence=goal.grounding_confidence,
                                                 description=goal.description)
                if isinstance(tracking_status, str):
                    goal.target["tracking_status"] = tracking_status
                if type(tracking_confidence) in (int, float) and math.isfinite(tracking_confidence):
                    goal.target["tracking_confidence"] = tracking_confidence
            command, state, reason = local_command(planning_config, np.asarray(depth), box, self.clock()-started)
            if goal.tracking_missing_since is not None:
                # Do not actuate on the first match after a tracking outage.
                # Confirm again in a subsequent fresh frame before approaching.
                goal.tracking_missing_since = None
                if state in {"aligning", "approaching"}:
                    command, state, reason = zero_axes(planning_config.step_seconds), "ready", "Destination recovered; waiting for a fresh confirmation frame"
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
            visibility = getattr(error, "status", "unconfirmed")
            if visibility not in {"out_of_view", "unconfirmed"}:
                visibility = "unconfirmed"
            goal.target = target_observation(self.config, goal.message.text, metadata, message.sequence,
                                             visibility=visibility, confidence=goal.grounding_confidence,
                                             description=goal.description)
            if goal.tracker_ready:
                goal.frames.clear()
                if goal.tracking_missing_since is None:
                    goal.tracking_missing_since = started
                goal.state = "stale"
                goal.reason = ("Tracked destination projects outside the camera view; holding for visual recovery" if visibility == "out_of_view"
                               else "Destination position is uncertain; holding and retrying visual tracking")
            else:
                goal.state, goal.reason = "lost", "Destination tracking could not initialize; position unconfirmed"
            LOGGER.warning("Destination tracking %s; holding: %s", visibility, error)
            return self.response(metadata, message, goal.state, goal.reason, started)
        except Exception as error:
            LOGGER.warning("Goal held after perception/planning failure: %s", error, exc_info=not isinstance(error, ValueError))
            goal.state, goal.reason = "error", str(error)
            return self.response(metadata, message, goal.state, goal.reason, started)
