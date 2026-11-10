"""Small, expiring display snapshots for DJI's operator video.

The video overlay is independent of flight commands. It contains only a current
correlated visual observation, never input pixels or a new inference request.
"""
from copy import deepcopy
import math
import threading
import time

from bridge_protocol import MAX_FRAME_AGE_SECONDS, FUTURE_TOLERANCE_SECONDS

CLEAR = {"protocol": 1, "action": "clear"}
DISPLAY_STATES = {"aligning", "approaching", "ready", "arrived", "blocked"}


def monotonic_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def snapshot_from_response(response, now_ns):
    """Restrict display data to the same frame accepted by the wire validator."""
    navigation = response.get("navigation", {})
    target = navigation.get("target", {})
    if (response.get("ok") is not True or type(response.get("bridge_protocol")) is not int
            or response.get("bridge_protocol") != 2
            or navigation.get("state") not in DISPLAY_STATES
            or target.get("fresh") is not True or target.get("visibility") != "visible"):
        return dict(CLEAR)
    if (any(type(target.get(key)) is not int for key in ("observed_sequence", "frame_id", "frame_monotonic_ns"))
            or target.get("observed_sequence") != navigation.get("sequence")
            or target.get("frame_id") != response.get("frame_id")
            or target.get("frame_monotonic_ns") != response.get("frame_monotonic_ns")
            or navigation.get("goal_id") != response.get("goal_id")):
        return dict(CLEAR)
    for key, maximum in (("session", 128), ("goal_id", 64), ("request_id", 256)):
        if not isinstance(response.get(key), str) or not 0 < len(response[key]) <= maximum:
            return dict(CLEAR)
    for value in (navigation.get("sequence"), response.get("frame_id"), response.get("frame_monotonic_ns")):
        if type(value) is not int or not 0 <= value < 2**63:
            return dict(CLEAR)
    timestamp = response["frame_monotonic_ns"]
    age = (now_ns-timestamp)/1e9
    if timestamp <= 0 or response["frame_id"] <= 0 or not -FUTURE_TOLERANCE_SECONDS <= age <= MAX_FRAME_AGE_SECONDS:
        return dict(CLEAR)
    size = target.get("image_size_px")
    if (not isinstance(size, (tuple, list)) or len(size) != 2
            or any(type(value) is not int for value in size)
            or not 0 < size[0] <= 4096 or not 0 < size[1] <= 2160):
        return dict(CLEAR)
    box = target.get("box_normalized")
    if (not isinstance(box, (tuple, list)) or len(box) != 4
            or not all(finite_number(value) for value in box)
            or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
        return dict(CLEAR)
    confidence = target.get("tracking_confidence")
    if confidence is None:
        confidence = target.get("grounding_confidence")
    distance, bearing = target.get("distance_m"), target.get("bearing_deg")
    if confidence is not None and (not finite_number(confidence) or not 0 <= confidence <= 1):
        return dict(CLEAR)
    if distance is not None and (not finite_number(distance) or distance <= 0):
        return dict(CLEAR)
    if bearing is not None and (not finite_number(bearing) or not -90 <= bearing <= 90):
        return dict(CLEAR)
    return dict(protocol=1, action="target", source="wide", session=response["session"],
                goal_id=response["goal_id"], request_id=response["request_id"],
                sequence=navigation["sequence"], frame_id=response["frame_id"],
                frame_monotonic_ns=timestamp, image_size_px=list(size),
                box_normalized=list(box), confidence=confidence, distance_m=distance,
                bearing_deg=bearing, state=navigation["state"], visibility="visible")


class OverlayState:
    """Latest-only cache; HTTP readers do not acquire the inference lock."""
    def __init__(self, clock_ns=monotonic_ns):
        self.clock_ns = clock_ns
        self.lock = threading.Lock()
        self.snapshot = dict(CLEAR)

    def clear(self):
        with self.lock:
            self.snapshot = dict(CLEAR)

    def update(self, response):
        try:
            value = snapshot_from_response(response, self.clock_ns())
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            value = dict(CLEAR)
        with self.lock:
            self.snapshot = value

    def current(self):
        with self.lock:
            if self.snapshot.get("action") == "target":
                age = (self.clock_ns()-self.snapshot["frame_monotonic_ns"])/1e9
                if not -FUTURE_TOLERANCE_SECONDS <= age <= MAX_FRAME_AGE_SECONDS:
                    self.snapshot = dict(CLEAR)
            return deepcopy(self.snapshot)
