"""Correlation and freshness checks for the continuous native bridge.

Both processes use CLOCK_MONOTONIC. The wire timestamp marks RGB decode, not
camera exposure. Legacy one-shot requests intentionally retain their protocol.
"""
import json
import math
import time

PROTOCOL_VERSION = 2
MAX_FRAME_AGE_SECONDS = 0.75
MAX_COMMAND_LEASE_SECONDS = 0.35
FUTURE_TOLERANCE_SECONDS = 0.02
CONTEXT_FIELDS = ("bridge_protocol", "session", "goal_id", "frame_id", "frame_monotonic_ns")
TERMINAL_STATES = {"arrived", "blocked", "lost", "error", "cancelled"}
HOLD_STATES = TERMINAL_STATES | {"grounding", "acquiring", "ready", "stale"}
ACTIVE_STATES = {"approaching", "aligning"}


def frame_age_seconds(metadata, now_ns=None):
    # Python's monotonic clock can use mach_absolute_time on older macOS Python,
    # whereas the native wire explicitly uses clock_gettime(CLOCK_MONOTONIC).
    now = time.clock_gettime_ns(time.CLOCK_MONOTONIC) if now_ns is None else now_ns
    return (now - metadata["frame_monotonic_ns"])/1_000_000_000


def validate_context(metadata, now_ns=None):
    if "bridge_protocol" not in metadata:
        return None
    if type(metadata["bridge_protocol"]) is not int or metadata["bridge_protocol"] != PROTOCOL_VERSION:
        raise ValueError("Unsupported native bridge protocol")
    for key, maximum in (("session", 128), ("goal_id", 64)):
        if not isinstance(metadata.get(key), str) or not 0 < len(metadata[key]) <= maximum:
            raise ValueError(f"Invalid bridge {key}")
    for key in ("frame_id", "frame_monotonic_ns"):
        if type(metadata.get(key)) is not int or not 0 < metadata[key] < 2**63:
            raise ValueError(f"Invalid bridge {key}")
    text = metadata.get("text")
    if not isinstance(text, str) or not text.startswith("@nav1:"):
        raise ValueError("Continuous bridge requires a navigation envelope")
    try:
        message = json.loads(text[len("@nav1:"):])
        if not isinstance(message, dict) or message.get("owner") != metadata["session"] or message.get("goal_id") != metadata["goal_id"]:
            raise ValueError("Navigation goal/session does not match the native frame context")
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Invalid navigation envelope") from error
    age = frame_age_seconds(metadata, now_ns)
    if age < -FUTURE_TOLERANCE_SECONDS:
        raise ValueError("Native frame timestamp is in the future; clocks must use CLOCK_MONOTONIC")
    if age > MAX_FRAME_AGE_SECONDS:
        raise ValueError("Native camera frame is stale")
    return {key: metadata[key] for key in CONTEXT_FIELDS}


def validate_axes(command):
    if not isinstance(command, dict):
        raise ValueError("Missing navigation command")
    for key in ("forward", "right", "up", "yaw", "duration_seconds"):
        value = command.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Navigation command must contain finite numeric axes and duration")
    if (math.hypot(command["forward"], command["right"]) > 2
            or abs(command["up"]) > 1 or abs(command["yaw"]) > 30
            or not 0 < command["duration_seconds"] <= 0.5):
        raise ValueError("Continuous navigation command exceeds native limits")


def correlate_response(response, metadata, now_ns=None):
    # Input context was freshness-checked before inference. It can legitimately
    # have aged while inference ran, so do not re-run arrival validation here.
    message = json.loads(metadata["text"][len("@nav1:"):])
    navigation = response.get("navigation")
    if (response.get("ok") is not True or response.get("request_id") != metadata["request_id"]
            or not isinstance(navigation, dict)
            or navigation.get("goal_id") != metadata["goal_id"]
            or type(navigation.get("sequence")) is not int
            or navigation["sequence"] != message["sequence"]):
        raise ValueError("Navigation response does not match the native request")
    state = navigation.get("state")
    if state not in HOLD_STATES | ACTIVE_STATES:
        raise ValueError("Unknown navigation state")
    command = response.get("command")
    validate_axes(command)
    age = frame_age_seconds(metadata, now_ns)
    if age < -FUTURE_TOLERANCE_SECONDS or age > MAX_FRAME_AGE_SECONDS:
        navigation["state"], navigation["reason"] = "stale", "Supporting camera frame exceeded the native freshness limit"
        state = "stale"
    if state in HOLD_STATES:
        response["command"] = dict(forward=0.0, right=0.0, up=0.0, yaw=0.0,
                                   duration_seconds=command["duration_seconds"])
    navigation.setdefault("timings_ms", {})["decoded_frame_age_ms"] = round(age*1000, 2)
    response.update({key: metadata[key] for key in CONTEXT_FIELDS})
    response["override_reason"] = "navigation:" + json.dumps(navigation, separators=(",", ":"), allow_nan=False)
    return response
