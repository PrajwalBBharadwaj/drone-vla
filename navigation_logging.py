"""Compact operator-facing navigation text; full observations stay on the wire."""
import math


def _number(value, decimals=2, signed=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        return "-"
    return format(value, ("+" if signed else "") + f".{decimals}f")


def _pixels(values, length):
    if not isinstance(values, (list, tuple)) or len(values) != length:
        return "-"
    return "[" + ",".join(_number(value, 0) for value in values) + "]"


def compact_navigation_line(sample):
    """Show the command and target measurement without bridge/session metadata."""
    if "hold_result" in sample:
        return "hold: " + str(sample["hold_result"]).replace("\n", " ")
    command = sample.get("command") or {}
    target = sample.get("target") or {}
    timings = sample.get("service_timings_ms", sample.get("timings_ms")) or {}
    confidence = target.get("tracking_confidence")
    if confidence is None:
        confidence = target.get("grounding_confidence")
    axes = " ".join(f"{label}={_number(command.get(axis), signed=True)}" for axis, label in
                    (("forward", "f"), ("right", "r"), ("up", "u"), ("yaw", "yaw")))
    parts = [str(sample.get("state", "unknown")), f"cmd({axes})"]
    if target:
        parts.extend((f"bbox={_pixels(target.get('box_px'), 4)}",
                      f"center={_pixels(target.get('center_px'), 2)}",
                      f"conf={_number(confidence, 3)}",
                      f"range={_number(target.get('distance_m'))}m",
                      f"bearing={_number(target.get('bearing_deg'), signed=True)}deg",
                      str(target.get("visibility", "unknown"))))
        if target.get("fresh") is False:
            parts.append("target=stale")
    if "service_ms" in timings:
        parts.append(f"{_number(timings['service_ms'], 0)}ms")
    if sample.get("state") not in {"aligning", "approaching"} and sample.get("reason"):
        parts.append(str(sample["reason"]).replace("\n", " ")[:180])
    return " ".join(parts)
