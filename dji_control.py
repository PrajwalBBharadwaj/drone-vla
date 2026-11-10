"""Submit a natural-language instruction to dji-lte, which supplies its camera frame to VLA."""

import argparse
import json
import math
import os
from pathlib import Path
import socket
import tempfile
import time

MAX_RESPONSE_BYTES = 65536


def joystick_command(command, default_duration=1.0):
    """PSDK body coordinates: forward/right/up positive, clockwise yaw positive."""
    if not command.valid:
        raise ValueError(command.reason or "Planner rejected command")

    def signed(direction, speed, positive, negative):
        value = getattr(direction, "value", direction)
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("Non-finite or negative speed")
        if value == "none" and speed == 0:
            return 0.0
        if value not in (positive, negative) or speed <= 0:
            raise ValueError("Invalid axis direction/speed pair")
        return speed if value == positive else -speed

    result = {
        "op": "command",
        "forward": signed(command.longitudinal_direction, command.longitudinal_speed_mps, "forward", "backward"),
        "right": signed(command.lateral_direction, command.lateral_speed_mps, "right", "left"),
        "up": signed(command.vertical_direction, command.vertical_speed_mps, "up", "down"),
        "yaw": signed(command.yaw_direction, command.yaw_speed_deg_s, "clockwise", "counterclockwise"),
        "duration_seconds": command.duration_seconds if command.duration_seconds is not None else default_duration,
    }
    duration = result["duration_seconds"]
    if (not math.isfinite(duration) or not 0 < duration <= 2
            or math.hypot(result["forward"], result["right"]) > 2
            or abs(result["up"]) > 1 or abs(result["yaw"]) > 30):
        raise ValueError("VLA limits: horizontal 2 m/s, vertical 1 m/s, yaw 30 deg/s, duration 2 s")
    return result


class DjiClient:
    def __init__(self, path, timeout=1.0):
        self.path = path
        self.timeout = timeout

    def request(self, payload):
        # A private directory prevents other users from replacing the reply socket.
        with tempfile.TemporaryDirectory(prefix="dji-vla-") as directory:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
                # Target status can exceed the small default Unix datagram
                # queue on some hosts. Match the protocol's receive capacity.
                client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, MAX_RESPONSE_BYTES)
                client.bind(str(Path(directory) / "client.sock"))
                client.settimeout(self.timeout)
                client.connect(self.path)
                client.send(json.dumps(payload, allow_nan=False).encode())
                # Status includes both structured navigation telemetry and its
                # serialized override reason. Read one extra byte so an
                # oversized datagram is rejected rather than silently parsed
                # as a truncated status response.
                raw = client.recv(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise RuntimeError("dji-lte response exceeds the 65536-byte limit")
                response = json.loads(raw)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "dji-lte rejected request"))
        return response

    def status(self):
        result = self.request({"op": "status"})
        if not result.get("eligible"):
            raise RuntimeError("Requires airborne standby/manual mode with operator joystick at zero")
        return result


def execute(client, text, default_duration=1.0, timeout=35.0,
            expected_session=None, expected_request_id=None, poll_interval=0.1):
    if not text.strip() or len(text.encode()) > 1000:
        raise ValueError("Instruction requires text (max 1000 bytes)")
    if not math.isfinite(default_duration) or not 0 < default_duration <= 2:
        raise ValueError("Duration must be in (0, 2] seconds")
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("Wait timeout must be finite and nonnegative; use 0 for no limit")
    before = client.status()
    if expected_session is not None and before["session"] != expected_session:
        raise RuntimeError("Manual session changed; navigation stopped")
    if expected_request_id is not None and before["request_id"] != expected_request_id:
        raise RuntimeError("Operator or another instruction superseded navigation")
    accepted = client.request({"op": "instruction", "text": text, "duration_seconds": default_duration,
                               "session": before["session"], "ticket": before["ticket"]})
    deadline = time.monotonic() + timeout if timeout else None
    while deadline is None or time.monotonic() < deadline:
        status = client.status()
        if before["session"] != status["session"]:
            raise RuntimeError("Manual session or operator input changed; VLA result discarded")
        if status["request_id"] != accepted["request_id"]:
            raise RuntimeError("VLA instruction was superseded")
        if status["state"] == "error":
            raise RuntimeError(status["inference_error"] or "VLA inference failed")
        if status["state"] in ("executing", "completed"):
            return status
        time.sleep(poll_interval)
    raise RuntimeError("Timed out waiting for dji-lte's VLA result")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", help="Natural-language motion command")
    parser.add_argument("--socket", default=os.environ.get("DJI_VLA_SOCKET"))
    parser.add_argument("--duration", type=float, default=1.0, help="Fallback duration in seconds (max 2)")
    parser.add_argument("--timeout", type=float, default=35.0,
                        help="CLI wait in seconds; 0 waits indefinitely. DJI's inference deadline is separate.")
    args = parser.parse_args()
    if not args.socket:
        parser.error("Set --socket or DJI_VLA_SOCKET")
    try:
        print(json.dumps(execute(DjiClient(args.socket), args.command, args.duration, args.timeout), indent=2))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"VLA control failed: {error}\n")


if __name__ == "__main__":
    main()
