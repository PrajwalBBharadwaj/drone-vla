"""Submit one persistent navigation goal; monitor native-owned camera updates."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
import uuid

from dji_control import DjiClient, execute
from navigation import RESULT_PREFIX, encode_message


class OwnedClient:
    """Remember acceptance even when Ctrl-C interrupts execute's status polling."""
    def __init__(self, client):
        self.client = client
        self.session = None
        self.request_id = None

    def status(self):
        return self.client.status()

    def request(self, payload):
        response = self.client.request(payload)
        if payload.get("op") in {"instruction", "navigate"}:
            self.session, self.request_id = payload["session"], response["request_id"]
        return response

    def hold_if_owned(self):
        if self.request_id is None:
            return
        # Recheck ownership inside execute as well, before obtaining a new ticket.
        status = self.status()
        if status["session"] == self.session and status["request_id"] == self.request_id:
            execute(self, "Hover", default_duration=0.1, timeout=5,
                    expected_session=self.session, expected_request_id=self.request_id, poll_interval=0.02)

    def cancel_if_owned(self, goal_id):
        if self.request_id is None:
            return
        status = self.status()
        navigation = status.get("navigation", {})
        if (status["session"] == self.session and status["request_id"] == self.request_id
                and navigation.get("state") not in {"arrived", "blocked", "lost", "error", "cancelled"}):
            self.client.request(dict(op="cancel", session=self.session, request_id=self.request_id, goal_id=goal_id))


def run_legacy_navigation(client, text, max_seconds=120, interval=0.1, duration=0.2,
                          emit=print, clock=time.monotonic, sleep=time.sleep):
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 300:
        raise ValueError("Navigation deadline must be in (0, 300] seconds")
    if not math.isfinite(interval) or not 0.05 <= interval <= 0.5:
        raise ValueError("Frame interval must be in [0.05, 0.5] seconds")
    if not math.isfinite(duration) or not 0.1 <= duration <= 0.5:
        raise ValueError("Navigation movement lease must be in [0.1, 0.5] seconds")
    owned = OwnedClient(client)
    initial = owned.status()
    session = initial["session"]
    expected = initial["request_id"]
    goal_id = uuid.uuid4().hex
    started = clock()
    samples = []
    sequence = 0
    previous_accepted = None
    try:
        while clock()-started < max_seconds:
            text_message = encode_message(op="start" if sequence == 0 else "step", goal_id=goal_id,
                                          owner=session, sequence=sequence, **({"text": text} if sequence == 0 else {}))
            sent = clock()
            status = execute(owned, text_message, duration, timeout=35,
                             expected_session=session, expected_request_id=expected, poll_interval=0.02)
            accepted = clock()
            expected = status["request_id"]
            reason = status.get("override_reason", "")
            if not reason.startswith(RESULT_PREFIX):
                raise RuntimeError("DJI bridge did not return navigation status; check its response forwarding")
            info = json.loads(reason[len(RESULT_PREFIX):])
            if info.get("goal_id") != goal_id or info.get("sequence") != sequence:
                raise RuntimeError("Navigation response does not match the submitted goal/sequence")
            sample = dict(sequence=sequence, state=info["state"],
                          round_trip_ms=round((accepted-sent)*1000, 2),
                          update_interval_ms=None if previous_accepted is None else round((accepted-previous_accepted)*1000, 2),
                          since_goal_ms=round((accepted-started)*1000, 2),
                          service_timings_ms=info.get("timings_ms", {}))
            previous_accepted = accepted
            samples.append(sample)
            emit(json.dumps(dict(sample, reason=info["reason"], command=status.get("command")), allow_nan=False))
            if info["state"] in {"arrived", "blocked", "lost", "error"}:
                return samples
            # execute returns on manual-loop acceptance, not at lease expiry. Do
            # not replace a still-executing command and mistake it for completion.
            sleep(max(interval, status.get("command", {}).get("duration_seconds", duration)))
            sequence += 1
        raise RuntimeError("Navigation deadline exceeded; holding if this client still owns control")
    finally:
        try:
            owned.hold_if_owned()
        except (OSError, ValueError, RuntimeError) as error:
            # No command after mode loss/operator takeover; existing leases expire.
            emit(json.dumps({"hold_result": str(error)}))


def run_navigation(client, text, max_seconds=120, interval=0.1, duration=0.35,
                   emit=print, clock=time.monotonic, sleep=time.sleep):
    if not isinstance(text, str) or not text.strip() or len(text.encode()) > 600:
        raise ValueError("Navigation goal requires nonblank text (max 600 UTF-8 bytes)")
    if not math.isfinite(max_seconds) or not 0.1 <= max_seconds <= 300:
        raise ValueError("Navigation deadline must be in [0.1, 300] seconds")
    if not math.isfinite(interval) or not 0.05 <= interval <= 0.5:
        raise ValueError("Navigation frame interval must be in [0.05, 0.5] seconds")
    if not math.isfinite(duration) or not 0.1 <= duration <= 0.35:
        raise ValueError("Continuous movement lease must be in [0.1, 0.35] seconds")
    owned = OwnedClient(client)
    initial = owned.status()
    if initial.get("bridge_protocol") != 2 or initial.get("continuous_navigation") is not True:
        raise RuntimeError("Continuous navigation requires a rebuilt DJI_VLA_CONTINUOUS_V2 binary; --legacy-steps explicitly selects the old step loop")
    session = initial["session"]
    goal_id = uuid.uuid4().hex
    # Validate the actual envelope size before submitting anything to DJI.
    encode_message(op="start", goal_id=goal_id, owner=session, sequence=0, text=text)
    started = clock()
    samples, last_signature = [], None
    try:
        accepted = owned.request(dict(op="navigate", text=text, goal_id=goal_id,
                                      duration_seconds=duration, interval_seconds=interval, max_seconds=max_seconds,
                                      session=session, ticket=initial["ticket"]))
        request_id = accepted["request_id"]
        while clock()-started < max_seconds:
            sent = clock()
            status = owned.status()
            received = clock()
            if status["session"] != session:
                raise RuntimeError("Manual session changed; navigation stopped")
            if status["request_id"] != request_id:
                raise RuntimeError("Operator or another instruction superseded navigation")
            info = status.get("navigation", {})
            if info.get("goal_id") != goal_id:
                raise RuntimeError("Native navigation goal does not match this client")
            state = info.get("state")
            command = status.get("command", {})
            signature = (info.get("sequence"), state, tuple(command.get(key) for key in ("forward", "right", "up", "yaw")))
            if signature != last_signature:
                sample = dict(sequence=info.get("sequence"), state=state, bridge_protocol=2,
                              status_round_trip_ms=round((received-sent)*1000, 2),
                              since_goal_ms=round((received-started)*1000, 2),
                              service_timings_ms=info.get("timings_ms", {}),
                              command=command, reason=info.get("reason", ""))
                samples.append(sample)
                emit(json.dumps(sample, allow_nan=False))
                last_signature = signature
            if state in {"arrived", "blocked", "lost", "error", "cancelled"}:
                if state == "error":
                    raise RuntimeError(info.get("reason") or status.get("inference_error") or "Navigation failed")
                return samples
            if status.get("state") == "error":
                raise RuntimeError(status.get("inference_error") or "Native navigation failed")
            sleep(min(0.1, interval))
        raise RuntimeError("Navigation deadline exceeded; cancelling if this client still owns the goal")
    finally:
        try:
            owned.cancel_if_owned(goal_id)
        except (OSError, ValueError, RuntimeError) as error:
            emit(json.dumps({"hold_result": str(error)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command")
    parser.add_argument("--socket", default=os.environ.get("DJI_VLA_SOCKET", "/run/dji-vla/control.sock"))
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--duration", type=float, help="Command lease in seconds (continuous default/max 0.35; legacy default 0.2)")
    parser.add_argument("--metrics", type=Path, help="Write JSON timing samples, including on interruption")
    parser.add_argument("--legacy-steps", action="store_true", help="Explicitly use terminal-driven steps with the old bridge")
    args = parser.parse_args()
    records = []
    def emit(line):
        print(line, flush=True)
        records.append(json.loads(line))
    try:
        runner = run_legacy_navigation if args.legacy_steps else run_navigation
        duration = args.duration if args.duration is not None else (0.2 if args.legacy_steps else 0.35)
        runner(DjiClient(args.socket), args.command, args.max_seconds, args.interval, duration, emit)
    except KeyboardInterrupt:
        print("Navigation interrupted", flush=True)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Navigation stopped: {error}\n")
    finally:
        if args.metrics:
            args.metrics.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        times = [row["round_trip_ms"] for row in records if "round_trip_ms" in row
                 and row.get("state") in {"aligning", "approaching"}]
        if times:
            ordered = sorted(times)
            print(json.dumps({"navigation_updates": len(times), "round_trip_p50_ms": statistics.median(times),
                              "round_trip_p95_ms": ordered[math.ceil(len(times)*0.95)-1],
                              "round_trip_max_ms": max(times),
                              "first_movement_accepted_ms": next(row["since_goal_ms"] for row in records
                                                               if row.get("state") in {"aligning", "approaching"})}))
        active = [row for row in records if row.get("bridge_protocol") == 2
                  and any(row.get("command", {}).get(axis, 0) for axis in ("forward", "right", "up", "yaw"))]
        ages = [row["service_timings_ms"]["decoded_frame_age_ms"] for row in active
                if "decoded_frame_age_ms" in row.get("service_timings_ms", {})]
        if active:
            summary = dict(first_nonzero_command_observed_ms=active[0]["since_goal_ms"], observed_motion_updates=len(active))
            if ages:
                summary["service_decoded_frame_age_p95_ms"] = sorted(ages)[math.ceil(len(ages)*0.95)-1]
            print(json.dumps(summary))


if __name__ == "__main__":
    main()
