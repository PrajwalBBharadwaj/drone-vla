"""Actual C++ video-overlay subscriber against the Python service's HTTP cache."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import time
import unittest

from navigation import encode_message
from target_overlay import monotonic_ns
from test_target_overlay import http_overlay, request


BINARY = Path(os.environ.get("VLA_OVERLAY_TEST_BINARY", "/tmp/dji-target-overlay-tests/test-target-overlay"))
RGB = bytes((17, 34, 51))*(1920*1080)


def publish_frame(server, sequence):
    metadata = dict(bridge_protocol=2, session="manual1", goal_id="goal1", request_id="video-request",
                    frame_id=sequence+1, frame_monotonic_ns=monotonic_ns(), default_duration=0.35,
                    width=1920, height=1080, format="RGB",
                    text=encode_message(op="step", owner="manual1", goal_id="goal1", sequence=sequence))
    header = json.dumps(metadata).encode()
    status, response = request(server, "POST", "/plan", struct.pack("!I", len(header))+header+RGB)
    if status != 200:
        raise AssertionError(response)


class Subscriber:
    def __init__(self, process):
        self.process = process

    def command(self, text):
        self.process.stdin.write(text+"\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 2)[0]:
            raise AssertionError("Native overlay subscriber did not respond")
        return self.process.stdout.readline().strip()

    def current(self):
        return json.loads(self.command("current"))

    def wait_for_visibility(self, visible, timeout=0.6):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            snapshot = self.current()
            if snapshot["visible"] is visible:
                return snapshot
            time.sleep(0.01)
        raise AssertionError(f"Native overlay did not become visible={visible}")


@contextmanager
def subscriber(server):
    process = subprocess.Popen([str(BINARY), "--poll", f"http://127.0.0.1:{server.server_port}/overlay"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, bufsize=1)
    client = Subscriber(process)
    try:
        if not select.select([process.stdout], [], [], 2)[0] or process.stdout.readline().strip() != "READY":
            raise AssertionError("Native overlay subscriber failed to start")
        yield client
    finally:
        if process.poll() is None:
            process.stdin.write("quit\n")
            process.stdin.flush()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        stderr = process.stderr.read()
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()
        if process.returncode != 0:
            raise AssertionError(stderr)


@unittest.skipUnless(BINARY.is_file(), "Build dji-lte/test/target_overlay first")
class NativeOverlayHttpTests(unittest.TestCase):
    def test_real_full_hd_metadata_projection_clear_epoch_and_expiry(self):
        with http_overlay() as (server, _), subscriber(server) as native:
            self.assertFalse(native.current()["visible"])
            publish_frame(server, 1)
            snapshot = native.wait_for_visibility(True)
            self.assertEqual(snapshot["box_px"], [480, 216, 1439, 863])
            self.assertEqual(snapshot["center_px"], [960, 540])
            self.assertEqual(snapshot["goal_id"], "goal1")
            self.assertEqual(snapshot["request_id"], "video-request")

            # Emulate native takeover/source epoch invalidation. Continued GETs
            # of the same still-fresh Python observation cannot revive it.
            self.assertEqual(native.command("invalidate"), "OK")
            time.sleep(0.15)
            self.assertFalse(native.current()["visible"])
            publish_frame(server, 2)
            self.assertEqual(native.wait_for_visibility(True)["sequence"], 2)

            server.overlay_state.clear()
            native.wait_for_visibility(False)
            publish_frame(server, 3)
            native.wait_for_visibility(True)
            # Neither repeated cache reads nor HTTP polls renew source age.
            native.wait_for_visibility(False, timeout=1.1)


if __name__ == "__main__":
    unittest.main()
