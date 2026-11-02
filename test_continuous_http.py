"""Real C++ worker <-> HTTP service <-> Unix CLI integration, with no SDK/model calls."""
from concurrent.futures import Future
from contextlib import contextmanager
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from dji_control import DjiClient
from navigation import NavigationEngine
from navigation_perception import GroundedTarget
from navigation_control import run_navigation
from test_navigation import BOX, configuration
from vla_service import VlaServer


def native_binary():
    return Path(os.environ.get("VLA_TEST_BINARY", "/tmp/dji-vla-build/test-vla-bridge"))


class PendingGrounding:
    def __init__(self):
        self.future = Future()
        self.future.set_running_or_notify_cancel()
        self.calls = 0

    def submit(self, *args):
        self.calls += 1
        return self.future

    def finish(self):
        if not self.future.done():
            self.future.set_result(GroundedTarget(found=True, box=BOX, confidence=0.99, reason="Selected building"))

    def shutdown(self, **kwargs):
        self.finish()


class Tracker:
    def start(self, frame, box):
        self.box = box

    def update(self, frame):
        return self.box


class Depth:
    def __init__(self):
        self.block = threading.Event()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.distance = 30.0

    def __call__(self, frame):
        if self.block.is_set():
            self.entered.set()
            self.release.wait(3)
        return np.full((frame.height, frame.width), self.distance)


class RecordingEngine:
    def __init__(self, engine):
        self.engine, self.requests = engine, []

    def handle(self, metadata, image, received_at=None):
        self.requests.append((dict(metadata), list(image.getdata())))
        return self.engine.handle(metadata, image, received_at)

    def cancel(self):
        self.engine.cancel()


class Harness:
    def __init__(self, process, client, grounder, depth, recording):
        self.process, self.client = process, client
        self.grounder, self.depth, self.recording = grounder, depth, recording

    def input(self, value):
        self.process.stdin.write(value+"\n")
        self.process.stdin.flush()

    def line(self):
        if not select.select([self.process.stdout], [], [], 3)[0]:
            raise RuntimeError("Native test driver did not reply")
        return self.process.stdout.readline().strip()

    def stats(self):
        self.input("stats")
        return json.loads(self.line())

    def status(self):
        return self.client.request({"op": "status"})

    def submit(self):
        before = self.client.status()
        return self.client.request(dict(op="navigate", text="Approach the blue building", goal_id="integration-goal",
                                        max_seconds=10, interval_seconds=0.1, duration_seconds=0.35,
                                        session=before["session"], ticket=before["ticket"]))


def wait_until(predicate, timeout=3):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("Continuous bridge condition timed out")


@contextmanager
def harness():
    grounder, depth = PendingGrounding(), Depth()
    engine = NavigationEngine(configuration(image_width=2, image_height=1, fx=1.4, fy=1.4, cx=1, cy=0.5),
                              grounder=lambda *_: None, depth=depth, tracker_factory=Tracker, executor=grounder)
    recording = RecordingEngine(engine)
    with tempfile.TemporaryDirectory(prefix="vla-http-", dir="/tmp") as directory, \
         patch("vla_service.VlaHandler.log_message", lambda *args: None), \
         VlaServer(("127.0.0.1", 0), navigation=recording) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        socket_path = str(Path(directory)/"control.sock")
        process = subprocess.Popen([str(native_binary()), "--integration", f"http://127.0.0.1:{server.server_port}/plan", socket_path],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, bufsize=1)
        instance = Harness(process, DjiClient(socket_path), grounder, depth, recording)
        try:
            if instance.line() != "READY":
                raise RuntimeError("Native test driver startup failed")
            wait_until(lambda: instance.status().get("frame_id", 0) > 0)
            yield instance
        finally:
            depth.release.set()
            grounder.finish()
            if process.poll() is None:
                try:
                    instance.input("quit")
                except BrokenPipeError:
                    pass
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            stderr = process.stderr.read()
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()
            server.shutdown()
            thread.join(timeout=2)
            engine.close()
            if process.returncode != 0:
                raise AssertionError("Native integration driver failed: "+stderr)


@unittest.skipUnless(native_binary().is_file(), "Build standalone native bridge tests first")
class ContinuousHttpTests(unittest.TestCase):
    def test_native_arbitration_and_watchdog_regressions(self):
        result = subprocess.run([str(native_binary())], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_actual_terminal_client_submits_once_and_observes_standoff_completion(self):
        with harness() as h:
            h.grounder.finish()
            original_request = h.client.request
            operations = []
            def request(payload):
                operations.append(payload["op"])
                return original_request(payload)
            h.client.request = request
            def observe(line):
                row = json.loads(line)
                if row.get("state") == "approaching":
                    h.depth.distance = 6  # Learned-depth estimate crosses configured stand-off.
            result = run_navigation(h.client, "Approach the blue building", max_seconds=5, emit=observe)
            self.assertEqual(operations.count("navigate"), 1)
            self.assertNotIn("instruction", operations)
            self.assertEqual(result[-1]["state"], "arrived")
            self.assertEqual(h.grounder.calls, 1)
            self.assertEqual(h.status()["command"]["forward"], 0)

    def test_one_goal_grounds_once_then_updates_and_holds_command_during_next_inference(self):
        with harness() as h:
            accepted = h.submit()
            wait_until(lambda: len(h.recording.requests) >= 2)
            self.assertEqual(h.stats()["moving_dispatches"], 0)
            self.assertEqual(h.grounder.calls, 1)
            h.grounder.finish()
            wait_until(lambda: h.status()["command"]["forward"] == 0.5)
            h.depth.block.set()
            self.assertTrue(h.depth.entered.wait(2))
            before = h.stats()
            time.sleep(0.08)  # No inference reply during this interval; the valid setpoint must still dispatch.
            after = h.stats()
            self.assertGreater(after["moving_dispatches"], before["moving_dispatches"]+3)
            metadata = [request[0] for request in h.recording.requests]
            self.assertGreaterEqual(len(metadata), 3)
            self.assertEqual(len({request["request_id"] for request in metadata}), 1)
            self.assertEqual(metadata[0]["request_id"], accepted["request_id"])
            self.assertEqual(len({request["frame_id"] for request in metadata}), len(metadata))
            self.assertTrue(all(request["bridge_protocol"] == 2 for request in metadata))
            self.assertTrue(all(pixels == [(255, 0, 0), (0, 255, 0)] for _, pixels in h.recording.requests))
            h.client.request(dict(op="cancel", session=accepted["session"], request_id=accepted["request_id"], goal_id="integration-goal"))
            h.depth.release.set()
            wait_until(lambda: h.status()["state"] == "cancelled")
            self.assertEqual(h.status()["command"]["forward"], 0)

    def test_frozen_camera_expires_movement_without_any_more_terminal_instructions(self):
        with harness() as h:
            h.submit()
            h.grounder.finish()
            wait_until(lambda: h.status()["command"]["forward"] == 0.5)
            h.input("freeze")
            time.sleep(0.6)
            first = h.stats()
            self.assertEqual(first["axes"]["forward"], 0)
            self.assertEqual(h.status()["navigation"]["state"], "stale")
            time.sleep(0.1)
            self.assertEqual(h.stats()["moving_dispatches"], first["moving_dispatches"])

    def test_operator_takeover_discards_late_http_result_and_centred_heartbeat_preserves_goal(self):
        with harness() as h:
            accepted = h.submit()
            h.input("center")
            self.assertEqual(h.status()["session"], accepted["session"])
            h.grounder.finish()
            wait_until(lambda: h.status()["command"]["forward"] == 0.5)
            h.depth.block.set()
            self.assertTrue(h.depth.entered.wait(2))
            h.input("takeover")
            wait_until(lambda: h.status()["session"] != accepted["session"])
            self.assertFalse(h.status()["eligible"])
            h.input("center")
            wait_until(lambda: h.status()["eligible"])
            stopped = h.stats()["moving_dispatches"]
            h.depth.release.set()
            time.sleep(0.25)
            self.assertEqual(h.stats()["moving_dispatches"], stopped)
            self.assertEqual(h.status()["command"]["forward"], 0)


if __name__ == "__main__":
    unittest.main()
