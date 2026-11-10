"""Actual HTTP frames, OpenCV target motion and correlated destination commands.

The depth backend is injected; there are no model downloads or aircraft calls.
This exercises the Python wire endpoint, not the unavailable native SDK binary.
"""
from concurrent.futures import Future
from contextlib import contextmanager
import http.client
import json
import struct
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from navigation import NavigationEngine, encode_message, simulation_forward_stop_config, zero_axes
from navigation_perception import GroundedTarget
from test_navigation import configuration
from test_target_tracking import box, scene
from vla_service import VlaServer


class InitialGrounding:
    def __init__(self):
        self.calls = 0

    def submit(self, grounder, instruction, image):
        self.calls += 1
        result = Future()
        result.set_result(GroundedTarget(found=True, box=box(300, 80), confidence=0.99,
                                         reason="Selected textured building on the right"))
        return result

    def shutdown(self, **kwargs):
        pass


@contextmanager
def live_endpoint(sim_test=False):
    grounding = InitialGrounding()
    depth = np.full((252, 448), 30.0, dtype=np.float32)
    config = configuration()
    if sim_test:
        config = simulation_forward_stop_config(config, 2)
    engine = NavigationEngine(config, grounder=lambda *_: None,
                              depth=lambda frame: depth, executor=grounding)
    with patch("vla_service.VlaHandler.log_message", lambda *args: None), \
         VlaServer(("127.0.0.1", 0), navigation=engine) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server, grounding, depth
        finally:
            server.shutdown()
            thread.join(timeout=2)
            engine.close()


def post_frame(server, sequence, image):
    envelope = dict(op="start" if sequence == 0 else "step", goal_id="destination", owner="manual", sequence=sequence)
    if sequence == 0:
        envelope["text"] = "Approach the textured building on the right"
    metadata = dict(bridge_protocol=2, session="manual", goal_id="destination", frame_id=sequence+1,
                    frame_monotonic_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC),
                    request_id="destination-request", default_duration=0.35,
                    width=image.width, height=image.height, format="RGB", text=encode_message(**envelope))
    header = json.dumps(metadata).encode()
    payload = struct.pack("!I", len(header))+header+image.tobytes()
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request("POST", "/plan", payload, {"Content-Type": "application/octet-stream"})
        response = connection.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise AssertionError(result)
        for field in ("session", "goal_id", "frame_id", "frame_monotonic_ns", "request_id"):
            if result[field] != metadata[field]:
                raise AssertionError(f"Mismatched {field}")
        return result
    finally:
        connection.close()


class DestinationHttpTests(unittest.TestCase):
    def test_actual_camera_target_changes_steering_then_obstacle_stops_all_axes(self):
        with live_endpoint() as (server, grounding, depth):
            self.assertEqual(post_frame(server, 0, scene(300, 80))["navigation"]["state"], "grounding")
            ready = post_frame(server, 1, scene(300, 80))
            self.assertEqual(ready["command"], zero_axes(0.35))

            right = post_frame(server, 2, scene(300, 80))
            self.assertGreater(right["command"]["yaw"], 0)
            self.assertEqual(right["command"]["forward"], 0)
            self.assertAlmostEqual(right["navigation"]["target"]["center_px"][0], 348, delta=1)

            centered = post_frame(server, 3, scene(176, 80))
            self.assertEqual(centered["navigation"]["state"], "approaching")
            self.assertEqual(centered["command"]["forward"], 0.5)
            self.assertAlmostEqual(centered["command"]["yaw"], 0, delta=0.1)

            left = post_frame(server, 4, scene(40, 80))
            self.assertLess(left["command"]["yaw"], 0)
            self.assertEqual(left["command"]["forward"], 0)
            self.assertTrue(left["navigation"]["target"]["fresh"])
            self.assertAlmostEqual(left["navigation"]["target"]["forward_distance_m"], 19.5, delta=0.01)

            depth[96:156, 212:236] = 2.5/0.65
            stopped = post_frame(server, 5, scene(40, 80))
            self.assertEqual(stopped["navigation"]["state"], "blocked")
            self.assertEqual(stopped["command"], zero_axes(0.35))
            self.assertEqual(grounding.calls, 1)

    def test_object_target_cannot_activate_simulator_speed_override_over_http(self):
        with live_endpoint(sim_test=True) as (server, grounding, _):
            post_frame(server, 0, scene(300, 80))
            post_frame(server, 1, scene(300, 80))
            turning = post_frame(server, 2, scene(300, 80))
            self.assertGreater(turning["command"]["yaw"], 0)
            self.assertNotIn("behavior", turning["navigation"])
            centered = post_frame(server, 3, scene(176, 80))
            self.assertEqual(centered["command"]["forward"], 0.2)
            self.assertNotIn("behavior", centered["navigation"])
            self.assertEqual(grounding.calls, 1)


if __name__ == "__main__":
    unittest.main()
