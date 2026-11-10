"""Destination servoing, current target telemetry, recovery holds and obstacles.

Perception is injected here to exercise goal behavior deterministically. Actual
optical flow and appearance matching have separate synthetic-image coverage.
"""
import json
import math
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from bridge_protocol import correlate_response
from navigation import NavigationEngine, simulation_forward_stop_config, zero_axes
from navigation_perception import TrackingLost
from test_bridge_protocol import NOW_NS, native_metadata
from test_navigation import BOX, Clock, ControlledExecutor, configuration


class MovingTarget:
    def __init__(self):
        self.box = BOX
        self.failure = None
        self.starts = 0

    def start(self, frame, box):
        self.box = box
        self.starts += 1

    def update(self, frame):
        if self.failure is not None:
            raise self.failure
        return self.box


class DestinationNavigationTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.executor = ControlledExecutor()
        self.tracker = MovingTarget()
        self.depth = Mock(return_value=np.full((252, 448), 30.0, dtype=np.float32))
        self.config = configuration(image_width=1920, image_height=1080,
                                    fx=1200, fy=1200, cx=960, cy=540)
        self.image = Image.new("RGB", (1920, 1080), "blue")
        self.engine = self.make_engine(self.config)
        self.addCleanup(lambda: self.engine.close())

    def make_engine(self, config):
        return NavigationEngine(config, grounder=Mock(), depth=self.depth,
                                tracker_factory=lambda: self.tracker,
                                clock=self.clock, executor=self.executor)

    def step(self, sequence, **updates):
        self.clock.sleep(0.2)
        metadata = native_metadata(sequence, frame_id=sequence + 1,
                                   frame_monotonic_ns=NOW_NS + int(self.clock.value * 1e9))
        metadata.update(updates)
        return self.engine.handle(metadata, self.image)

    def acquire(self, box=BOX):
        self.assertEqual(self.step(0)["navigation"]["state"], "grounding")
        self.executor.finish(box=box)
        ready = self.step(1)
        self.assertEqual(ready["navigation"]["state"], "ready")
        self.assertEqual(ready["command"], zero_axes(0.35))

    def test_destination_servo_changes_yaw_as_target_crosses_the_image(self):
        self.acquire()
        self.tracker.box = (0.1, 0.25, 0.3, 0.75)
        left = self.step(2)
        self.assertEqual(left["navigation"]["state"], "aligning")
        self.assertLess(left["command"]["yaw"], 0)
        self.assertEqual(left["command"]["forward"], 0)

        self.tracker.box = BOX
        center = self.step(3)
        self.assertEqual(center["navigation"]["state"], "approaching")
        self.assertEqual(center["command"]["forward"], 0.5)
        self.assertAlmostEqual(center["command"]["yaw"], 0)

        self.tracker.box = (0.7, 0.25, 0.9, 0.75)
        right = self.step(4)
        self.assertEqual(right["navigation"]["state"], "aligning")
        self.assertGreater(right["command"]["yaw"], 0)
        self.assertEqual(right["command"]["forward"], 0)
        self.assertEqual(self.executor.submissions, 1)
        self.assertEqual(self.tracker.starts, 1)

    def test_small_bearing_error_corrects_yaw_while_approaching(self):
        self.acquire()
        center = (960 + 1200 * math.tan(math.radians(2))) / 1920
        self.tracker.box = (center - 0.1, 0.25, center + 0.1, 0.75)
        response = self.step(2)
        self.assertEqual(response["navigation"]["state"], "approaching")
        self.assertEqual(response["command"]["forward"], 0.5)
        self.assertGreater(response["command"]["yaw"], 0)
        self.assertLess(response["command"]["yaw"], self.config.max_yaw_deg_s)

    def test_simulator_forward_override_does_not_disable_object_destination_steering(self):
        self.engine.close()
        self.engine = self.make_engine(simulation_forward_stop_config(self.config, 2))
        self.acquire((0.65, 0.25, 0.85, 0.75))
        response = self.step(2)
        self.assertGreater(response["command"]["yaw"], 0)
        self.assertEqual(response["command"]["forward"], 0)
        self.assertNotEqual(response["navigation"].get("behavior"), "sim-forward-stop-test")
        self.assertEqual(response["navigation"]["target"]["visibility"], "visible")
        self.tracker.box = BOX
        response = self.step(3)
        self.assertEqual(response["command"]["forward"], 0.2)
        self.assertEqual(self.executor.submissions, 1)

    def test_target_pixels_are_in_original_camera_resolution_and_distance_is_body_relative(self):
        self.acquire((0.65, 0.25, 0.85, 0.75))
        response = self.step(2)
        target = response["navigation"]["target"]
        self.assertEqual(target["visibility"], "visible")
        self.assertTrue(target["fresh"])
        self.assertEqual(target["image_size_px"], [1920, 1080])
        self.assertEqual(target["center_px"], [1440.0, 540.0])
        self.assertEqual(target["box_px"], [1248.0, 270.0, 1632.0, 810.0])
        self.assertAlmostEqual(target["forward_distance_m"], 19.5, delta=0.01)
        self.assertAlmostEqual(target["distance_m"], math.hypot(19.5, 7.8), delta=0.01)
        self.assertAlmostEqual(target["position_body_m"]["forward"], 19.5, delta=0.01)
        self.assertAlmostEqual(target["position_body_m"]["right"], 7.8, delta=0.01)
        self.assertAlmostEqual(target["position_body_m"]["up"], 0, delta=0.01)
        self.assertAlmostEqual(target["bearing_deg"], math.degrees(math.atan2(480, 1200)), delta=0.01)
        self.assertEqual(target["observed_sequence"], 2)
        self.assertEqual(target["frame_id"], 3)
        self.assertEqual(target["distance_source"], "monocular_metric_estimate")
        self.assertIn("building", target["instruction"].lower())
        self.assertEqual(target["description"], "One blue building")
        json.dumps(response, allow_nan=False)

    def test_target_location_and_distance_are_logged_on_every_successful_update(self):
        self.acquire()
        with self.assertLogs("drone_vla.navigation", level="DEBUG") as logs:
            self.step(2)
            self.tracker.box = (0.65, 0.25, 0.85, 0.75)
            self.depth.return_value = np.full((252, 448), 16.0)
            response = self.step(3)
        observed = [entry for entry in logs.output if '"center_px"' in entry and '"distance_m"' in entry]
        self.assertEqual(len(observed), 2)
        self.assertEqual(response["navigation"]["target"]["center_px"], [1440.0, 540.0])
        self.assertAlmostEqual(response["navigation"]["target"]["forward_distance_m"], 10.4, delta=0.01)

    def test_immediate_front_obstacle_stops_translation_and_yaw_even_with_target_off_center(self):
        self.acquire((0.65, 0.25, 0.85, 0.75))
        self.assertGreater(self.step(2)["command"]["yaw"], 0)
        depth = np.full((252, 448), 30.0, dtype=np.float32)
        depth[115:137, 216:232] = 2.0 / self.config.depth_scale
        self.depth.return_value = depth
        response = self.step(3)
        self.assertEqual(response["navigation"]["state"], "blocked")
        self.assertEqual(response["command"], zero_axes(0.35))
        self.assertIn("obstacle", response["navigation"]["reason"].lower())

    def test_target_standoff_ends_goal_with_zero_motion(self):
        self.acquire()
        self.depth.return_value = np.full((252, 448), 6.0)
        response = self.step(2)
        self.assertEqual(response["navigation"]["state"], "arrived")
        self.assertEqual(response["command"], zero_axes(0.35))
        self.assertAlmostEqual(response["navigation"]["target"]["distance_m"], 3.9, delta=0.01)

    def test_uncertain_tracking_holds_and_retry_requires_fresh_confirmation_before_motion(self):
        self.acquire()
        self.assertGreater(self.step(2)["command"]["forward"], 0)
        self.tracker.failure = TrackingLost("Tracking confidence fell")
        response = self.step(3)
        self.assertEqual(response["navigation"]["state"], "stale")
        self.assertEqual(response["command"], zero_axes(0.35))
        target = response["navigation"]["target"]
        self.assertEqual(target["visibility"], "unconfirmed")
        self.assertIsNone(target["center_px"])
        self.assertIsNone(target["distance_m"])
        self.assertFalse(target["fresh"])
        self.assertNotIn("left the camera", response["navigation"]["reason"].lower())
        self.tracker.failure = None
        response = self.step(4)
        self.assertEqual(response["navigation"]["state"], "ready")
        self.assertEqual(response["command"], zero_axes(0.35))
        response = self.step(5)
        self.assertEqual(response["navigation"]["state"], "approaching")
        self.assertEqual(response["command"]["forward"], 0.5)
        self.assertEqual(self.executor.submissions, 1)

    def test_confirmed_full_exit_is_distinguished_from_uncertain_tracking(self):
        self.acquire()
        self.tracker.failure = TrackingLost("Target outside camera view", status="out_of_view")
        response = self.step(2)
        self.assertEqual(response["navigation"]["state"], "stale")
        self.assertEqual(response["navigation"]["target"]["visibility"], "out_of_view")
        self.assertEqual(response["command"], zero_axes(0.35))

    def test_recovery_deadline_ends_goal_without_reusing_last_motion(self):
        self.engine.close()
        self.engine = self.make_engine(self.config.model_copy(update={"target_recovery_seconds": 0.5}))
        self.acquire()
        self.tracker.failure = TrackingLost("Unable to confirm target")
        self.assertEqual(self.step(2)["navigation"]["state"], "stale")
        self.clock.sleep(0.4)
        response = self.step(3)
        self.assertEqual(response["navigation"]["state"], "lost")
        self.assertEqual(response["command"], zero_axes(0.35))
        self.tracker.failure = None
        self.assertEqual(self.step(4)["navigation"]["state"], "lost")

    def test_cancel_during_recovery_cannot_resume_old_destination(self):
        self.acquire()
        self.tracker.failure = TrackingLost("Unable to confirm target")
        self.assertEqual(self.step(2)["navigation"]["state"], "stale")
        self.engine.cancel()
        self.tracker.failure = None
        response = self.step(3)
        self.assertEqual(response["navigation"]["state"], "error")
        self.assertEqual(response["command"], zero_axes(0.35))

    def test_invalid_depth_never_outputs_motion_or_nonfinite_json(self):
        self.acquire()
        self.depth.return_value = np.full((252, 448), float("nan"))
        response = self.step(2)
        self.assertEqual(response["command"], zero_axes(0.35))
        self.assertEqual(response["navigation"]["state"], "blocked")
        json.dumps(response, allow_nan=False)

    def test_frame_expiry_invalidates_target_freshness_as_well_as_motion(self):
        self.acquire()
        response = self.step(2)
        self.assertTrue(response["navigation"]["target"]["fresh"])
        source_ns = NOW_NS + int(self.clock.value * 1e9)
        metadata = native_metadata(2, frame_id=3, frame_monotonic_ns=source_ns)
        response = correlate_response(response, metadata, now_ns=source_ns + 751_000_000)
        self.assertEqual(response["navigation"]["state"], "stale")
        self.assertEqual(response["command"], zero_axes(0.35))
        self.assertFalse(response["navigation"]["target"]["fresh"])


if __name__ == "__main__":
    unittest.main()
