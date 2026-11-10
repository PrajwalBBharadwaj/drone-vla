"""Explicit two-ended HITL speed override; normal motion limits remain intact."""
import time
import unittest
from unittest.mock import Mock

from navigation import NavigationEngine, encode_message, simulation_forward_stop_config
from test_bridge_protocol import native_metadata
from test_continuous_http import harness, native_binary, wait_until
from test_navigation import Clock, ControlledExecutor, configuration, texture


class SimulationMarkerTests(unittest.TestCase):
    def test_object_goals_never_emit_the_simulation_behavior_marker(self):
        for enabled in (False, True):
            with self.subTest(sim_profile=enabled):
                config = configuration()
                if enabled:
                    config = simulation_forward_stop_config(config)
                executor = ControlledExecutor()
                engine = NavigationEngine(config, grounder=Mock(), depth=Mock(), clock=Clock(), executor=executor)
                try:
                    result = engine.handle(native_metadata(0), texture())
                    self.assertEqual(result["navigation"]["state"], "grounding")
                    self.assertNotIn("behavior", result["navigation"])
                    self.assertFalse(any(result["command"][key] for key in ("forward", "right", "up", "yaw")))
                finally:
                    engine.close()

    def test_literal_forward_emits_the_marker_only_with_explicit_test_profile(self):
        for enabled in (False, True):
            with self.subTest(sim_profile=enabled):
                config = configuration()
                if enabled:
                    config = simulation_forward_stop_config(config)
                executor = ControlledExecutor()
                engine = NavigationEngine(config, grounder=Mock(), depth=Mock(), clock=Clock(), executor=executor)
                try:
                    metadata = native_metadata(0, text=encode_message(op="start", goal_id="goal1", owner="manual1",
                                                                     sequence=0, text="move forward"))
                    result = engine.handle(metadata, texture())
                    self.assertEqual(result["navigation"]["state"], "ready" if enabled else "grounding")
                    self.assertEqual(executor.submissions, 0 if enabled else 1)
                    if enabled:
                        self.assertEqual(result["navigation"]["behavior"], "sim-forward-stop-test")
                    else:
                        self.assertNotIn("behavior", result["navigation"])
                    self.assertFalse(any(result["command"][key] for key in ("forward", "right", "up", "yaw")))
                finally:
                    engine.close()


@unittest.skipUnless(native_binary().is_file(), "Build the native HITL speed test driver first")
class SimulationSpeedIntegrationTests(unittest.TestCase):
    def test_literal_forward_needs_no_object_or_model_then_stops_for_obstacle(self):
        with harness(sim_profile=True, sim_output_speed=2.0) as h:
            h.depth.block.set()
            h.submit("move forward")
            self.assertTrue(h.depth.entered.wait(2))
            self.assertEqual(h.grounder.calls, 0)
            self.assertEqual(h.status()["command"]["forward"], 0)
            h.depth.release.set()
            wait_until(lambda: h.status()["command"]["forward"] == 2.0)
            self.assertEqual(h.grounder.calls, 0)
            wait_until(lambda: h.stats()["axes"]["forward"] == 2.0)
            h.depth.distance = 3
            wait_until(lambda: h.status()["navigation"]["state"] == "blocked")
            self.assertIn("obstacle", h.status()["navigation"]["reason"])
            self.assertTrue(all(h.status()["command"][key] == 0 for key in ("forward", "right", "up", "yaw")))
            self.assertEqual(h.grounder.calls, 0)

    def test_python_only_speed_two_and_obstacle_stop_need_no_native_override(self):
        with harness(sim_profile=True, sim_output_speed=2.0) as h:
            h.submit("move forward")
            wait_until(lambda: h.status()["command"]["forward"] == 2.0)
            status = h.status()
            # The original deployed bridge has neither of these optional
            # status fields, so this also runs against the pre-override binary.
            self.assertEqual(status.get("sim_forward_speed_mps", 0), 0)
            self.assertFalse(status.get("sim_forward_override_active", False))
            self.assertTrue(all(status["command"][key] == 0 for key in ("right", "up", "yaw")))
            wait_until(lambda: h.stats()["axes"]["forward"] == 2.0)
            h.depth.distance = 3
            wait_until(lambda: h.status()["navigation"]["state"] == "blocked")
            self.assertTrue(all(h.status()["command"][key] == 0 for key in ("forward", "right", "up", "yaw")))
            wait_until(lambda: h.stats()["axes"]["forward"] == 0)

    def test_both_opt_ins_scale_only_clear_forward_motion_then_obstacle_stops_it(self):
        with harness(native_sim_speed=True, sim_profile=True) as h:
            h.depth.block.set()
            h.submit("move forward")
            self.assertTrue(h.depth.entered.wait(2))
            self.assertEqual(h.grounder.calls, 0)
            self.assertEqual(h.status()["command"]["forward"], 0)
            self.assertFalse(h.status()["sim_forward_override_active"])
            h.depth.release.set()
            wait_until(lambda: h.status()["command"]["forward"] == 15)
            status = h.status()
            self.assertEqual(status["sim_forward_speed_mps"], 15)
            self.assertTrue(status["sim_forward_override_active"])
            self.assertEqual(status["navigation"]["original_planner_forward_mps"], 0.2)
            self.assertTrue(status["navigation"]["sim_forward_override_applied"])
            self.assertTrue(all(status["command"][key] == 0 for key in ("right", "up", "yaw")))
            wait_until(lambda: h.stats()["axes"]["forward"] == 15)
            h.depth.distance = 3  # A close plane obstructs the existing low-speed test corridor.
            wait_until(lambda: h.status()["navigation"]["state"] == "blocked")
            stopped = h.status()
            self.assertFalse(stopped["sim_forward_override_active"])
            self.assertTrue(all(stopped["command"][key] == 0 for key in ("forward", "right", "up", "yaw")))
            self.assertIn("obstacle", stopped["navigation"]["reason"])
            wait_until(lambda: h.stats()["axes"]["forward"] == 0)

    def test_native_override_without_python_test_marker_preserves_normal_speed(self):
        with harness(native_sim_speed=True) as h:
            h.submit()
            h.grounder.finish()
            wait_until(lambda: h.status()["command"]["forward"] == 0.5)
            self.assertFalse(h.status()["sim_forward_override_active"])
            self.assertNotIn("behavior", h.status()["navigation"])

    def test_python_test_profile_without_native_override_stays_at_point_two(self):
        with harness(sim_profile=True) as h:
            h.submit("move forward")
            wait_until(lambda: h.status()["command"]["forward"] == 0.2)
            self.assertEqual(h.status()["sim_forward_speed_mps"], 0)
            self.assertFalse(h.status()["sim_forward_override_active"])

    def test_frozen_camera_expires_the_high_speed_setpoint(self):
        with harness(native_sim_speed=True, sim_profile=True) as h:
            h.submit("move forward")
            wait_until(lambda: h.status()["command"]["forward"] == 15)
            h.input("freeze")
            time.sleep(0.6)
            status = h.status()
            self.assertEqual(status["command"]["forward"], 0)
            self.assertFalse(status["sim_forward_override_active"])
            self.assertEqual(status["navigation"]["state"], "stale")


if __name__ == "__main__":
    unittest.main()
