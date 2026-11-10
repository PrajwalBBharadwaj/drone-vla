"""Goal lifecycle, actual optical flow, obstacle rejection and bridge ownership.

No model downloads, hardware commands, system service changes or test-image
fallback in the runtime path. Perception backends are injected where indicated.
"""
from concurrent.futures import Future
import io
import json
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from navigation import NavigationConfig, NavigationEngine, encode_message, is_sim_forward_instruction, local_command, simulation_forward_stop_config, zero_axes
from navigation_control import run_legacy_navigation as run_navigation
from navigation_perception import FeatureTracker, GroundedTarget
from vla_service import plan_request


def configuration(**updates):
    values = dict(image_width=448, image_height=252, fx=300, fy=300, cx=224, cy=126,
                  camera_forward_aligned=True, vehicle_radius_m=0.4,
                  clearance_margin_m=0.3, braking_deceleration_mps2=0.5)
    values.update(updates)
    return NavigationConfig(**values)


BOX = (0.35, 0.25, 0.65, 0.75)


def texture():
    return Image.fromarray(np.random.default_rng(42).integers(0, 256, (252, 448, 3), dtype=np.uint8))


def body(text, frame=None, request_id="request-1", **updates):
    frame = frame or texture()
    metadata = dict(text=text, request_id=request_id, width=frame.width, height=frame.height,
                    format="RGB", default_duration=0.2)
    metadata.update(updates)
    header = json.dumps(metadata).encode()
    return struct.pack("!I", len(header)) + header + frame.tobytes()


def message(sequence=0, **updates):
    values = dict(op="start" if sequence == 0 else "step", goal_id="goal1", owner="manual1", sequence=sequence)
    if sequence == 0:
        values["text"] = "Approach the blue building while avoiding the tree"
    values.update(updates)
    return encode_message(**values)


class ControlledExecutor:
    def __init__(self):
        self.future = Future()
        self.submissions = 0
        self.future.set_running_or_notify_cancel()

    def submit(self, *args):
        self.submissions += 1
        return self.future

    def finish(self, **updates):
        values = dict(found=True, box=BOX, confidence=0.95, reason="One blue building")
        values.update(updates)
        self.future.set_result(GroundedTarget(**values))

    def shutdown(self, **kwargs):
        pass


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.executor = ControlledExecutor()
        self.depth = Mock(return_value=np.full((252, 448), 30, dtype=np.float32))
        self.engine = NavigationEngine(configuration(), grounder=Mock(), depth=self.depth,
                                       clock=self.clock, executor=self.executor)

    def step(self, sequence, **updates):
        self.clock.sleep(0.2)
        return self.engine.handle(dict(text=message(sequence, **updates), request_id="r", default_duration=0.2), texture())

    def acquired(self):
        self.step(0)
        self.executor.finish()
        response = self.step(1)
        self.assertEqual(response["navigation"]["state"], "ready")
        self.assertEqual(response["command"], zero_axes(0.2))

    def test_grounding_async_then_real_tracker_no_repeated_model_calls(self):
        first = self.step(0)
        waiting = self.step(1)
        self.assertEqual(first["navigation"]["state"], "grounding")
        self.assertEqual(waiting["command"], zero_axes(0.2))
        self.depth.assert_not_called()
        self.executor.finish()
        self.assertEqual(self.step(2)["navigation"]["state"], "ready")
        for sequence in range(3, 6):
            response = self.step(sequence)
            self.assertEqual(response["navigation"]["state"], "approaching")
            self.assertEqual(response["command"]["forward"], 0.5)
        self.assertEqual(self.executor.submissions, 1)
        self.assertEqual(self.depth.call_count, 3)

    def test_unconfirmed_target_holds_instead_of_reusing_motion(self):
        self.acquired()
        response = self.engine.handle(dict(text=message(2), request_id="r", default_duration=0.2),
                                      Image.new("RGB", (448, 252)))
        self.assertEqual(response["command"], zero_axes(0.2))
        self.assertEqual(response["navigation"]["state"], "stale")
        self.assertEqual(response["navigation"]["target"]["visibility"], "unconfirmed")
        self.assertIsNone(response["navigation"]["target"]["center_px"])
        self.depth.assert_not_called()

    def test_buffer_replay_is_batched_while_fresh_frames_continue(self):
        clock = self.clock
        class SlowTracker:
            def start(self, frame, box):
                pass

            def update(self, frame):
                clock.sleep(0.1)
                return BOX
        self.engine.tracker_factory = SlowTracker
        self.step(0)
        for sequence in range(1, 12):
            self.step(sequence)
        self.executor.finish()
        response = self.step(12)
        self.assertEqual(response["navigation"]["state"], "acquiring")
        self.assertEqual(response["command"], zero_axes(0.2))
        self.assertLess(response["navigation"]["timings_ms"]["service_ms"], 350)
        for sequence in range(13, 40):
            response = self.step(sequence)
            self.assertEqual(response["command"], zero_axes(0.2))
            self.depth.assert_not_called()
            if response["navigation"]["state"] == "ready":
                break
        self.assertEqual(response["navigation"]["state"], "ready")
        self.assertEqual(self.step(sequence+1)["navigation"]["state"], "approaching")

    def test_slow_perception_holds_and_next_fresh_frame_can_retry(self):
        self.acquired()
        def slow(frame):
            self.clock.sleep(0.4)
            return np.full((252, 448), 30)
        self.depth.side_effect = slow
        response = self.step(2)
        self.assertEqual(response["navigation"]["state"], "stale")
        self.assertEqual(response["command"], zero_axes(0.2))
        self.depth.side_effect = None
        self.assertEqual(self.step(3)["navigation"]["state"], "approaching")

    def test_frame_gap_session_and_replayed_sequence_hold(self):
        for case in ("gap", "owner", "replay"):
            with self.subTest(case=case):
                self.setUp()
                self.acquired()
                if case == "gap":
                    self.clock.sleep(1.1)
                result = self.step(1 if case == "replay" else 2, **({"owner": "manual2"} if case == "owner" else {}))
                self.assertIn(result["navigation"]["state"], {"error", "lost"})
                self.assertEqual(result["command"], zero_axes(0.2))

    def test_grounder_ambiguity_exception_and_deadline_hold(self):
        for case in ("ambiguous", "exception", "timeout"):
            with self.subTest(case=case):
                self.setUp()
                self.step(0)
                if case == "ambiguous":
                    self.executor.finish(found=False, box=None, reason="Two blue buildings")
                elif case == "exception":
                    self.executor.future.set_exception(RuntimeError("Ollama offline"))
                else:
                    self.engine.goal.started = -self.engine.config.grounding_timeout_seconds-1
                response = self.step(1)
                self.assertEqual(response["navigation"]["state"], "error")
                self.assertEqual(response["command"], zero_axes(0.2))

    def test_cancelled_grounding_cannot_be_used_by_next_goal(self):
        self.step(0)
        self.engine.cancel()
        response = self.step(1)
        self.assertEqual(response["navigation"]["state"], "error")
        response = self.step(0, goal_id="new_goal")
        self.assertEqual(response["navigation"]["state"], "error")
        self.assertEqual(self.executor.submissions, 1)

    def test_frame_geometry_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "calibration"):
            self.engine.handle(dict(text=message(), request_id="r", default_duration=0.2), Image.new("RGB", (224, 126)))

    def test_http_contract_uses_live_pixels_and_temporary_bypass_cannot_enable_navigation(self):
        with self.assertRaisesRegex(ValueError, "not configured"):
            plan_request(body(message()))
        with self.assertRaisesRegex(ValueError, "visual checks"):
            plan_request(body(message()), commands_only=True, navigation=self.engine)
        with patch("vla_service.tempfile.TemporaryDirectory", side_effect=AssertionError("No PNG round trip")), \
             patch("vla_service.time.monotonic", self.clock):
            response = plan_request(body(message()), navigation=self.engine)
        self.assertEqual(response["request_id"], "request-1")
        self.assertTrue(response["override_reason"].startswith("navigation:"))
        self.assertEqual(response["command"], zero_axes(0.2))
        fake = Mock()
        from prototype import interpret_command
        fake_plan = lambda text, image: SimpleNamespace(command=interpret_command(text), override_reason=None)
        plan_request(body("Hover"), planner=fake_plan, navigation=fake)
        fake.cancel.assert_called_once()


class GeometryTests(unittest.TestCase):
    def test_metric_backend_runs_actual_transformers_architecture_without_downloads(self):
        from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation, Dinov2Config, DPTImageProcessor
        from navigation_perception import MetricDepth
        # Tiny random weights exercise the real preprocessing/model/upsampling
        # contract, not accuracy or timing of the pretrained outdoor model.
        backbone = Dinov2Config(hidden_size=24, num_hidden_layers=4, num_attention_heads=3,
                                image_size=28, patch_size=14, out_indices=[1, 2, 3, 4],
                                reshape_hidden_states=False)
        config = DepthAnythingConfig(backbone_config=backbone, reassemble_hidden_size=24,
                                     neck_hidden_sizes=[8, 8, 8, 8], fusion_hidden_size=8,
                                     head_hidden_size=8, depth_estimation_type="metric", max_depth=80)
        model = DepthAnythingForDepthEstimation(config)
        with patch("transformers.AutoModelForDepthEstimation.from_pretrained", return_value=model), \
             patch("transformers.AutoImageProcessor.from_pretrained", return_value=DPTImageProcessor()):
            depth = MetricDepth(device="cpu")(texture())
        self.assertEqual(depth.shape, (252, 448))
        self.assertTrue(np.all(np.isfinite(depth)))
        self.assertGreater(float(depth.min()), 0)
        self.assertLessEqual(float(depth.max()), 80)

    def test_clear_corridor_approach_and_standoff_hold(self):
        config = configuration()
        command, state, _ = local_command(config, np.full((252, 448), 30), BOX, 0.1)
        self.assertEqual(state, "approaching")
        self.assertEqual(command["forward"], 0.5)
        command, state, _ = local_command(config, np.full((252, 448), 6), BOX, 0.1)
        self.assertEqual(state, "arrived")
        self.assertEqual(command, zero_axes(0.2))

    def test_obstacle_on_route_never_returns_forward_motion(self):
        depth = np.full((252, 448), 30.0)
        depth[96:156, 212:236] = 2.5/0.65
        command, state, _ = local_command(configuration(), depth, BOX, 0.1)
        self.assertIn(state, {"aligning", "blocked"})
        self.assertEqual(command["forward"], 0)
        self.assertEqual(command["right"], 0)
        self.assertEqual(command["up"], 0)

    def test_more_stopping_distance_rejects_obstacle_previously_outside_horizon(self):
        depth = np.full((252, 448), 30.0)
        depth[120:132, :] = 5/0.65
        command, state, _ = local_command(configuration(), depth, BOX, 0.1)
        self.assertEqual(state, "approaching")
        command, state, _ = local_command(configuration(max_speed_mps=1, braking_deceleration_mps2=0.1), depth, BOX, 0.1)
        self.assertEqual(command["forward"], 0)
        self.assertEqual(state, "blocked")

    def test_off_axis_destination_keeps_steering_instead_of_driving_a_clear_unrelated_heading(self):
        depth = np.full((252, 448), 30.0)
        depth[110:142, 319:330] = 2.5/0.65
        command, state, _ = local_command(configuration(), depth, (0.68, 0.25, 0.8, 0.75), 0.1)
        self.assertEqual(state, "aligning")
        self.assertEqual(command["forward"], 0)
        self.assertGreater(command["yaw"], 0)

    def test_all_headings_blocked_unknown_depth_and_wrong_size_hold(self):
        depth = np.full((252, 448), 30.0)
        depth[120:132, :] = 2.5/0.65
        command, state, _ = local_command(configuration(), depth, BOX, 0.1)
        self.assertEqual(state, "blocked")
        self.assertEqual(command, zero_axes(0.2))
        for value in (float("nan"), float("inf"), 0, -1):
            with self.subTest(value=value):
                depth = np.full((252, 448), 30.0)
                depth[0, 0] = value
                self.assertEqual(local_command(configuration(), depth, BOX, 0.1)[0], zero_axes(0.2))
        with self.assertRaisesRegex(ValueError, "aligned"):
            local_command(configuration(), np.ones((100, 100)), BOX, 0.1)

    def test_calibration_and_envelope_limits(self):
        for updates in ({"fx": float("nan")}, {"cx": 900}, {"camera_forward_aligned": False},
                        {"max_speed_mps": 2}, {"step_seconds": 2}, {"braking_deceleration_mps2": 0}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                configuration(**updates)
        for updates in ({"sequence": -1}, {"goal_id": "../bad"}, {"text": " "}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                message(**updates)
        with self.assertRaises(ValueError):
            GroundedTarget(found=True, box=(0.8, 0.1, 0.3, 0.2), confidence=0.9, reason="bad box")
        with self.assertRaises(ValueError):
            message(text="中"*500)

    def test_real_optical_flow_tracks_translation_and_rejects_occlusion(self):
        import cv2
        frame = texture()
        tracker = FeatureTracker()
        tracker.start(frame, BOX)
        moved = Image.fromarray(cv2.warpAffine(np.asarray(frame), np.float32([[1, 0, 4], [0, 1, 2]]), frame.size))
        box = tracker.update(moved)
        self.assertAlmostEqual(box[0], BOX[0]+4/448, delta=0.003)
        self.assertAlmostEqual(box[1], BOX[1]+2/252, delta=0.003)
        with self.assertRaises(ValueError):
            tracker.update(Image.new("RGB", frame.size))


class SimForwardStopTests(unittest.TestCase):
    def test_simulation_policy_is_disabled_by_default(self):
        self.assertFalse(configuration().sim_forward_stop_test)
        self.assertIsNone(configuration().sim_forward_speed_mps)

    def test_python_only_test_speed_changes_output_without_saved_or_planning_speed(self):
        original = configuration(max_speed_mps=0.5)
        before = original.model_dump()
        temporary = simulation_forward_stop_config(original, forward_speed_mps=2)
        self.assertEqual(original.model_dump(), before)
        self.assertEqual(temporary.max_speed_mps, 0.2)
        self.assertEqual(temporary.sim_forward_speed_mps, 2)
        self.assertEqual(temporary.clearance_margin_m, 0.2)
        self.assertEqual(temporary.minimum_clearance_m, 2)
        self.assertEqual(NavigationConfig.model_validate(temporary.model_dump()), temporary)
        command, state, reason = local_command(temporary, np.full((252, 448), 30.0), BOX, 0.1)
        self.assertEqual(state, "approaching")
        self.assertEqual(command, dict(forward=2.0, right=0.0, up=0.0, yaw=0.0,
                                       duration_seconds=temporary.step_seconds))
        self.assertIn("SIM forward-stop", reason)

    def test_python_only_test_speed_still_returns_zero_for_obstacle(self):
        temporary = simulation_forward_stop_config(configuration(), forward_speed_mps=2)
        depth = np.full((252, 448), 30.0)
        depth[96:156, 212:236] = 2.5/0.65
        command, state, reason = local_command(temporary, depth, BOX, 0.1)
        self.assertEqual(state, "blocked")
        self.assertEqual(command, zero_axes(temporary.step_seconds))
        self.assertIn("obstacle", reason)

    def test_normal_approach_does_not_use_the_optional_simulator_output_speed(self):
        config = configuration(sim_forward_speed_mps=2)
        self.assertFalse(config.sim_forward_stop_test)
        command, state, _ = local_command(config, np.full((252, 448), 30.0), BOX, 0.1)
        self.assertEqual(state, "approaching")
        self.assertEqual(command["forward"], 0.5)
        with self.assertRaises(ValueError):
            configuration(max_speed_mps=2)

    def test_python_only_simulator_speed_never_exceeds_native_wire_limit(self):
        for speed in (0, -1, 2.0001, 15, float("nan"), float("inf")):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                configuration(sim_forward_stop_test=True, sim_forward_speed_mps=speed)
            with self.subTest(helper_speed=speed), self.assertRaises(ValueError):
                simulation_forward_stop_config(configuration(), forward_speed_mps=speed)
        for speed in (True, False, "2"):
            with self.subTest(helper_coercion=speed), self.assertRaises(ValueError):
                simulation_forward_stop_config(configuration(), forward_speed_mps=speed)
        for speed in (0.05, 0.2, 1, 2):
            with self.subTest(valid_speed=speed):
                profile = simulation_forward_stop_config(configuration(), forward_speed_mps=speed)
                self.assertEqual(profile.sim_forward_speed_mps, speed)

    def test_temporary_profile_preserves_original_calibration_and_motion_timing(self):
        original = configuration(max_speed_mps=0.8, clearance_margin_m=1.0,
                                 minimum_clearance_m=5, standoff_m=7,
                                 vehicle_radius_m=0.8, braking_deceleration_mps2=0.4,
                                 depth_scale=0.7, max_processing_seconds=0.3,
                                 max_frame_gap_seconds=0.8, step_seconds=0.15)
        before = original.model_dump()
        temporary = simulation_forward_stop_config(original)
        self.assertIsNot(temporary, original)
        self.assertEqual(original.model_dump(), before)
        self.assertTrue(temporary.sim_forward_stop_test)
        self.assertEqual(temporary.max_speed_mps, 0.2)
        self.assertEqual(temporary.clearance_margin_m, 0.2)
        self.assertEqual(temporary.minimum_clearance_m, 2.0)
        changed = {"sim_forward_stop_test", "max_speed_mps", "clearance_margin_m", "minimum_clearance_m"}
        self.assertEqual(temporary.model_dump(exclude=changed), original.model_dump(exclude=changed))
        self.assertEqual(NavigationConfig.model_validate(temporary.model_dump()), temporary)

    def test_temporary_profile_never_increases_an_already_slower_speed(self):
        for speed in (0.05, 0.1, 0.2):
            with self.subTest(speed=speed):
                original = configuration(max_speed_mps=speed)
                self.assertEqual(simulation_forward_stop_config(original).max_speed_mps, speed)
                self.assertEqual(original.max_speed_mps, speed)

    def test_normal_approach_still_turns_toward_an_off_centre_target(self):
        command, state, _ = local_command(configuration(), np.full((252, 448), 30.0),
                                          (0.7, 0.25, 0.85, 0.75), 0.1)
        self.assertEqual(state, "aligning")
        self.assertEqual(command["forward"], 0)
        self.assertNotEqual(command["yaw"], 0)

    def test_simulation_moves_straight_when_corridor_is_clear_despite_target_bearing(self):
        config = configuration(sim_forward_stop_test=True)
        command, state, _ = local_command(config, np.full((252, 448), 30.0),
                                          (0.7, 0.05, 0.85, 0.2), 0.1)
        self.assertEqual(state, "approaching")
        self.assertEqual(command, dict(forward=config.max_speed_mps, right=0.0, up=0.0,
                                       yaw=0.0, duration_seconds=config.step_seconds))

    def test_simulation_stops_for_centre_obstacle_even_when_side_heading_is_free(self):
        depth = np.full((252, 448), 30.0)
        depth[96:156, 212:236] = 2.5/0.65
        normal_command, normal_state, _ = local_command(configuration(), depth, BOX, 0.1)
        self.assertEqual(normal_state, "blocked")
        self.assertEqual(normal_command, zero_axes(0.2))
        command, state, _ = local_command(configuration(sim_forward_stop_test=True), depth, BOX, 0.1)
        self.assertEqual(state, "blocked")
        self.assertEqual(command, zero_axes(0.2))

    def test_simulation_uses_corridor_depth_instead_of_target_arrival(self):
        # Scaled depth is 3.9 m: inside normal 5 m target stand-off, beyond
        # this fixture's 3.7 m stopping-check horizon including its footprint.
        depth = np.full((252, 448), 6.0)
        self.assertEqual(local_command(configuration(), depth, BOX, 0.1)[1], "arrived")
        command, state, _ = local_command(configuration(sim_forward_stop_test=True), depth, BOX, 0.1)
        self.assertEqual(state, "approaching")
        self.assertEqual(command["forward"], 0.5)
        self.assertEqual(command["yaw"], 0)

    def test_simulation_continues_to_hold_for_unknown_or_invalid_depth(self):
        for value in (float("nan"), float("inf"), 0, -1):
            with self.subTest(value=value):
                depth = np.full((252, 448), 30.0)
                depth[0, 0] = value
                command, state, _ = local_command(configuration(sim_forward_stop_test=True), depth, BOX, 0.1)
                self.assertEqual(state, "blocked")
                self.assertEqual(command, zero_axes(0.2))


class SimForwardStopLifecycleTests(unittest.TestCase):
    """Only literal-forward diagnostics skip object grounding and tracking."""
    def setUp(self):
        self.reset_engine()

    def reset_engine(self):
        self.clock = Clock()
        self.executor = ControlledExecutor()
        self.depth = Mock(return_value=np.full((252, 448), 30.0, dtype=np.float32))
        self.tracker_factory = Mock(side_effect=AssertionError("Simulator corridor test must not create a tracker"))
        self.engine = NavigationEngine(
            simulation_forward_stop_config(configuration(), forward_speed_mps=2),
            grounder=Mock(), depth=self.depth, tracker_factory=self.tracker_factory,
            clock=self.clock, executor=self.executor)
        self.addCleanup(self.engine.close)

    def step(self, sequence, image=None, **updates):
        self.clock.sleep(0.2)
        metadata = dict(text=message(sequence), request_id="sim-request", default_duration=0.35,
                        bridge_protocol=2, session="manual1", goal_id="goal1", frame_id=sequence+1,
                        frame_monotonic_ns=50_000_000_000+round(self.clock()*1e9))
        metadata.update(updates)
        return self.engine.handle(metadata, texture() if image is None else image)

    def ready(self):
        result = self.step(0, text=message(0, text="move forward"))
        self.assertEqual(result["navigation"]["state"], "ready")
        self.assertEqual(result["command"], zero_axes(0.35))
        self.depth.assert_not_called()
        self.tracker_factory.assert_not_called()
        return result

    def test_literal_forward_recognition_is_exact_and_does_not_accept_negation_or_sequences(self):
        for text in ("forward", "move forward", "go forward", "fly forward", "move straight",
                     "go straight", "fly straight", "move straight ahead", "go straight ahead",
                     "fly straight ahead", "  MOVE   FORWARD!!!  ", "Fly Straight Ahead?"):
            with self.subTest(accepted=text):
                self.assertTrue(is_sim_forward_instruction(text))
        for text in (None, 2, "", " ", "don't move forward", "do not move forward", "stop then move forward",
                     "move forward then turn left", "move forward and stop", "move forward at 2 m/s",
                     "move forward for 3 seconds", "move toward that tree", "approach forward building"):
            with self.subTest(rejected=text):
                self.assertFalse(is_sim_forward_instruction(text))

    def test_literal_forward_holds_without_grounder_then_uses_fresh_depth_for_motion_and_stop(self):
        result = self.step(0, text=message(0, text="move forward"))
        self.assertEqual(result["navigation"]["state"], "ready")
        self.assertEqual(result["command"], zero_axes(0.35))
        self.assertIsNone(self.engine.goal.future)
        self.assertEqual(self.executor.submissions, 0)
        self.engine.grounder.assert_not_called()
        self.tracker_factory.assert_not_called()
        self.depth.assert_not_called()
        result = self.step(1, Image.new("RGB", (448, 252), "black"))
        self.assertEqual(result["navigation"]["state"], "approaching")
        self.assertEqual(result["command"], dict(forward=2.0, right=0.0, up=0.0, yaw=0.0,
                                                duration_seconds=0.35))
        self.depth.return_value = np.full((252, 448), 3.0, dtype=np.float32)
        result = self.step(2, Image.new("RGB", (448, 252), "red"))
        self.assertEqual(result["navigation"]["state"], "blocked")
        self.assertIn("obstacle in the current forward corridor", result["navigation"]["reason"])
        self.assertEqual(result["command"], zero_axes(0.35))
        self.assertEqual(self.executor.submissions, 0)
        self.engine.grounder.assert_not_called()
        self.tracker_factory.assert_not_called()

    def test_normal_navigation_does_not_shortcut_literal_forward_grounding(self):
        self.engine.config = configuration()
        result = self.step(0, text=message(0, text="move forward"))
        self.assertEqual(result["navigation"]["state"], "grounding")
        self.assertEqual(result["command"], zero_axes(0.35))
        self.assertIs(self.engine.goal.future, self.executor.future)
        self.assertEqual(self.executor.submissions, 1)
        self.depth.assert_not_called()
        self.tracker_factory.assert_not_called()

    def test_negated_or_compound_forward_is_not_direct_motion(self):
        for text in ("don't move forward", "move forward then turn left", "move forward at 2 m/s"):
            with self.subTest(text=text):
                self.reset_engine()
                result = self.step(0, text=message(0, text=text))
                self.assertEqual(result["navigation"]["state"], "grounding")
                self.assertEqual(result["command"], zero_axes(0.35))
                self.assertEqual(self.executor.submissions, 1)
                self.depth.assert_not_called()
                self.tracker_factory.assert_not_called()

    def test_literal_forward_retains_source_sequence_owner_and_gap_guards(self):
        for case in ("sequence", "frame_id", "source_time", "owner", "gap"):
            with self.subTest(case=case):
                self.reset_engine()
                self.assertEqual(self.step(0, text=message(0, text="move forward"))["navigation"]["state"], "ready")
                updates = {}
                if case == "sequence":
                    updates["text"] = encode_message(op="step", goal_id="goal1", owner="manual1", sequence=0)
                elif case == "frame_id":
                    updates["frame_id"] = self.engine.goal.last_frame_id
                elif case == "source_time":
                    updates["frame_monotonic_ns"] = self.engine.goal.last_source_ns
                elif case == "owner":
                    updates.update(session="manual2", text=message(1, owner="manual2"))
                else:
                    self.clock.sleep(1.1)
                result = self.step(1, **updates)
                self.assertIn(result["navigation"]["state"], {"error", "lost"})
                self.assertEqual(result["command"], zero_axes(0.35))
                self.assertEqual(self.executor.submissions, 0)
                self.depth.assert_not_called()
                self.tracker_factory.assert_not_called()

    def test_literal_forward_can_replace_pending_grounding_without_reading_its_late_result(self):
        self.step(0)
        old_future = self.executor.future
        result = self.step(0, text=message(0, goal_id="goal2", text="move forward"), goal_id="goal2", frame_id=2)
        self.assertEqual(result["navigation"]["state"], "ready")
        self.assertEqual(result["command"], zero_axes(0.35))
        self.assertIsNone(self.engine.goal.future)
        self.assertFalse(old_future.cancelled())
        self.executor.finish(found=False, box=None, reason="Old target was absent")
        result = self.step(1, text=message(1, goal_id="goal2"), goal_id="goal2", frame_id=3)
        self.assertEqual(result["navigation"]["goal_id"], "goal2")
        self.assertEqual(result["navigation"]["state"], "approaching")
        self.assertEqual(result["command"]["forward"], 2.0)
        self.assertEqual(self.executor.submissions, 1)
        self.tracker_factory.assert_not_called()

    def test_cancelling_literal_forward_before_its_fresh_frame_keeps_zero(self):
        self.step(0, text=message(0, text="move forward"))
        self.engine.cancel()
        result = self.step(1)
        self.assertEqual(result["navigation"]["state"], "error")
        self.assertEqual(result["command"], zero_axes(0.35))
        self.assertEqual(self.executor.submissions, 0)
        self.depth.assert_not_called()
        self.tracker_factory.assert_not_called()

    def test_object_goal_keeps_bounded_grounding_history_and_replays_even_with_sim_flag(self):
        tracker = Mock()
        tracker.update.return_value = BOX
        self.engine.tracker_factory = Mock(return_value=tracker)
        self.assertEqual(self.step(0)["navigation"]["state"], "grounding")
        for sequence in range(1, 41):
            image = Image.new("RGB", (448, 252), "white" if sequence % 2 else "black")
            result = self.step(sequence, image)
            self.assertEqual(result["navigation"]["state"], "grounding")
            self.assertEqual(result["command"], zero_axes(0.35))
            self.assertLessEqual(len(self.engine.goal.frames), self.engine.goal.frames.maxlen)
        self.assertGreater(len(self.engine.goal.frames), 1)
        self.executor.finish()
        result = self.step(41, Image.new("RGB", (448, 252), "blue"))
        self.assertEqual(result["navigation"]["state"], "ready")
        self.assertEqual(result["command"], zero_axes(0.35))
        self.assertNotIn("behavior", result["navigation"])
        self.depth.assert_not_called()
        tracker.start.assert_called_once()
        self.assertGreaterEqual(tracker.update.call_count, 40)
        self.assertEqual(self.executor.submissions, 1)

    def test_ready_holds_then_next_fresh_frame_moves_without_target_tracking(self):
        self.ready()
        result = self.step(2, Image.new("RGB", (448, 252), "black"))
        self.assertEqual(result["navigation"]["state"], "approaching")
        self.assertEqual(result["navigation"]["behavior"], "sim-forward-stop-test")
        self.assertEqual(result["command"], dict(forward=2.0, right=0.0, up=0.0, yaw=0.0,
                                                duration_seconds=0.35))
        self.assertEqual(result["navigation"]["timings_ms"]["tracking_ms"], 0)
        self.depth.assert_called_once()
        self.tracker_factory.assert_not_called()
        self.assertEqual(self.executor.submissions, 0)

    def test_new_camera_view_uses_current_depth_to_stop_without_target_loss(self):
        self.ready()
        self.assertEqual(self.step(2)["command"]["forward"], 2.0)
        near = np.full((252, 448), 30.0, dtype=np.float32)
        near[96:156, 212:236] = 2.5/0.65
        self.depth.return_value = near
        result = self.step(3, Image.new("RGB", (448, 252), "red"))
        self.assertEqual(result["navigation"]["state"], "blocked")
        self.assertEqual(result["command"], zero_axes(0.35))
        self.assertIn("obstacle in the current forward corridor", result["navigation"]["reason"])
        self.tracker_factory.assert_not_called()
        self.depth.return_value = np.full((252, 448), 30.0, dtype=np.float32)
        self.assertEqual(self.step(4)["navigation"]["state"], "blocked")
        self.assertEqual(self.executor.submissions, 0)

    def test_initial_grounding_rejection_or_failure_never_starts_simulator_motion(self):
        for case in ("ambiguous", "low_confidence", "exception", "deadline"):
            with self.subTest(case=case):
                self.reset_engine()
                self.step(0)
                if case == "ambiguous":
                    self.executor.finish(found=False, box=None, reason="Multiple trees")
                elif case == "low_confidence":
                    self.executor.finish(confidence=0.1)
                elif case == "exception":
                    self.executor.future.set_exception(RuntimeError("Grounding unavailable"))
                else:
                    self.engine.goal.started = -self.engine.config.grounding_timeout_seconds-1
                result = self.step(1)
                self.assertEqual(result["navigation"]["state"], "error")
                self.assertEqual(result["command"], zero_axes(0.35))
                self.depth.assert_not_called()
                self.tracker_factory.assert_not_called()

    def test_simulator_mode_retains_sequence_source_owner_and_gap_guards(self):
        for case in ("sequence", "frame_id", "source_time", "owner", "gap"):
            with self.subTest(case=case):
                self.reset_engine()
                self.ready()
                updates = {}
                if case == "sequence":
                    updates["text"] = encode_message(op="step", goal_id="goal1", owner="manual1", sequence=0)
                elif case == "frame_id":
                    updates["frame_id"] = self.engine.goal.last_frame_id
                elif case == "source_time":
                    updates["frame_monotonic_ns"] = self.engine.goal.last_source_ns
                elif case == "owner":
                    updates.update(session="manual2", text=message(2, owner="manual2"))
                else:
                    self.clock.sleep(1.1)
                result = self.step(2, **updates)
                self.assertIn(result["navigation"]["state"], {"error", "lost"})
                self.assertEqual(result["command"], zero_axes(0.35))
                self.depth.assert_not_called()
                self.tracker_factory.assert_not_called()

    def test_slow_depth_still_holds_and_retries_on_next_fresh_frame(self):
        self.ready()
        def slow(frame):
            self.clock.sleep(0.4)
            return np.full((252, 448), 30.0, dtype=np.float32)
        self.depth.side_effect = slow
        result = self.step(2)
        self.assertEqual(result["navigation"]["state"], "stale")
        self.assertEqual(result["command"], zero_axes(0.35))
        self.depth.side_effect = None
        self.assertEqual(self.step(3)["command"]["forward"], 2.0)
        self.tracker_factory.assert_not_called()


class SimForwardStopStartupTests(unittest.TestCase):
    def test_explicit_output_speed_requires_the_simulator_test_mode(self):
        from vla_service import main
        environment = dict(VLA_COMMANDS_ONLY="0", VLA_NAVIGATION_CONFIG="", VLA_SIM_FORWARD_STOP_TEST="0")
        cases = (
            ["--sim-forward-speed-mps", "2"],
            ["--sim-forward-speed-mps", "2", "--navigation-config", "/unused/navigation.json"],
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), patch.dict("os.environ", environment, clear=True), \
                 patch("sys.argv", ["vla_service.py"]+arguments), patch("sys.stderr", new_callable=io.StringIO) as error:
                with self.assertRaises(SystemExit) as exit_status:
                    main()
                self.assertEqual(exit_status.exception.code, 2)
                self.assertIn("--sim-forward-stop-test", error.getvalue())

    def test_runtime_output_speed_also_requires_the_simulator_test_mode(self):
        from vla_service import main
        environment = dict(VLA_COMMANDS_ONLY="0", VLA_NAVIGATION_CONFIG="/unused/navigation.json",
                           VLA_SIM_FORWARD_STOP_TEST="0", VLA_SIM_FORWARD_SPEED_MPS="2")
        with patch.dict("os.environ", environment, clear=True), patch("sys.argv", ["vla_service.py"]), \
             patch("sys.stderr", new_callable=io.StringIO) as error:
            with self.assertRaises(SystemExit) as exit_status:
                main()
            self.assertEqual(exit_status.exception.code, 2)
            self.assertIn("--sim-forward-stop-test", error.getvalue())

    def test_invalid_output_speed_is_rejected_before_reading_calibration(self):
        from vla_service import main
        environment = dict(VLA_COMMANDS_ONLY="0", VLA_SIM_FORWARD_STOP_TEST="0")
        for speed in ("0", "-1", "2.01", "15", "nan", "inf", "bad"):
            with self.subTest(speed=speed), patch.dict("os.environ", environment, clear=True), \
                 patch("sys.argv", ["vla_service.py", "--sim-forward-stop-test", "--navigation-config", "/unused/navigation.json",
                                    "--sim-forward-speed-mps", speed]), \
                 patch("sys.stderr", new_callable=io.StringIO) as error:
                with self.assertRaises(SystemExit) as exit_status:
                    main()
                self.assertEqual(exit_status.exception.code, 2)
                self.assertIn("speed", error.getvalue().lower())

    def test_simulation_flag_requires_calibrated_navigation_and_visual_checks(self):
        from vla_service import main
        cases = (
            ["--sim-forward-stop-test"],
            ["--sim-forward-stop-test", "--navigation-config", "/unused/navigation.json", "--commands-only"],
        )
        environment = dict(VLA_COMMANDS_ONLY="0", VLA_NAVIGATION_CONFIG="", VLA_SIM_FORWARD_STOP_TEST="0")
        for arguments in cases:
            with self.subTest(arguments=arguments), patch.dict("os.environ", environment, clear=True), \
                 patch("sys.argv", ["vla_service.py"]+arguments), patch("sys.stderr", new_callable=io.StringIO) as error:
                with self.assertRaises(SystemExit) as exit_status:
                    main()
                self.assertEqual(exit_status.exception.code, 2)
                self.assertIn("requires --navigation-config and visual checks", error.getvalue())

    def test_runtime_environment_activation_uses_the_same_navigation_guard(self):
        from vla_service import main
        environment = dict(VLA_COMMANDS_ONLY="0", VLA_NAVIGATION_CONFIG="", VLA_SIM_FORWARD_STOP_TEST="1")
        with patch.dict("os.environ", environment, clear=True), patch("sys.argv", ["vla_service.py"]), \
             patch("sys.stderr", new_callable=io.StringIO) as error:
            with self.assertRaises(SystemExit) as exit_status:
                main()
            self.assertEqual(exit_status.exception.code, 2)
            self.assertIn("requires --navigation-config and visual checks", error.getvalue())


class ClientTests(unittest.TestCase):
    def bridge(self, takeover=False):
        executor = ControlledExecutor()
        clock = Clock()
        engine = NavigationEngine(configuration(), grounder=Mock(),
                                  depth=lambda frame: np.full((252, 448), 30.0), clock=clock, executor=executor)
        class Bridge:
            def __init__(self):
                self.sent = []
                self.current = dict(session="s", ticket="t", request_id="initial", eligible=True, state="completed")

            def status(self):
                return dict(self.current)

            def request(self, payload):
                self.sent.append(payload)
                request_id = str(len(self.sent))
                if payload["text"] == "Hover":
                    self.current.update(request_id=request_id, command=zero_axes(0.1))
                else:
                    text_message = payload["text"]
                    result = engine.handle(dict(text=text_message, request_id=request_id, default_duration=0.2), texture())
                    if len(self.sent) == 1:
                        executor.finish()
                    if len(self.sent) == 3:
                        engine.goal.state, engine.goal.reason = "blocked", "Test obstacle"
                    self.current.update(result, session="s", ticket="t", eligible=True, state="executing")
                return {"ok": True, "request_id": request_id}
        bridge = Bridge()
        def sleep(seconds):
            clock.sleep(seconds)
            if takeover and len(bridge.sent) >= 1:
                bridge.current["request_id"] = "operator"
        return bridge, clock, sleep

    def test_navigation_round_trip_stops_on_block_and_sends_owned_hover(self):
        bridge, clock, sleep = self.bridge()
        output = []
        records = run_navigation(bridge, "Approach that blue building", emit=output.append, clock=clock, sleep=sleep)
        self.assertEqual([record["state"] for record in records], ["grounding", "ready", "approaching", "blocked"])
        self.assertEqual(bridge.sent[-1]["text"], "Hover")
        self.assertTrue(all(request["session"] == "s" for request in bridge.sent))
        self.assertTrue(all(request["duration_seconds"] <= 0.2 for request in bridge.sent))

    def test_operator_takeover_stops_without_sending_hover_or_new_instruction(self):
        bridge, clock, sleep = self.bridge(takeover=True)
        with self.assertRaisesRegex(RuntimeError, "superseded"):
            run_navigation(bridge, "Approach the blue building", emit=Mock(), clock=clock, sleep=sleep)
        self.assertEqual(len(bridge.sent), 1)


if __name__ == "__main__":
    unittest.main()
