"""Goal lifecycle, actual optical flow, obstacle rejection and bridge ownership.

No model downloads, hardware commands, system service changes or test-image
fallback in the runtime path. Perception backends are injected where indicated.
"""
from concurrent.futures import Future
import json
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from navigation import NavigationConfig, NavigationEngine, encode_message, local_command, zero_axes
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

    def test_lost_target_stops_instead_of_reusing_motion(self):
        self.acquired()
        response = self.engine.handle(dict(text=message(2), request_id="r", default_duration=0.2),
                                      Image.new("RGB", (448, 252)))
        self.assertEqual(response["command"], zero_axes(0.2))
        self.assertEqual(response["navigation"]["state"], "lost")
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

    def test_clear_forward_detour_is_not_undone_by_yawing_toward_blocked_target(self):
        depth = np.full((252, 448), 30.0)
        depth[110:142, 319:330] = 2.5/0.65
        command, state, _ = local_command(configuration(), depth, (0.68, 0.25, 0.8, 0.75), 0.1)
        self.assertEqual(state, "approaching")
        self.assertEqual(command["forward"], 0.5)
        self.assertEqual(command["yaw"], 0)

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
