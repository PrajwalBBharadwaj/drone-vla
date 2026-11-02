import importlib.util
import contextlib
import io
import os
import struct
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dji_control import execute, joystick_command
from vla_service import VlaHandler, decode_request, plan_request


def motion(**updates):
    values = dict(valid=True, reason=None, longitudinal_direction="none", longitudinal_speed_mps=0.0,
                  lateral_direction="none", lateral_speed_mps=0.0, vertical_direction="none", vertical_speed_mps=0.0,
                  yaw_direction="none", yaw_speed_deg_s=0.0, duration_seconds=None)
    values.update(updates)
    return SimpleNamespace(**values)


def camera_request(**updates):
    metadata = dict(request_id="request-1", text="forward", width=2, height=1, format="RGB", default_duration=1)
    metadata.update(updates)
    header = json.dumps(metadata).encode()
    return struct.pack("!I", len(header)) + header + bytes([255, 0, 0, 0, 255, 0])


class FakeClient:
    def __init__(self, sessions=("a", "a"), state="executing"):
        self.sessions = iter(sessions)
        self.sent = []
        self.state = state

    def status(self):
        return dict(session=next(self.sessions), ticket="ticket", eligible=True, request_id="request-1",
                    state=self.state, inference_error="Model unavailable")

    def request(self, payload):
        self.sent.append(payload)
        return dict(ok=True, request_id="request-1")


class ControlTests(unittest.TestCase):
    def test_http_handler_forwards_commands_only_mode(self):
        body = camera_request(text="Move forward at 0.5 m/s")
        def planner(text, image):
            self.assertEqual(text, "Move forward at 0.5 m/s")
            self.assertIsNone(image)
            return SimpleNamespace(command=motion(longitudinal_direction="forward", longitudinal_speed_mps=0.5), override_reason=None)
        handler = SimpleNamespace(path="/plan", headers={"Content-Length": str(len(body))}, rfile=io.BytesIO(body),
                                  server=SimpleNamespace(inference_lock=threading.Lock(), planner=planner, commands_only=True), reply=Mock())
        VlaHandler.do_POST(handler)
        status, result = handler.reply.call_args.args
        self.assertEqual(status, 200)
        self.assertEqual(result["command"]["forward"], 0.5)
        self.assertIn("Visual checks bypassed", result["override_reason"])
        self.assertFalse(handler.server.inference_lock.locked())

    @unittest.skipUnless(importlib.util.find_spec("pydantic") and importlib.util.find_spec("ollama")
                         and importlib.util.find_spec("PIL"), "Planner dependencies unavailable")
    def test_commands_only_startup_skips_depth_preload(self):
        import vla_service
        for arguments, environment in ((["--preload", "--commands-only"], "0"), (["--preload"], "1")):
            with patch.dict(os.environ, {"VLA_COMMANDS_ONLY": environment}), patch("sys.argv", ["vla_service.py", *arguments]), \
                 patch("vla_service.VlaServer") as factory, patch("vision_test._get_depth_pipeline") as depth, \
                 patch("vla_service.logging.basicConfig"), contextlib.redirect_stdout(io.StringIO()):
                factory.return_value.__enter__.return_value.serve_forever.side_effect = KeyboardInterrupt()
                vla_service.main()
            factory.assert_called_once_with(("127.0.0.1", 8766), commands_only=True)
            depth.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("pydantic") and importlib.util.find_spec("ollama")
                         and importlib.util.find_spec("PIL"), "Planner dependencies unavailable")
    def test_commands_only_skips_vision_and_keeps_motion_limits(self):
        from prototype import DroneMotionCommand
        requested = DroneMotionCommand(**vars(motion(longitudinal_direction="forward", longitudinal_speed_mps=0.5)))
        with patch("integrated_control.interpret_command", return_value=requested) as interpret, \
             patch("integrated_control.assess_image") as vision, patch("integrated_control.assess_monocular_depth") as depth, \
             patch("vla_service.tempfile.TemporaryDirectory", side_effect=AssertionError("No image preparation needed")):
            result = plan_request(camera_request(text="Move forward at 0.5 m/s"), commands_only=True)
            self.assertEqual(result["command"]["forward"], 0.5)
            self.assertEqual(result["command"]["duration_seconds"], 1)
            self.assertIn("Visual checks bypassed", result["override_reason"])
            interpret.assert_called_once_with("Move forward at 0.5 m/s")
            vision.assert_not_called()
            depth.assert_not_called()
            requested.longitudinal_speed_mps = 5
            with self.assertRaisesRegex(ValueError, "VLA limits"):
                plan_request(camera_request(), commands_only=True)
        with self.assertRaises(ValueError):
            plan_request(camera_request()[:-1], commands_only=True)

    def test_commands_only_health_reports_mode(self):
        handler = SimpleNamespace(path="/health", reply=Mock(), server=SimpleNamespace(commands_only=True))
        VlaHandler.do_GET(handler)
        handler.reply.assert_called_once_with(200, {"ok": True, "mode": "commands-only"})

    @unittest.skipUnless(importlib.util.find_spec("pydantic") and importlib.util.find_spec("ollama"), "Planner dependencies unavailable")
    def test_explicit_hover_stop_hold_are_zero_without_model(self):
        from prototype import interpret_command
        with patch("prototype.chat") as model:
            for text in ("Hover", " STOP! ", "hold", "Hold   position."):
                command = interpret_command(text)
                axes = joystick_command(command)
                self.assertEqual([axes[key] for key in ("forward", "right", "up", "yaw")], [0, 0, 0, 0])
                self.assertEqual(axes["duration_seconds"], 1)
            model.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("pydantic") and importlib.util.find_spec("ollama"), "Planner dependencies unavailable")
    def test_hover_with_additional_motion_still_uses_model(self):
        from prototype import DroneMotionCommand, interpret_command
        requested = DroneMotionCommand(**vars(motion(longitudinal_direction="forward", longitudinal_speed_mps=1)))
        response = SimpleNamespace(done_reason="stop", message=SimpleNamespace(content=requested.model_dump_json()))
        with patch("prototype.chat", return_value=response) as model:
            self.assertEqual(interpret_command("Hover then move forward at 1 m/s").longitudinal_speed_mps, 1)
            model.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec("pydantic") and importlib.util.find_spec("ollama")
                         and importlib.util.find_spec("PIL"), "Planner dependencies unavailable")
    def test_default_camera_planner_returns_hover_without_inference(self):
        with patch("prototype.chat") as model, patch("integrated_control.assess_image") as vision, \
             patch("integrated_control.assess_monocular_depth") as depth:
            result = plan_request(camera_request(text="Hover"))
        self.assertEqual(result["request_id"], "request-1")
        self.assertEqual([result["command"][key] for key in ("forward", "right", "up", "yaw")], [0, 0, 0, 0])
        for stage in (model, vision, depth):
            stage.assert_not_called()

    def test_disconnected_client_does_not_trigger_second_response(self):
        handler = SimpleNamespace(send_response=Mock(), send_header=Mock(), end_headers=Mock(),
                                  wfile=SimpleNamespace(write=Mock(side_effect=BrokenPipeError())), close_connection=False)
        with self.assertLogs("drone_vla.http", level="WARNING"):
            VlaHandler.reply(handler, 200, {"ok": True})
        self.assertTrue(handler.close_connection)
        handler.send_response.assert_called_once_with(200)

    @unittest.skipUnless(importlib.util.find_spec("pydantic") and importlib.util.find_spec("ollama")
                         and importlib.util.find_spec("PIL"), "Planner dependencies unavailable")
    def test_truncated_model_output_is_rejected(self):
        import camera_test
        import prototype
        import vision_test
        truncated = SimpleNamespace(done_reason="length", message=SimpleNamespace(content="{}"))
        with patch("camera_test.Path.exists", return_value=True), patch("camera_test.chat", return_value=truncated), self.assertRaisesRegex(ValueError, "token budget"):
            camera_test.describe_image("frame.png")
        with patch("prototype.chat", return_value=truncated), self.assertRaisesRegex(ValueError, "token budget"):
            prototype.interpret_command("Move forward at 1 m/s")
        with patch("vision_test.describe_image", return_value="wall"), patch("vision_test.chat", return_value=truncated), self.assertRaisesRegex(ValueError, "token budget"):
            vision_test.assess_image("frame.png")

    def test_sdk_signs(self):
        result = joystick_command(motion(longitudinal_direction="backward", longitudinal_speed_mps=1.0,
                                         lateral_direction="left", lateral_speed_mps=1.0,
                                         vertical_direction="down", vertical_speed_mps=0.5,
                                         yaw_direction="counterclockwise", yaw_speed_deg_s=15))
        self.assertEqual([result[k] for k in ("forward", "right", "up", "yaw")], [-1, -1, -0.5, -15])

    def test_rejections(self):
        cases = [dict(valid=False, reason="ambiguous"), dict(duration_seconds=3), dict(duration_seconds=float("nan")),
                 dict(longitudinal_speed_mps=1), dict(vertical_direction="up", vertical_speed_mps=2),
                 dict(longitudinal_direction="forward", longitudinal_speed_mps=2,
                      lateral_direction="right", lateral_speed_mps=2),
                 dict(yaw_direction="clockwise", yaw_speed_deg_s=float("inf"))]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(ValueError):
                joystick_command(motion(**values))

    def test_cli_submits_only_text(self):
        client = FakeClient()
        self.assertEqual(execute(client, "forward")["state"], "executing")
        self.assertEqual(client.sent, [dict(op="instruction", text="forward", duration_seconds=1,
                                           session="a", ticket="ticket")])

    def test_cli_wait_can_be_disabled_but_still_reports_bridge_failure(self):
        with patch("dji_control.time.monotonic", side_effect=AssertionError("No deadline should be read")):
            self.assertEqual(execute(FakeClient(), "Hover", timeout=0)["state"], "executing")
            with self.assertRaisesRegex(RuntimeError, "Model unavailable"):
                execute(FakeClient(state="error"), "Hover", timeout=0)
        for timeout in (-1, float("inf"), float("nan")):
            client = FakeClient()
            with self.assertRaisesRegex(ValueError, "Wait timeout"):
                execute(client, "Hover", timeout=timeout)
            self.assertEqual(client.sent, [])

    def test_session_change_reports_discard(self):
        with self.assertRaisesRegex(RuntimeError, "session"):
            execute(FakeClient(("a", "b")), "forward")

    def test_inference_failure_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "Model unavailable"):
            execute(FakeClient(state="error"), "forward")

    def test_invalid_camera_requests(self):
        for body in (b"", camera_request()[:-1], camera_request(width=100), camera_request(format="YUV"),
                     camera_request(text=""), camera_request(default_duration=float("nan"))):
            with self.subTest(body=body[:20]), self.assertRaises(ValueError):
                decode_request(body)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow unavailable")
    def test_received_camera_pixels_reach_planner(self):
        from PIL import Image
        def planner(text, path):
            self.assertEqual(text, "forward")
            with Image.open(path) as frame:
                self.assertEqual(list(frame.getdata()), [(255, 0, 0), (0, 255, 0)])
            return SimpleNamespace(command=motion(), override_reason="Path blocked")
        result = plan_request(camera_request(), planner)
        self.assertEqual(result["request_id"], "request-1")
        self.assertEqual(result["command"]["forward"], 0)
        self.assertEqual(result["override_reason"], "Path blocked")

    @unittest.skipUnless(importlib.util.find_spec("pydantic") and importlib.util.find_spec("ollama")
                         and importlib.util.find_spec("PIL"), "Planner dependencies unavailable")
    def test_real_planner_blocked_camera_sends_zero(self):
        from integrated_control import plan_motion
        from prototype import DroneMotionCommand
        from vision_test import VisualSceneAssessment
        requested = DroneMotionCommand(**vars(motion(longitudinal_direction="forward", longitudinal_speed_mps=1)))
        blocked = VisualSceneAssessment(path_status="blocked", obstacle_present=True, obstacle_description="wall",
                                        central_region_description="wall", confidence=1.0, explanation="Path blocked by wall")
        with patch("integrated_control.interpret_command", return_value=requested), patch("integrated_control.assess_image", return_value=blocked), patch("integrated_control.assess_monocular_depth", return_value=None):
            result = plan_request(camera_request(), planner=plan_motion)
        self.assertEqual(result["command"]["forward"], 0)
        self.assertIn("suppressed", result["override_reason"])


if __name__ == "__main__":
    unittest.main()
