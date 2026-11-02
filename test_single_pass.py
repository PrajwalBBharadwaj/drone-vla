import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from dji_control import joystick_command
from single_pass import SceneMotion, plan_frame
from vla_service import plan_request
from test_dji_control import camera_request


def model_response(**updates):
    values = dict(valid=True, forward=0.5, right=0, up=0, yaw=0,
                  path="clear", confidence=0.95, obstacle=False, reason="Open corridor")
    values.update(updates)
    return SimpleNamespace(done_reason="stop", message=SimpleNamespace(content=SceneMotion(**values).model_dump_json()))


class SinglePassTests(unittest.TestCase):
    def test_hover_needs_neither_model_nor_depth(self):
        with patch("single_pass.Client") as client, patch("single_pass.assess_monocular_depth") as depth:
            self.assertEqual(joystick_command(plan_frame("Hover", Image.new("RGB", (640, 360))).command)["forward"], 0)
        client.assert_not_called()
        depth.assert_not_called()

    def test_one_multimodal_call_receives_live_pixels_without_png_file(self):
        def inspect(**kwargs):
            images = kwargs["messages"][1]["images"]
            image = Image.open(io.BytesIO(images[0]))
            self.assertEqual(image.size, (2, 1))
            self.assertEqual(kwargs["keep_alive"], -1)
            return model_response(forward=0, yaw=10)
        with patch("single_pass.Client") as client, patch("single_pass.assess_monocular_depth") as depth, \
             patch("vla_service.tempfile.TemporaryDirectory", side_effect=AssertionError("No temporary image file")):
            client.return_value.chat.side_effect = inspect
            result = plan_request(camera_request(text="Rotate clockwise at 10 deg/s"), planner=plan_frame)
        client.return_value.chat.assert_called_once()
        depth.assert_not_called()
        self.assertEqual(result["command"]["yaw"], 10)

    def test_blocked_uncertain_and_obstacle_all_suppress_motion(self):
        for values in ({"path": "blocked"}, {"path": "uncertain"}, {"obstacle": True}):
            with self.subTest(values=values), patch("single_pass.Client") as client, \
                 patch("single_pass.assess_monocular_depth") as depth:
                client.return_value.chat.return_value = model_response(**values)
                command = joystick_command(plan_frame("Forward at 0.5 m/s", Image.new("RGB", (640, 360))).command)
                self.assertEqual([command[axis] for axis in ("forward", "right", "up", "yaw")], [0, 0, 0, 0])
                depth.assert_not_called()

    def test_truncation_and_unobserved_motion_rejected(self):
        for response in (SimpleNamespace(done_reason="length"), model_response(right=0.5),
                         model_response(forward=-0.5), model_response(up=0.5), model_response(valid=False)):
            with self.subTest(response=response), patch("single_pass.Client") as client:
                client.return_value.chat.return_value = response
                with self.assertRaises(ValueError):
                    plan_frame("Move", Image.new("RGB", (640, 360)))

    def test_clear_forward_reuses_in_memory_depth_safeguards(self):
        frame = Image.new("RGB", (640, 360))
        with patch("single_pass.Client") as client, patch("single_pass.assess_monocular_depth", return_value="depth") as depth, \
             patch("single_pass.apply_scene_safeguards", return_value="decision") as safeguards:
            client.return_value.chat.return_value = model_response()
            self.assertEqual(plan_frame("Forward at 0.5 m/s", frame), "decision")
        depth.assert_called_once_with(frame)
        self.assertEqual(safeguards.call_args.args[0].longitudinal_speed_mps, 0.5)


if __name__ == "__main__":
    unittest.main()
