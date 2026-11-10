"""Compact operator text, full metrics, and final validated command logging."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from navigation_control import main
from navigation_logging import compact_navigation_line
from test_bridge_protocol import native_metadata, native_response
from test_navigation import body
from vla_service import plan_request


def observation():
    return dict(state="aligning", sequence=519, bridge_protocol=2,
                since_goal_ms=1234, status_round_trip_ms=3.1,
                command=dict(forward=0.0, right=0.0, up=0.0, yaw=3.64, duration_seconds=0.35),
                service_timings_ms=dict(service_ms=216.11, decoded_frame_age_ms=230),
                target=dict(instruction="Approach that black building", visibility="visible", fresh=True,
                            box_px=[661.18, 365.53, 1449.7, 933.45], center_px=[1055.44, 649.49],
                            distance_m=8.802, bearing_deg=4.552, tracking_confidence=0.935396671,
                            grounding_confidence=0.95, frame_id=51313, frame_monotonic_ns=3200470818418))


class NavigationLoggingTests(unittest.TestCase):
    def test_compact_line_keeps_control_and_target_without_bridge_identifiers(self):
        row = observation()
        before = deepcopy(row)
        line = compact_navigation_line(row)
        self.assertIn("aligning cmd(f=+0.00 r=+0.00 u=+0.00 yaw=+3.64)", line)
        self.assertIn("bbox=[661,366,1450,933] center=[1055,649]", line)
        self.assertIn("conf=0.935 range=8.80m bearing=+4.55deg visible 216ms", line)
        for detail in ("bridge_protocol", "frame_monotonic_ns", "51313", "519", "instruction"):
            self.assertNotIn(detail, line)
        self.assertEqual(row, before)

    def test_uncertain_measurement_does_not_print_stale_pixels_as_current(self):
        row = observation()
        row.update(state="stale", reason="Destination position is uncertain; holding")
        row["target"].update(visibility="unconfirmed", fresh=False, box_px=None,
                              center_px=None, distance_m=None, tracking_confidence=None)
        line = compact_navigation_line(row)
        self.assertIn("bbox=- center=-", line)
        self.assertIn("conf=0.950 range=-m", line)
        self.assertIn("unconfirmed target=stale", line)
        self.assertIn("Destination position is uncertain; holding", line)

    def cli(self, verbose=False):
        row = observation()
        def run(client, instruction, max_seconds, interval, duration, emit):
            emit(json.dumps(row))
            return [row]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)/"metrics.json"
            arguments = ["navigation_control.py", "Approach that building", "--metrics", str(destination)]
            if verbose:
                arguments.append("--verbose-json")
            output = io.StringIO()
            with patch("sys.argv", arguments), patch("sys.stdout", output), \
                 patch("navigation_control.DjiClient"), patch("navigation_control.run_navigation", side_effect=run):
                main()
            return output.getvalue().splitlines(), json.loads(destination.read_text()), row

    def test_cli_default_is_compact_but_export_retains_full_observation(self):
        output, metrics, row = self.cli()
        self.assertEqual(output[0], compact_navigation_line(row))
        self.assertEqual(metrics, [row])
        self.assertEqual(metrics[0]["target"]["frame_id"], 51313)
        self.assertEqual(metrics[0]["bridge_protocol"], 2)

    def test_cli_verbose_json_preserves_original_machine_readable_output(self):
        output, metrics, row = self.cli(verbose=True)
        self.assertEqual(json.loads(output[0]), row)
        self.assertEqual(metrics, [row])

    def test_http_logs_final_expired_frame_hold_instead_of_prevalidation_motion(self):
        metadata = native_metadata()
        result = native_response(metadata)
        result["navigation"]["target"] = observation()["target"]
        navigation = Mock()
        navigation.handle.return_value = result
        payload = body(metadata["text"], **{key: value for key, value in metadata.items() if key != "text"})
        with patch("bridge_protocol.frame_age_seconds", side_effect=[0.1, 0.8]), \
             self.assertLogs("drone_vla", level="INFO") as logs:
            actual = plan_request(payload, navigation=navigation)
        self.assertEqual(actual["navigation"]["state"], "stale")
        self.assertEqual(actual["command"]["forward"], 0)
        self.assertEqual(len(logs.output), 1)
        self.assertIn("stale cmd(f=+0.00", logs.output[0])
        self.assertIn("target=stale", logs.output[0])
        self.assertNotIn("request1", logs.output[0])


if __name__ == "__main__":
    unittest.main()
