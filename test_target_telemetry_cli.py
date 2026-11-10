"""Live target observations survive CLI printing and metrics collection."""
from copy import deepcopy
import json
import socket
import unittest
from unittest.mock import Mock, patch

from dji_control import DjiClient, MAX_RESPONSE_BYTES
from navigation_control import RESULT_PREFIX, run_legacy_navigation, run_navigation
from test_navigation import Clock


def target(center_x=320, visibility="tracked"):
    return dict(instruction="Approach the blue building", visibility=visibility, fresh=True,
                box_normalized=[0.25, 0.25, 0.75, 0.75],
                center_px=[center_x, 180], box_px=[160, 90, 480, 270],
                image_size_px=[640, 360], bearing_deg=3.5, distance_m=12.1,
                forward_distance_m=12.0, position_body_m=dict(forward=12.0, right=1.5, up=0.0))


class TelemetryClient:
    """Intentionally reuse a telemetry object to exercise stored snapshots."""
    def __init__(self, observations):
        self.observations = iter(observations)
        self.target = target()
        self.current = dict(ok=True, eligible=True, bridge_protocol=2, continuous_navigation=True,
                            session="manual", ticket="ticket", request_id="initial", state="idle")
        self.goal = None
        self.sent = []
        self.last_state = "arrived"

    def request(self, payload):
        self.sent.append(payload)
        if payload["op"] == "navigate":
            self.goal = payload["goal_id"]
            self.current.update(request_id="goal-request", state="executing")
        return dict(ok=True, request_id=self.current["request_id"])

    def status(self):
        if self.goal:
            state, center = next(self.observations, ("arrived", self.target["center_px"][0]))
            self.target["center_px"][0] = center
            self.current["navigation"] = dict(goal_id=self.goal, sequence=7, state=state,
                                             reason="Live target", timings_ms={}, target=self.target)
            self.current["command"] = dict(forward=0.5 if state == "approaching" else 0,
                                           right=0, up=0, yaw=0)
        return dict(self.current)


class TargetTelemetryClientTests(unittest.TestCase):
    def test_continuous_emits_target_changes_when_axes_and_sequence_are_unchanged(self):
        client = TelemetryClient([("approaching", 320), ("approaching", 350), ("arrived", 350)])
        clock, lines = Clock(), []
        samples = run_navigation(client, "Approach the blue building", emit=lines.append,
                                 clock=clock, sleep=clock.sleep)
        self.assertEqual([row["target"]["center_px"][0] for row in samples], [320, 350, 350])
        self.assertEqual([row["target"] for row in samples],
                         [json.loads(line)["target"] for line in lines])
        self.assertEqual(samples[0]["target"]["position_body_m"], target()["position_body_m"])
        self.assertEqual([request["op"] for request in client.sent], ["navigate"])
        client.target["position_body_m"]["right"] = 99
        self.assertEqual(samples[0]["target"]["position_body_m"]["right"], 1.5)

    def test_unchanged_target_does_not_duplicate_identical_status(self):
        client = TelemetryClient([("approaching", 320), ("approaching", 320), ("arrived", 320)])
        clock, lines = Clock(), []
        samples = run_navigation(client, "Approach the blue building", emit=lines.append,
                                 clock=clock, sleep=clock.sleep)
        self.assertEqual([row["state"] for row in samples], ["approaching", "arrived"])
        self.assertEqual(len(lines), 2)

    def test_visibility_changes_are_reported_without_pixel_change(self):
        client = TelemetryClient([("approaching", 320), ("approaching", 320), ("arrived", 320)])
        clock, lines = Clock(), []
        calls = 0
        def sleep(seconds):
            nonlocal calls
            clock.sleep(seconds)
            calls += 1
            if calls == 1:
                client.target["visibility"] = "partial"
        samples = run_navigation(client, "Approach the blue building", emit=lines.append,
                                 clock=clock, sleep=sleep)
        self.assertEqual([row["target"]["visibility"] for row in samples], ["tracked", "partial", "partial"])

    def test_native_watchdog_invalidates_last_observation_without_mutating_input(self):
        client = TelemetryClient([("approaching", 320), ("approaching", 320), ("arrived", 320)])
        clock, lines = Clock(), []
        calls = 0
        def sleep(seconds):
            nonlocal calls
            clock.sleep(seconds)
            calls += 1
            client.current["state"] = "stale" if calls == 1 else "executing"
        samples = run_navigation(client, "Approach the blue building", emit=lines.append,
                                 clock=clock, sleep=sleep)
        self.assertEqual([row["target"]["fresh"] for row in samples], [True, False, True])
        self.assertEqual([json.loads(line)["target"]["fresh"] for line in lines], [True, False, True])
        self.assertTrue(client.target["fresh"])

    def test_navigation_hold_states_invalidate_copied_target(self):
        for state in ("stale", "lost", "error", "cancelled"):
            with self.subTest(state=state):
                client = TelemetryClient([(state, 320), ("arrived", 320)])
                clock, lines = Clock(), []
                try:
                    run_navigation(client, "Approach the blue building", emit=lines.append,
                                   clock=clock, sleep=clock.sleep)
                except RuntimeError:
                    self.assertEqual(state, "error")
                self.assertFalse(json.loads(lines[0])["target"]["fresh"])
                self.assertTrue(client.target["fresh"])

    def test_legacy_forwards_complete_target_in_printed_and_returned_metrics(self):
        clock, lines = Clock(), []
        current = dict(session="manual", request_id="initial")
        class Client:
            def status(self):
                return dict(current)
        observation = target()
        def execute(owned, envelope, *args, **kwargs):
            message = json.loads(envelope[len("@nav1:"):])
            info = dict(goal_id=message["goal_id"], sequence=message["sequence"],
                        state="arrived", reason="At destination", timings_ms={}, target=deepcopy(observation))
            return dict(request_id="step-request", override_reason=RESULT_PREFIX+json.dumps(info),
                        command=dict(forward=0, right=0, up=0, yaw=0))
        with patch("navigation_control.execute", side_effect=execute):
            samples = run_legacy_navigation(Client(), "Approach the blue building", emit=lines.append,
                                            clock=clock, sleep=clock.sleep)
        self.assertEqual(samples[0]["target"], observation)
        self.assertEqual(json.loads(lines[0])["target"], observation)

    def test_legacy_native_watchdog_does_not_present_old_target_as_fresh(self):
        clock, lines = Clock(), []
        class Client:
            def status(self):
                return dict(session="manual", request_id="initial")
        observation = target()
        def execute(owned, envelope, *args, **kwargs):
            message = json.loads(envelope[len("@nav1:"):])
            info = dict(goal_id=message["goal_id"], sequence=message["sequence"],
                        state="lost", reason="Tracking ended", timings_ms={}, target=observation)
            return dict(request_id="step-request", state="stale", override_reason=RESULT_PREFIX+json.dumps(info),
                        command=dict(forward=0, right=0, up=0, yaw=0))
        with patch("navigation_control.execute", side_effect=execute):
            samples = run_legacy_navigation(Client(), "Approach the blue building", emit=lines.append,
                                            clock=clock, sleep=clock.sleep)
        self.assertFalse(samples[0]["target"]["fresh"])
        self.assertFalse(json.loads(lines[0])["target"]["fresh"])
        self.assertTrue(observation["fresh"])


class TargetStatusDatagramTests(unittest.TestCase):
    def receive(self, datagram):
        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        client.recv.side_effect = lambda capacity: datagram[:capacity]
        with patch("dji_control.socket.socket", return_value=client):
            response = DjiClient("/run/dji-vla/control.sock").request({"op": "status"})
        client.recv.assert_called_once_with(MAX_RESPONSE_BYTES+1)
        client.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_RCVBUF, MAX_RESPONSE_BYTES)
        return response

    def test_large_utf8_status_with_duplicated_navigation_is_received_whole(self):
        observation = target()
        observation["instruction"] = "接近蓝色建筑"*80
        navigation = dict(state="approaching", sequence=100, target=observation)
        expected = dict(ok=True, navigation=navigation,
                        override_reason=RESULT_PREFIX+json.dumps(navigation))
        datagram = json.dumps(expected).encode()
        self.assertGreater(len(datagram), 4096)
        self.assertLess(len(datagram), MAX_RESPONSE_BYTES)
        self.assertEqual(self.receive(datagram), expected)

    def test_exact_response_bound_is_accepted(self):
        data = b'{"ok":true}'
        self.assertEqual(self.receive(data + b" "*(MAX_RESPONSE_BYTES-len(data))), {"ok": True})

    def test_oversized_datagram_is_rejected_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "65536-byte limit"):
            self.receive(b'{"ok":true}' + b" "*MAX_RESPONSE_BYTES)


if __name__ == "__main__":
    unittest.main()
