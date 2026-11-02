"""The terminal submits once; native updates and cancellation preserve ownership."""
import itertools
import unittest
from unittest.mock import Mock

from navigation_control import run_navigation
from test_navigation import Clock


class NativeClient:
    def __init__(self, phases=("grounding", "ready", "approaching", "arrived")):
        self.phases = iter(phases)
        self.current = dict(ok=True, eligible=True, bridge_protocol=2, continuous_navigation=True,
                            session="manual", ticket="ticket", request_id="initial", state="idle")
        self.sent = []
        self.goal = None
        self.sequence = 0

    def status(self):
        if self.goal:
            state = next(self.phases, "arrived")
            self.current["navigation"] = dict(goal_id=self.goal, state=state, sequence=self.sequence,
                                                reason="Test state", timings_ms={"decoded_frame_age_ms": 100})
            self.current["command"] = dict(forward=0.5 if state == "approaching" else 0, right=0, up=0, yaw=0)
            self.sequence += 1
        return dict(self.current)

    def request(self, payload):
        self.sent.append(payload)
        if payload["op"] == "navigate":
            self.goal = payload["goal_id"]
            self.current.update(request_id="goal-request", state="executing")
        elif payload["op"] == "cancel":
            self.current["state"] = "cancelled"
        return dict(ok=True, request_id=self.current["request_id"])


class ContinuousClientTests(unittest.TestCase):
    def test_one_submission_monitors_independent_updates_without_step_requests(self):
        client, clock = NativeClient(), Clock()
        result = run_navigation(client, "Approach the blue building", emit=Mock(), clock=clock, sleep=clock.sleep)
        self.assertEqual([row["state"] for row in result], ["grounding", "ready", "approaching", "arrived"])
        self.assertEqual([request["op"] for request in client.sent], ["navigate"])
        self.assertEqual(client.sent[0]["session"], "manual")
        self.assertEqual(client.sent[0]["ticket"], "ticket")
        self.assertEqual(client.sent[0]["duration_seconds"], 0.35)
        self.assertEqual(client.sent[0]["max_seconds"], 120)

    def test_old_bridge_requires_explicit_legacy_mode_before_any_action(self):
        client = NativeClient()
        client.current["bridge_protocol"] = 1
        with self.assertRaisesRegex(RuntimeError, "rebuilt"):
            run_navigation(client, "Approach the building")
        self.assertEqual(client.sent, [])

    def test_ctrl_c_cancels_owned_goal_without_model_or_hover_request(self):
        client = NativeClient(phases=itertools.repeat("grounding"))
        def interrupt(seconds):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            run_navigation(client, "Approach the building", emit=Mock(), sleep=interrupt)
        self.assertEqual([request["op"] for request in client.sent], ["navigate", "cancel"])
        self.assertEqual(client.sent[1]["goal_id"], client.sent[0]["goal_id"])
        self.assertEqual(client.sent[1]["request_id"], "goal-request")

    def test_takeover_never_sends_cancel_for_another_request_or_session(self):
        for field in ("session", "request_id"):
            with self.subTest(field=field):
                client = NativeClient(phases=itertools.repeat("approaching"))
                clock = Clock()
                def takeover(seconds):
                    clock.sleep(seconds)
                    client.current[field] = "operator"
                with self.assertRaisesRegex(RuntimeError, "session changed|superseded"):
                    run_navigation(client, "Approach building", emit=Mock(), clock=clock, sleep=takeover)
                self.assertEqual([request["op"] for request in client.sent], ["navigate"])

    def test_deadline_cancels_owned_goal(self):
        client, clock = NativeClient(phases=itertools.repeat("grounding")), Clock()
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            run_navigation(client, "Approach building", max_seconds=0.2, emit=Mock(), clock=clock, sleep=clock.sleep)
        self.assertEqual([request["op"] for request in client.sent], ["navigate", "cancel"])

    def test_bad_status_cancels_known_owned_request(self):
        client = NativeClient()
        original_status = client.status
        def bad_status():
            status = original_status()
            if client.goal:
                status["navigation"] = {}
            return status
        client.status = bad_status
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            run_navigation(client, "Approach building", emit=Mock())
        self.assertEqual([request["op"] for request in client.sent], ["navigate", "cancel"])

    def test_invalid_input_does_not_contact_bridge(self):
        for updates in ({"text": " "}, {"text": "中"*300}, {"max_seconds": float("nan")},
                        {"duration": 0.5}, {"interval": 0.01}):
            with self.subTest(updates=updates):
                client = Mock()
                values = dict(text="Approach building")
                values.update(updates)
                with self.assertRaises(ValueError):
                    run_navigation(client, **values)
                client.status.assert_not_called()


if __name__ == "__main__":
    unittest.main()
