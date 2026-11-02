"""Native frame correlation, command freshness, and continuous goal regressions.

These tests inject clocks and perception. They exercise the on-wire checks and
goal lifecycle without downloading models, opening cameras, or commanding DJI.
"""
import json
import math
import unittest
from unittest.mock import Mock

import numpy as np

from bridge_protocol import CONTEXT_FIELDS, correlate_response, validate_context
from navigation import NavigationEngine, encode_message, zero_axes
from test_navigation import BOX, Clock, ControlledExecutor, configuration, texture


NOW_NS = 50_000_000_000


def native_metadata(sequence=1, **updates):
    envelope = dict(op="step", goal_id="goal1", owner="manual1", sequence=sequence)
    if sequence == 0:
        envelope.update(op="start", text="Approach the blue building")
    metadata = dict(
        bridge_protocol=2, session="manual1", goal_id="goal1", frame_id=sequence+1,
        frame_monotonic_ns=NOW_NS-100_000_000, request_id="request1",
        default_duration=0.35, text=encode_message(**envelope),
    )
    metadata.update(updates)
    return metadata


def native_response(metadata=None, **updates):
    metadata = metadata or native_metadata()
    message = json.loads(metadata["text"][len("@nav1:"):])
    response = dict(
        ok=True, request_id=metadata["request_id"],
        command=dict(forward=0.5, right=0.0, up=0.0, yaw=0.0, duration_seconds=0.35),
        navigation=dict(goal_id=metadata["goal_id"], sequence=message["sequence"],
                        state="approaching", reason="Observed corridor is clear", timings_ms={}),
    )
    response.update(updates)
    return response


class ContextTests(unittest.TestCase):
    def test_legacy_context_is_not_reinterpreted(self):
        self.assertIsNone(validate_context({"text": "Hover"}, now_ns=NOW_NS))

    def test_valid_context_echoes_only_correlation_fields(self):
        metadata = native_metadata()
        context = validate_context(metadata, now_ns=NOW_NS)
        self.assertEqual(context, {key: metadata[key] for key in CONTEXT_FIELDS})

    def test_missing_required_context_fields_are_rejected(self):
        for key in ("session", "goal_id", "frame_id", "frame_monotonic_ns", "text"):
            with self.subTest(key=key):
                metadata = native_metadata()
                del metadata[key]
                with self.assertRaises(ValueError):
                    validate_context(metadata, now_ns=NOW_NS)

    def test_protocol_and_integer_fields_reject_coercion_and_overflow(self):
        for key, value in (
            ("bridge_protocol", True), ("bridge_protocol", 1), ("bridge_protocol", "2"),
            ("frame_id", True), ("frame_id", 0), ("frame_id", -1),
            ("frame_id", 2**63), ("frame_id", 1.0), ("frame_id", "1"),
            ("frame_monotonic_ns", False), ("frame_monotonic_ns", 0),
            ("frame_monotonic_ns", 2**63), ("frame_monotonic_ns", 1.0),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_context(native_metadata(**{key: value}), now_ns=NOW_NS)

    def test_context_identity_strings_are_bounded(self):
        for key, value in (("session", ""), ("session", "s"*129), ("session", 1),
                           ("goal_id", ""), ("goal_id", "g"*65), ("goal_id", None)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_context(native_metadata(**{key: value}), now_ns=NOW_NS)

    def test_envelope_must_be_navigation_for_the_same_owner_and_goal(self):
        invalid_texts = [
            "Hover", "@nav1:{", "@nav1:[]", "@nav1:null",
            encode_message(op="step", goal_id="other", owner="manual1", sequence=1),
            encode_message(op="step", goal_id="goal1", owner="other", sequence=1),
        ]
        for text in invalid_texts:
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_context(native_metadata(text=text), now_ns=NOW_NS)

    def test_source_age_and_future_tolerance_boundaries(self):
        for offset_ns in (-750_000_000, 20_000_000):
            with self.subTest(allowed_offset_ns=offset_ns):
                self.assertIsNotNone(validate_context(
                    native_metadata(frame_monotonic_ns=NOW_NS+offset_ns), now_ns=NOW_NS))
        for offset_ns in (-750_000_001, 20_000_001):
            with self.subTest(rejected_offset_ns=offset_ns), self.assertRaises(ValueError):
                validate_context(native_metadata(frame_monotonic_ns=NOW_NS+offset_ns), now_ns=NOW_NS)


class ResponseTests(unittest.TestCase):
    def test_active_response_echoes_native_context_and_preserves_valid_axes(self):
        metadata = native_metadata()
        response = correlate_response(native_response(metadata), metadata, now_ns=NOW_NS)
        self.assertEqual({key: response[key] for key in CONTEXT_FIELDS},
                         {key: metadata[key] for key in CONTEXT_FIELDS})
        self.assertEqual(response["command"]["forward"], 0.5)
        self.assertEqual(response["navigation"]["timings_ms"]["decoded_frame_age_ms"], 100)
        self.assertEqual(json.loads(response["override_reason"][len("navigation:"):]),
                         response["navigation"])

    def test_response_request_goal_and_sequence_must_match(self):
        for case in ("ok", "request", "goal", "sequence", "sequence_bool", "missing_navigation"):
            with self.subTest(case=case):
                response = native_response()
                if case == "ok":
                    response["ok"] = False
                elif case == "request":
                    response["request_id"] = "another-request"
                elif case == "goal":
                    response["navigation"]["goal_id"] = "another-goal"
                elif case == "sequence":
                    response["navigation"]["sequence"] += 1
                elif case == "sequence_bool":
                    response["navigation"]["sequence"] = True
                else:
                    del response["navigation"]
                with self.assertRaises(ValueError):
                    correlate_response(response, native_metadata(), now_ns=NOW_NS)

    def test_every_hold_state_suppresses_even_a_nonzero_model_command(self):
        for state in ("arrived", "blocked", "lost", "error", "cancelled",
                      "grounding", "acquiring", "ready", "stale"):
            with self.subTest(state=state):
                response = native_response()
                response["navigation"]["state"] = state
                result = correlate_response(response, native_metadata(), now_ns=NOW_NS)
                self.assertEqual(result["command"], zero_axes(0.35))
                self.assertEqual(result["navigation"]["state"], state)

    def test_unknown_state_is_rejected(self):
        response = native_response()
        response["navigation"]["state"] = "flying"
        with self.assertRaises(ValueError):
            correlate_response(response, native_metadata(), now_ns=NOW_NS)

    def test_nonfinite_bool_and_missing_axes_are_rejected(self):
        for key in ("forward", "right", "up", "yaw", "duration_seconds"):
            for value in (float("nan"), float("inf"), -float("inf"), True, "0", None):
                with self.subTest(key=key, value=value):
                    response = native_response()
                    response["command"][key] = value
                    with self.assertRaises(ValueError):
                        correlate_response(response, native_metadata(), now_ns=NOW_NS)
            with self.subTest(missing_axis=key):
                response = native_response()
                del response["command"][key]
                with self.assertRaises(ValueError):
                    correlate_response(response, native_metadata(), now_ns=NOW_NS)

    def test_speed_limits_apply_to_combined_horizontal_motion(self):
        cases = [dict(forward=2.01), dict(right=-2.01), dict(forward=1.5, right=1.5),
                 dict(up=1.01), dict(up=-1.01), dict(yaw=30.01), dict(yaw=-30.01),
                 dict(duration_seconds=0), dict(duration_seconds=-0.1), dict(duration_seconds=0.50001)]
        for updates in cases:
            with self.subTest(updates=updates):
                response = native_response()
                response["command"].update(updates)
                with self.assertRaises(ValueError):
                    correlate_response(response, native_metadata(), now_ns=NOW_NS)

    def test_expired_source_after_inference_becomes_a_zero_stale_response(self):
        metadata = native_metadata(frame_monotonic_ns=NOW_NS)
        validate_context(metadata, now_ns=NOW_NS)
        result = correlate_response(native_response(metadata), metadata, now_ns=NOW_NS+750_000_001)
        self.assertEqual(result["command"], zero_axes(0.35))
        self.assertEqual(result["navigation"]["state"], "stale")
        self.assertIn("freshness", result["navigation"]["reason"])
        self.assertEqual(json.loads(result["override_reason"][len("navigation:"):])["state"], "stale")

    def test_source_at_exact_age_limit_remains_valid(self):
        metadata = native_metadata(frame_monotonic_ns=NOW_NS)
        result = correlate_response(native_response(metadata), metadata, now_ns=NOW_NS+750_000_000)
        self.assertEqual(result["navigation"]["state"], "approaching")
        self.assertEqual(result["command"]["forward"], 0.5)

    def test_clock_inconsistency_after_validation_also_holds(self):
        metadata = native_metadata(frame_monotonic_ns=NOW_NS)
        result = correlate_response(native_response(metadata), metadata, now_ns=NOW_NS-20_000_001)
        self.assertEqual(result["command"], zero_axes(0.35))
        self.assertEqual(result["navigation"]["state"], "stale")


class FixedTracker:
    """Inject only tracking; actual geometry and state transitions stay in use."""
    def start(self, frame, box):
        self.box = box

    def update(self, frame):
        return self.box


class ContinuousEngineTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.executor = ControlledExecutor()
        self.depth = Mock(return_value=np.full((252, 448), 30, dtype=np.float32))
        self.engine = NavigationEngine(configuration(), grounder=Mock(), depth=self.depth,
                                       tracker_factory=FixedTracker, clock=self.clock, executor=self.executor)
        self.frame = texture()

    def tearDown(self):
        self.engine.close()

    def step(self, sequence, advance=0.2, **updates):
        self.clock.sleep(advance)
        metadata = native_metadata(sequence, frame_monotonic_ns=NOW_NS+round(self.clock()*1e9), **updates)
        return self.engine.handle(metadata, self.frame)

    def acquired(self):
        self.assertEqual(self.step(0)["navigation"]["state"], "grounding")
        self.executor.finish()
        result = self.step(1)
        self.assertEqual(result["navigation"]["state"], "ready")
        self.assertEqual(result["command"], zero_axes(0.35))
        self.depth.assert_not_called()

    def test_forward_sequence_gaps_from_transport_loss_are_accepted(self):
        self.acquired()
        for sequence in (3, 7, 8):
            with self.subTest(sequence=sequence):
                result = self.step(sequence)
                self.assertEqual(result["navigation"]["state"], "approaching")
                self.assertEqual(result["command"]["forward"], 0.5)
                self.assertEqual(result["command"]["duration_seconds"], 0.35)
        self.assertEqual(self.executor.submissions, 1)

    def test_repeated_or_backward_sequences_hold_instead_of_reusing_motion(self):
        for sequence in (1, 0):
            with self.subTest(sequence=sequence):
                self.setUp()
                self.acquired()
                # A backward step uses op=step rather than a new start envelope.
                text = encode_message(op="step", goal_id="goal1", owner="manual1", sequence=sequence)
                result = self.step(2, text=text)
                self.assertEqual(result["navigation"]["state"], "error")
                self.assertEqual(result["command"], zero_axes(0.35))
                self.depth.assert_not_called()
                self.engine.close()

    def test_source_frame_id_or_timestamp_replay_holds_despite_new_request_sequence(self):
        for case in ("frame_id", "timestamp", "source_gap"):
            with self.subTest(case=case):
                self.setUp()
                self.acquired()
                previous = self.engine.goal
                updates = {"frame_id": previous.last_frame_id} if case == "frame_id" else {}
                delta_ns = {"frame_id": 200_000_000, "timestamp": 0,
                            "source_gap": 1_100_000_000}[case]
                metadata = native_metadata(2, frame_monotonic_ns=previous.last_source_ns+delta_ns, **updates)
                self.clock.sleep(0.2)
                result = self.engine.handle(metadata, self.frame)
                self.assertEqual(result["navigation"]["state"], "lost" if case == "source_gap" else "error")
                self.assertEqual(result["command"], zero_axes(0.35))
                self.depth.assert_not_called()
                self.engine.close()

    def test_high_frequency_grounding_history_remains_bounded_until_deadline(self):
        # Thirty requests per second used to exhaust 256 frames in 8.5 seconds.
        # Grounding may now last 60 seconds; sampled history must retain its
        # original reference while accepting fresh observations throughout.
        self.engine.config = configuration(processing_width=224)
        result = self.step(0, advance=0)
        first_pixels = np.asarray(self.engine.goal.frames[0][1]).copy()
        for sequence in range(1, 1771):
            result = self.step(sequence, advance=1/30)
            self.assertEqual(result["navigation"]["state"], "grounding")
            self.assertEqual(result["command"], zero_axes(0.35))
        history = self.engine.goal.frames
        self.assertGreater(len(history), 256)
        self.assertLessEqual(len(history), math.ceil(60/0.2)+4)
        self.assertEqual(history[0][0], 0)
        np.testing.assert_array_equal(np.asarray(history[0][1]), first_pixels)
        self.depth.assert_not_called()
        self.executor.finish()
        # Replay/acquisition holds even after the delayed grounding completes.
        result = self.step(1771, advance=1/30)
        self.assertEqual(result["navigation"]["state"], "ready")
        self.assertEqual(result["command"], zero_axes(0.35))
        self.depth.assert_not_called()

    def test_grounding_completion_requires_another_fresh_frame_before_motion(self):
        self.acquired()
        result = self.step(2)
        self.assertEqual(result["navigation"]["state"], "approaching")
        self.assertEqual(result["command"]["forward"], 0.5)
        self.depth.assert_called_once()


if __name__ == "__main__":
    unittest.main()
