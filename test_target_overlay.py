"""Correlated display snapshots, expiry, and nonblocking HTTP overlay reads."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import http.client
import json
import struct
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from bridge_protocol import correlate_response
from navigation import encode_message, zero_axes
from target_overlay import CLEAR, OverlayState, monotonic_ns, snapshot_from_response
from test_bridge_protocol import NOW_NS, native_metadata, native_response
from test_dji_control import motion
from vla_service import VlaServer


def target(metadata, sequence, width=1920, height=1080):
    return dict(visibility="visible", fresh=True, observed_sequence=sequence,
                frame_id=metadata["frame_id"], frame_monotonic_ns=metadata["frame_monotonic_ns"],
                image_size_px=[width, height], box_normalized=[0.25, 0.2, 0.75, 0.8],
                tracking_confidence=0.935, grounding_confidence=0.95,
                distance_m=8.8, bearing_deg=4.55)


def valid_response():
    metadata = native_metadata()
    value = native_response(metadata)
    value["navigation"]["target"] = target(metadata, 1)
    return correlate_response(value, metadata, now_ns=NOW_NS)


class OverlaySnapshotTests(unittest.TestCase):
    def test_validated_snapshot_has_only_video_fields_and_normalized_box(self):
        response = valid_response()
        value = snapshot_from_response(response, NOW_NS)
        self.assertEqual(value["protocol"], 1)
        self.assertEqual(value["action"], "target")
        self.assertEqual(value["source"], "wide")
        self.assertEqual(value["box_normalized"], [0.25, 0.2, 0.75, 0.8])
        self.assertEqual(value["image_size_px"], [1920, 1080])
        self.assertEqual(value["confidence"], 0.935)
        self.assertEqual(value["distance_m"], 8.8)
        self.assertEqual(value["bearing_deg"], 4.55)
        for key in ("session", "goal_id", "request_id", "frame_id", "frame_monotonic_ns"):
            self.assertEqual(value[key], response[key])
        self.assertNotIn("command", value)
        self.assertNotIn("instruction", value)

    def test_hold_states_and_unconfirmed_or_pending_observations_clear_previous_box(self):
        for state in ("stale", "lost", "error", "cancelled", "grounding", "acquiring", "unknown"):
            with self.subTest(state=state):
                value = valid_response()
                value["navigation"]["state"] = state
                self.assertEqual(snapshot_from_response(value, NOW_NS), CLEAR)
        for visibility in ("unconfirmed", "out_of_view", "pending", None):
            with self.subTest(visibility=visibility):
                value = valid_response()
                value["navigation"]["target"]["visibility"] = visibility
                self.assertEqual(snapshot_from_response(value, NOW_NS), CLEAR)

    def test_fresh_visible_held_target_can_still_be_inspected_in_video(self):
        for state in ("ready", "arrived", "blocked"):
            with self.subTest(state=state):
                value = valid_response()
                value["navigation"]["state"] = state
                value["command"] = zero_axes(0.35)
                self.assertEqual(snapshot_from_response(value, NOW_NS)["action"], "target")

    def test_observation_and_validated_frame_context_must_match(self):
        for key, bad in (("observed_sequence", 9), ("frame_id", 99), ("frame_monotonic_ns", 123)):
            with self.subTest(key=key):
                value = valid_response()
                value["navigation"]["target"][key] = bad
                self.assertEqual(snapshot_from_response(value, NOW_NS), CLEAR)
        value = valid_response()
        value["navigation"]["goal_id"] = "old-goal"
        self.assertEqual(snapshot_from_response(value, NOW_NS), CLEAR)

    def test_bool_and_float_frame_identity_aliases_are_rejected(self):
        for key in ("observed_sequence", "frame_id", "frame_monotonic_ns"):
            for convert in (bool, float):
                with self.subTest(key=key, convert=convert):
                    value = valid_response()
                    value["navigation"]["target"][key] = convert(value["navigation"]["target"][key])
                    self.assertEqual(snapshot_from_response(value, NOW_NS), CLEAR)
        value = valid_response()
        value["bridge_protocol"] = 2.0
        self.assertEqual(snapshot_from_response(value, NOW_NS), CLEAR)

    def test_missing_identity_dimensions_and_bad_geometry_clear(self):
        for key, bad in (("session", ""), ("goal_id", ""), ("request_id", ""),
                         ("session", 42), ("bridge_protocol", None), ("ok", False)):
            with self.subTest(field=key, bad=bad):
                value = valid_response()
                value[key] = bad
                self.assertEqual(snapshot_from_response(value, NOW_NS), CLEAR)
        for box in ([0.7, 0.2, 0.3, 0.8], [-0.1, 0.2, 0.3, 0.8],
                    [0.2, 0.2, 1.1, 0.8], [0.2, 0.2, float("nan"), 0.8],
                    [False, 0.2, 0.3, 0.8], [0.2, 0.2, 0.8], None):
            with self.subTest(box=box):
                value = valid_response()
                value["navigation"]["target"]["box_normalized"] = box
                self.assertEqual(snapshot_from_response(value, NOW_NS), CLEAR)
        for size in ([0, 1080], [1920, 2161], [4097, 1080], [1920.0, 1080], [True, 1080], None):
            with self.subTest(size=size):
                value = valid_response()
                value["navigation"]["target"]["image_size_px"] = size
                self.assertEqual(snapshot_from_response(value, NOW_NS), CLEAR)

    def test_nonfinite_or_invalid_display_estimates_clear(self):
        for key, values in (("tracking_confidence", (float("nan"), 1.1, -0.1, True)),
                            ("distance_m", (float("inf"), -1, 0, True)),
                            ("bearing_deg", (float("nan"), -91, 91, True))):
            for bad in values:
                with self.subTest(field=key, bad=bad):
                    value = valid_response()
                    value["navigation"]["target"][key] = bad
                    self.assertEqual(snapshot_from_response(value, NOW_NS), CLEAR)

    def test_source_age_and_future_tolerance_boundaries(self):
        response = valid_response()
        source_ns = response["frame_monotonic_ns"]
        for offset in (-20_000_000, 750_000_000):
            with self.subTest(allowed_age_ns=offset):
                self.assertEqual(snapshot_from_response(response, source_ns+offset)["action"], "target")
        for offset in (-20_000_001, 750_000_001):
            with self.subTest(rejected_age_ns=offset):
                self.assertEqual(snapshot_from_response(response, source_ns+offset), CLEAR)

    def test_reads_do_not_refresh_ttl_and_return_independent_snapshot(self):
        now = [NOW_NS]
        cache = OverlayState(clock_ns=lambda: now[0])
        response = valid_response()
        cache.update(response)
        response["navigation"]["target"]["box_normalized"][0] = 0
        first = cache.current()
        self.assertEqual(first["box_normalized"][0], 0.25)
        first["box_normalized"][0] = 0.1
        self.assertEqual(cache.current()["box_normalized"][0], 0.25)
        now[0] = response["frame_monotonic_ns"]+750_000_000
        self.assertEqual(cache.current()["action"], "target")
        now[0] += 1
        self.assertEqual(cache.current(), CLEAR)
        now[0] -= 100_000_000
        self.assertEqual(cache.current(), CLEAR)

    def test_malformed_update_never_retains_or_raises_from_previous_box(self):
        cache = OverlayState(clock_ns=lambda: NOW_NS)
        for malformed in (None, [], {}, dict(navigation=None), dict(navigation=[]),
                          dict(ok=True, bridge_protocol=2, navigation=dict(target=None))):
            with self.subTest(malformed=malformed):
                cache.update(valid_response())
                self.assertEqual(cache.current()["action"], "target")
                cache.update(malformed)
                self.assertEqual(cache.current(), CLEAR)


class NavigationFixture:
    def __init__(self):
        self.block = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled = 0

    def handle(self, metadata, image, received_at=None):
        if self.block:
            self.entered.set()
            if not self.release.wait(2):
                raise RuntimeError("Test fixture did not release inference")
        message = json.loads(metadata["text"][len("@nav1:"):])
        info = dict(goal_id=message["goal_id"], sequence=message["sequence"],
                    state="approaching", reason="Target visible", timings_ms={})
        if metadata.get("bridge_protocol") == 2:
            info["target"] = target(metadata, message["sequence"], image.width, image.height)
        return dict(ok=True, request_id=metadata["request_id"],
                    command=dict(forward=0.5, right=0.0, up=0.0, yaw=0.0, duration_seconds=0.35),
                    navigation=info)

    def cancel(self):
        self.cancelled += 1


@contextmanager
def http_overlay():
    navigation = NavigationFixture()
    planner = lambda *_: SimpleNamespace(command=motion(duration_seconds=0.2), override_reason="")
    with patch("vla_service.VlaHandler.log_message", lambda *args: None), \
         VlaServer(("127.0.0.1", 0), planner=planner, navigation=navigation) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server, navigation
        finally:
            navigation.release.set()
            server.shutdown()
            thread.join(timeout=2)


def camera_body(sequence=1, legacy=False, text=None):
    image = Image.new("RGB", (32, 18), "blue")
    metadata = dict(request_id="video-request", width=32, height=18, format="RGB", default_duration=0.35)
    if text is None:
        text = encode_message(op="step", owner="manual1", goal_id="goal1", sequence=sequence)
    metadata["text"] = text
    if not legacy:
        metadata.update(bridge_protocol=2, session="manual1", goal_id="goal1", frame_id=sequence+1,
                        frame_monotonic_ns=monotonic_ns())
    header = json.dumps(metadata).encode()
    return struct.pack("!I", len(header))+header+image.tobytes(), metadata


def request(server, method, path, payload=None):
    connection = http.client.HTTPConnection(*server.server_address, timeout=1)
    try:
        connection.request(method, path, payload, {"Content-Type": "application/octet-stream"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


class OverlayHttpTests(unittest.TestCase):
    def test_validated_plan_publishes_target_to_overlay_endpoint(self):
        with http_overlay() as (server, _):
            self.assertEqual(request(server, "GET", "/overlay"), (200, CLEAR))
            payload, metadata = camera_body()
            status, response = request(server, "POST", "/plan", payload)
            self.assertEqual(status, 200)
            status, overlay = request(server, "GET", "/overlay")
            self.assertEqual(status, 200)
            self.assertEqual(overlay["action"], "target")
            self.assertEqual(overlay["frame_id"], metadata["frame_id"])
            self.assertEqual(overlay["box_normalized"], response["navigation"]["target"]["box_normalized"])
            self.assertEqual(overlay["image_size_px"], [32, 18])

    def test_get_remains_responsive_while_another_frame_is_in_inference(self):
        with http_overlay() as (server, navigation):
            request(server, "POST", "/plan", camera_body()[0])
            navigation.block = True
            with ThreadPoolExecutor(max_workers=1) as worker:
                pending = worker.submit(request, server, "POST", "/plan", camera_body(2)[0])
                try:
                    self.assertTrue(navigation.entered.wait(1))
                    self.assertTrue(server.inference_lock.locked())
                    self.assertFalse(pending.done())
                    status, overlay = request(server, "GET", "/overlay")
                    self.assertEqual(status, 200)
                    self.assertEqual(overlay["action"], "target")
                    self.assertEqual(overlay["sequence"], 1)
                    self.assertFalse(pending.done())
                finally:
                    navigation.release.set()
                self.assertEqual(pending.result(timeout=1)[0], 200)
            self.assertEqual(request(server, "GET", "/overlay")[1]["sequence"], 2)

    def test_malformed_post_clears_previous_overlay_immediately(self):
        with http_overlay() as (server, _):
            request(server, "POST", "/plan", camera_body()[0])
            self.assertEqual(request(server, "GET", "/overlay")[1]["action"], "target")
            status, _ = request(server, "POST", "/plan", b"bad")
            self.assertEqual(status, 400)
            self.assertEqual(request(server, "GET", "/overlay"), (200, CLEAR))

    def test_hover_cancels_goal_and_clears_display(self):
        with http_overlay() as (server, navigation):
            request(server, "POST", "/plan", camera_body()[0])
            status, _ = request(server, "POST", "/plan", camera_body(legacy=True, text="Hover")[0])
            self.assertEqual(status, 200)
            self.assertEqual(navigation.cancelled, 1)
            self.assertEqual(request(server, "GET", "/overlay"), (200, CLEAR))

    def test_uncorrelated_legacy_navigation_cannot_publish_box(self):
        with http_overlay() as (server, _):
            request(server, "POST", "/plan", camera_body()[0])
            status, _ = request(server, "POST", "/plan", camera_body(legacy=True)[0])
            self.assertEqual(status, 200)
            self.assertEqual(request(server, "GET", "/overlay"), (200, CLEAR))

    def test_expired_source_frame_clears_on_get_without_any_new_plan(self):
        with http_overlay() as (server, _):
            payload, metadata = camera_body()
            request(server, "POST", "/plan", payload)
            self.assertEqual(request(server, "GET", "/overlay")[1]["action"], "target")
            server.overlay_state.clock_ns = lambda: metadata["frame_monotonic_ns"]+750_000_001
            self.assertEqual(request(server, "GET", "/overlay"), (200, CLEAR))


if __name__ == "__main__":
    unittest.main()
