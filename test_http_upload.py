"""Exercise the real full-HD wire size, including Expect: 100-continue.

The original two-pixel fixture never crossed libcurl's large-upload threshold.
These tests use decoded RGB and zero-motion navigation, without models or SDKs.
"""
from contextlib import contextmanager
import http.client
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import threading
import time
import unittest

from navigation import encode_message, zero_axes
from vla_service import VlaHandler, VlaServer


WIDTH, HEIGHT = 1920, 1080
PIXEL = (17, 34, 51)
RGB = bytes(PIXEL) * (WIDTH * HEIGHT)


class GroundingFixture:
    def __init__(self):
        self.received = []

    def handle(self, metadata, image, received_at=None):
        self.received.append((dict(metadata), image.size,
                              [image.getpixel(point) for point in
                               ((0, 0), (WIDTH-1, 0), (0, HEIGHT-1), (WIDTH-1, HEIGHT-1))]))
        message = json.loads(metadata["text"][len("@nav1:"):])
        return dict(ok=True, request_id=metadata["request_id"], command=zero_axes(0.35),
                    navigation=dict(goal_id=metadata["goal_id"], sequence=message["sequence"],
                                    state="grounding", reason="Grounding pending", timings_ms={}))


@contextmanager
def server_fixture(protocol=None):
    navigation, uploads = GroundingFixture(), []

    class Handler(VlaHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            uploads.append(dict(self.headers))
            super().do_POST()

    if protocol is not None:
        Handler.protocol_version = protocol
    with VlaServer(("127.0.0.1", 0), navigation=navigation) as server:
        server.RequestHandlerClass = Handler
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server, navigation, uploads
        finally:
            server.shutdown()
            thread.join(timeout=2)


def body(age_ns=0):
    metadata = dict(bridge_protocol=2, session="upload-session", goal_id="upload-goal",
                    frame_id=1, frame_monotonic_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC)-age_ns,
                    request_id="upload-request", default_duration=0.35,
                    width=WIDTH, height=HEIGHT, format="RGB",
                    text=encode_message(op="start", goal_id="upload-goal", owner="upload-session",
                                        sequence=0, text="Approach the tree"))
    header = json.dumps(metadata).encode()
    return struct.pack("!I", len(header))+header+RGB


def response(reader):
    status = reader.readline().decode().strip()
    headers = {}
    while True:
        line = reader.readline()
        if line == b"\r\n":
            break
        if not line:
            raise AssertionError("HTTP connection closed before response headers")
        key, value = line.decode().split(":", 1)
        headers[key.lower()] = value.strip()
    return status, headers


class FullHdUploadTests(unittest.TestCase):
    def assert_received_frame(self, navigation):
        self.assertEqual(len(navigation.received), 1)
        metadata, size, corners = navigation.received[0]
        self.assertEqual((metadata["width"], metadata["height"]), (WIDTH, HEIGHT))
        self.assertEqual(size, (WIDTH, HEIGHT))
        self.assertEqual(corners, [PIXEL]*4)

    def test_expect_continue_is_acknowledged_before_camera_body_is_sent(self):
        with server_fixture() as (server, navigation, _):
            payload = body()
            with socket.create_connection(server.server_address, timeout=0.5) as client:
                with client.makefile("rb") as reader:
                    client.sendall((f"POST /plan HTTP/1.1\r\nHost: localhost\r\n"
                                    f"Content-Type: application/octet-stream\r\n"
                                    f"Content-Length: {len(payload)}\r\n"
                                    "Expect: 100-continue\r\n\r\n").encode())
                    # The client has not sent a single image byte yet. Waiting
                    # for body arrival before acknowledging reproduces the bug.
                    interim, _ = response(reader)
                    self.assertEqual(interim, "HTTP/1.1 100 Continue")
                    client.sendall(payload)
                    status, headers = response(reader)
                    self.assertEqual(status, "HTTP/1.1 200 OK")
                    result = json.loads(reader.read(int(headers["content-length"])))
                    self.assertEqual(result["navigation"]["state"], "grounding")
                    self.assertEqual(result["command"], zero_axes(0.35))
                    self.assertLessEqual(result["navigation"]["timings_ms"]["decoded_frame_age_ms"], 750)
                    self.assertEqual(headers.get("connection"), "close")
            self.assert_received_frame(navigation)

    def test_updated_native_upload_also_works_with_an_http10_server(self):
        binary = Path(os.environ.get("VLA_TEST_BINARY", "/tmp/dji-vla-ready-build/test-vla-bridge"))
        if not binary.is_file():
            self.skipTest("Build the native full-resolution HTTP driver first")
        with server_fixture(protocol="HTTP/1.0") as (server, navigation, uploads):
            result = subprocess.run([str(binary), "--large-frame", f"http://127.0.0.1:{server.server_port}/plan"],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            status = json.loads(result.stdout)
            self.assertEqual(status["navigation"]["state"], "grounding")
            self.assertFalse(any(key.lower() == "expect" for key in uploads[0]))
            self.assert_received_frame(navigation)

    def test_genuinely_old_full_hd_frame_remains_rejected(self):
        with server_fixture() as (server, navigation, _):
            connection = http.client.HTTPConnection(*server.server_address, timeout=2)
            try:
                connection.request("POST", "/plan", body(age_ns=800_000_000),
                                   {"Content-Type": "application/octet-stream"})
                reply = connection.getresponse()
                self.assertEqual(reply.status, 400)
                self.assertIn("Native camera frame is stale", json.loads(reply.read())["error"])
                self.assertEqual(navigation.received, [])
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
