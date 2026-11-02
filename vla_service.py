"""Receive dji-lte's RGB camera frame + instruction and return the applied control command."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
from pathlib import Path
import struct
import tempfile
import threading
import time

from dji_control import joystick_command
from bridge_protocol import correlate_response, validate_context

MAX_BODY_BYTES = 4 + 8192 + 4096 * 2160 * 3
LOGGER = logging.getLogger("drone_vla.http")


def decode_request(body):
    if len(body) < 4 or len(body) > MAX_BODY_BYTES:
        raise ValueError("Invalid camera request size")
    header_size = struct.unpack("!I", body[:4])[0]
    if not 0 < header_size <= 8192 or 4 + header_size > len(body):
        raise ValueError("Invalid camera request header")
    metadata = json.loads(body[4:4 + header_size])
    width, height = metadata["width"], metadata["height"]
    if type(width) is not int or type(height) is not int or not 0 < width <= 4096 or not 0 < height <= 2160:
        raise ValueError("Invalid camera dimensions")
    if metadata["format"] != "RGB":
        raise ValueError("Camera format must be RGB")
    text = metadata["text"]
    if not isinstance(text, str) or not text.strip() or len(text.encode()) > 1000:
        raise ValueError("Invalid natural-language instruction")
    request_id = metadata["request_id"]
    if not isinstance(request_id, str) or not 0 < len(request_id) <= 256:
        raise ValueError("Invalid request ID")
    duration = metadata["default_duration"]
    if type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= 2:
        raise ValueError("Duration must be in (0, 2] seconds")
    rgb = body[4 + header_size:]
    if len(rgb) != width * height * 3:
        raise ValueError("Camera byte count does not match dimensions")
    validate_context(metadata)
    return metadata, rgb


def plan_request(body, planner=None, commands_only=False, navigation=None):
    received = time.monotonic()
    metadata, rgb = decode_request(body)
    LOGGER.info("request=%s camera=%sx%s received", metadata["request_id"], metadata["width"], metadata["height"])
    if metadata["text"].startswith("@nav1:"):
        if commands_only:
            raise ValueError("Navigation requires visual checks; disable commands-only mode")
        if navigation is None:
            raise ValueError("Navigation is not configured; set --navigation-config with calibrated camera geometry")
        from PIL import Image
        result = navigation.handle(metadata, Image.frombytes("RGB", (metadata["width"], metadata["height"]), rgb), received_at=received)
        return correlate_response(result, metadata) if metadata.get("bridge_protocol") == 2 else result
    if navigation is not None:
        # Ordinary instructions, including Hover, retire a persistent goal.
        navigation.cancel()
    if planner is None:
        from integrated_control import plan_motion
        planner = plan_motion
    if commands_only:
        # The camera payload is still validated, but no image inference is run.
        decision = planner(metadata["text"], None)
    elif getattr(planner, "accepts_frame", False):
        from PIL import Image
        decision = planner(metadata["text"], Image.frombytes("RGB", (metadata["width"], metadata["height"]), rgb))
    else:
        from PIL import Image
        with tempfile.TemporaryDirectory(prefix="vla-camera-") as directory:
            # The planner's image-path API receives only pixels sent by dji-lte.
            image_path = str(Path(directory) / "frame.png")
            Image.frombytes("RGB", (metadata["width"], metadata["height"]), rgb).save(image_path)
            decision = planner(metadata["text"], image_path)
    command = joystick_command(decision.command, metadata["default_duration"])
    command.pop("op")
    override = decision.override_reason or ""
    if commands_only:
        override = "Visual checks bypassed (commands-only mode)" + ("; " + override if override else "")
    return {"ok": True, "request_id": metadata["request_id"], "command": command,
            "override_reason": override, "timings_ms": {"service_ms": round((time.monotonic()-received)*1000, 2)}}


class VlaServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, planner=None, commands_only=False, navigation=None):
        super().__init__(address, VlaHandler)
        self.planner = planner
        self.commands_only = commands_only
        self.navigation = navigation
        self.inference_lock = threading.Lock()


class VlaHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def reply(self, status, payload):
        data = json.dumps(payload, allow_nan=False).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            LOGGER.warning("Client disconnected before response delivery; result discarded")
            self.close_connection = True

    def do_GET(self):
        if self.path == "/health":
            health = {"ok": True, "mode": "commands-only" if self.server.commands_only else "vision"}
            if getattr(self.server, "navigation", None) is not None:
                health["navigation"] = "enabled"
                health["bridge_protocol"] = 2
                health["continuous_navigation"] = True
            self.reply(200, health)
        else:
            self.reply(404, {"ok": False, "error": "Unknown endpoint"})

    def do_POST(self):
        if self.path != "/plan":
            self.reply(404, {"ok": False, "error": "Unknown endpoint"})
            return
        if not self.server.inference_lock.acquire(blocking=False):
            self.reply(503, {"ok": False, "error": "VLA inference is busy"})
            return
        started = time.monotonic()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 4 <= length <= MAX_BODY_BYTES:
                raise ValueError("Invalid Content-Length")
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Truncated camera request")
            self.reply(200, plan_request(body, self.server.planner, self.server.commands_only,
                                         getattr(self.server, "navigation", None)))
        except (ValueError, KeyError, TypeError, UnicodeError) as error:
            LOGGER.warning("Request rejected: %s", error)
            self.reply(400, {"ok": False, "error": str(error)})
        except Exception as error:
            LOGGER.exception("Inference failed")
            self.reply(500, {"ok": False, "error": str(error)})
        finally:
            LOGGER.info("POST /plan elapsed=%.2fs", time.monotonic() - started)
            self.server.inference_lock.release()


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--preload", action="store_true", help="Load depth weights before accepting requests")
    parser.add_argument("--commands-only", action="store_true", default=os.environ.get("VLA_COMMANDS_ONLY") == "1",
                        help="Parse instructions without image/depth assessment; also set VLA_COMMANDS_ONLY=1")
    parser.add_argument("--navigation-config", default=os.environ.get("VLA_NAVIGATION_CONFIG"),
                        help="Enable persistent goal navigation with calibrated camera/controller JSON")
    parser.add_argument("--planner", choices=("legacy", "single-pass"), default="legacy",
                        help="One-shot planner; persistent navigate requests use the separate goal loop")
    args = parser.parse_args()
    navigation = None
    if args.navigation_config and not args.commands_only:
        from navigation import NavigationConfig, NavigationEngine
        navigation = NavigationEngine(NavigationConfig.model_validate_json(Path(args.navigation_config).read_text()))
        if args.preload:
            navigation.preload()
    elif args.preload and not args.commands_only:
        from vision_test import _get_depth_pipeline
        _get_depth_pipeline()
    try:
        # Keep the original constructor invocation compatible with existing integrations.
        with VlaServer((args.host, args.port), commands_only=args.commands_only) as server:
            server.navigation = navigation
            if args.planner == "single-pass" and not args.commands_only:
                from single_pass import plan_frame
                server.planner = plan_frame
            print(f"VLA service listening at http://{args.host}:{args.port}/plan", flush=True)
            print("VLA mode: " + ("commands-only (visual checks bypassed)" if args.commands_only else "vision"), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    finally:
        if navigation is not None:
            navigation.close()


if __name__ == "__main__":
    main()
