"""Run the compiled C++ bridge against the Python service without model/SDK calls."""
import os
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
import unittest

from test_dji_control import motion
from vla_service import VlaServer


class HttpIntegrationTest(unittest.TestCase):
    def test_cpp_camera_request_and_python_command_response(self):
        from PIL import Image
        binary = Path(os.environ.get("VLA_TEST_BINARY", "/tmp/test-vla-bridge"))
        if not binary.is_file() and "VLA_TEST_BINARY" not in os.environ:
            binary = Path("/tmp/dji-vla-build/test-vla-bridge")
        if not binary.is_file():
            self.skipTest("Compile dji-lte/test/test_vla_bridge.cpp first")
        received = []

        def planner(text, path):
            with Image.open(path) as frame:
                if frame.size != (2, 1) or list(frame.getdata()) != [(255, 0, 0), (0, 255, 0)]:
                    raise RuntimeError("C++ camera pixels changed in HTTP transport")
            received.append(text)
            command = motion(duration_seconds=0.02)
            reason = "Path blocked" if text == "blocked" else ""
            if text != "blocked":
                command = motion(longitudinal_direction="forward", longitudinal_speed_mps=3 if text == "overlimit" else 1,
                                 duration_seconds=0.02)
            return SimpleNamespace(command=command, override_reason=reason)

        with VlaServer(("127.0.0.1", 0), planner) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = subprocess.run([str(binary), f"http://127.0.0.1:{server.server_port}/plan"],
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(received, ["forward", "blocked", "overlimit"])
            finally:
                server.shutdown()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
