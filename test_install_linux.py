"""Deployment tests exercise plans/guards without root, package downloads, or service changes."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
import install_linux as installer


class RecordingRunner(installer.Runner):
    def __init__(self):
        super().__init__(dry_run=True)
        self.commands = []
        self.files = {}

    def run(self, *command, env=None):
        self.commands.append(tuple(map(str, command)))

    def write(self, path, text, mode=0o644):
        self.files[str(path)] = (text, mode)


class InstallerTests(unittest.TestCase):
    def test_navigation_plan_copies_configuration_preloads_small_depth_and_preserves_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "navigation.json"
            config.write_text('{"image_width":1920}')  # Full validation is delegated to the installed venv.
            args = installer.parse_args(["--torch-profile", "jetson-cu126", "--navigation-config", str(config), "--planner", "single-pass"])
            runner = RecordingRunner()
            with contextlib.redirect_stdout(io.StringIO()):
                installer.install(args, runner)
        self.assertEqual(runner.files["/etc/drone-vla/navigation.json"][0], '{"image_width":1920}')
        service = runner.files["/etc/systemd/system/drone-vla.service"][0]
        self.assertIn("--navigation-config /etc/drone-vla/navigation.json", service)
        self.assertIn("--planner single-pass", service)
        self.assertTrue(any("--navigation" in command and "--preload" in command for command in runner.commands))
        self.assertTrue(any("NavigationConfig.model_validate_json" in " ".join(command) for command in runner.commands))
        self.assertIn("navigate)", runner.files["/usr/local/bin/drone-vla"][0])
        self.assertNotIn("ExecStart", runner.files["/etc/systemd/system/dji-lte.service.d/drone-vla.conf"][0])

    def test_vla_port_is_consistent_and_configurable(self):
        for requested, port in (([], 8766), (["--vla-port", "8877"], 8877)):
            runner = RecordingRunner()
            args = installer.parse_args(["--torch-profile", "cpu", *requested])
            with contextlib.redirect_stdout(io.StringIO()):
                installer.install(args, runner)
            service = runner.files["/etc/systemd/system/drone-vla.service"][0]
            dropin = runner.files["/etc/systemd/system/dji-lte.service.d/drone-vla.conf"][0]
            self.assertIn(f"--port {port} --preload", service)
            self.assertIn(f"http://127.0.0.1:{port}/health", service)
            self.assertIn(f"DJI_VLA_URL=http://127.0.0.1:{port}/plan", dropin)
            self.assertNotIn(":8765", service + dropin)
        for port in ("0", "65536", "-1"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                installer.parse_args(["--vla-port", port])

    def test_removes_only_legacy_cuda13_packages_from_venv(self):
        runner = RecordingRunner()
        runner.dry_run = False
        python = Path("/opt/drone-vla/.venv/bin/python")
        packages = '[{"name":"nvidia-cusparselt-cu13"},{"name":"nvidia_cuda_runtime_cu13"},{"name":"nvidia-cublas-cu12"},{"name":"torch"}]'
        with patch.object(installer.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout=packages)) as run:
            installer.remove_legacy_cuda13_packages(python, runner)
        self.assertEqual(run.call_args.args[0], [str(python), "-m", "pip", "list", "--format=json"])
        self.assertEqual(runner.commands, [(str(python), "-m", "pip", "uninstall", "-y", "nvidia-cusparselt-cu13", "nvidia_cuda_runtime_cu13")])

    def test_cuda13_cleanup_is_noop_for_clean_environment(self):
        runner = RecordingRunner()
        runner.dry_run = False
        with patch.object(installer.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout='[{"name":"torch"}]')):
            installer.remove_legacy_cuda13_packages(Path("/opt/drone-vla/.venv/bin/python"), runner)
        self.assertEqual(runner.commands, [])

    def test_service_accepts_direct_binary_and_paladin_wrapper(self):
        binary = Path("/usr/local/bin/dji_lte")
        self.assertTrue(installer.service_launches_binary(f"{{ path={binary} ; argv[]={binary} ; }}", binary))
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory) / "paladin.sh"
            start = f"{{ path={wrapper} ; argv[]={wrapper} ; }}"
            wrapper.write_text('#!/bin/bash\nARGS=""\n    /usr/local/bin/dji_lte $ARGS > /var/log/dji-lte/run.log\n')
            self.assertTrue(installer.service_launches_binary(start, binary))
            wrapper.write_text('#!/bin/bash\n# /usr/local/bin/dji_lte\necho /usr/local/bin/dji_lte\n/usr/local/bin/dji_lte-other\n')
            self.assertFalse(installer.service_launches_binary(start, binary))
            wrapper.write_text('#!/bin/bash\nexec /usr/local/bin/dji_lte --restart\n')
            self.assertTrue(installer.service_launches_binary(start, binary))
        self.assertFalse(installer.service_launches_binary(start, binary))

    def test_jetson_profile_matches_cuda126(self):
        release = "# R36 (release), REVISION: 4.3, BOARD: generic"
        self.assertEqual(installer.detect_profile("aarch64", release), "jetson-cu126")
        self.assertEqual(installer.detect_profile("aarch64", release.replace("4.3", "4.7")), "jetson-cu126")
        with self.assertRaises(ValueError):
            installer.detect_profile("aarch64", release.replace("R36", "R35"))
        with self.assertRaises(ValueError):
            installer.detect_profile("x86_64", "", "jetson-cu126")

    def test_generic_host_uses_cpu(self):
        self.assertEqual(installer.detect_profile("x86_64", ""), "cpu")
        self.assertEqual(installer.detect_profile("aarch64", ""), "cpu")
        self.assertEqual(installer.detect_profile("x86_64", "", "cuda126"), "cuda126")

    def test_bridge_binary_detection_handles_chunk_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "dji_lte"
            binary.write_bytes(b"x" * (1024 * 1024 - 5) + b"DJI_VLA_SOCKET" + b"\0DJI_VLA_URL")
            self.assertTrue(installer.has_vla_bridge(binary))
            binary.write_bytes(b"DJI_VLA_SOCKET")
            self.assertFalse(installer.has_vla_bridge(binary))

    def test_continuous_bridge_detection_requires_new_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "dji_lte"
            binary.write_bytes(b"DJI_VLA_SOCKET\0DJI_VLA_URL")
            self.assertTrue(installer.has_vla_bridge(binary))
            self.assertFalse(installer.has_vla_bridge(binary, continuous=True))
            # Include a split protocol marker to exercise the bounded scanner.
            binary.write_bytes(b"DJI_VLA_SOCKET\0DJI_VLA_URL\0".ljust(1024 * 1024 - 8, b"x")
                               + b"DJI_VLA_CONTINUOUS_V2")
            self.assertTrue(installer.has_vla_bridge(binary, continuous=True))

    def test_navigation_rejects_one_shot_bridge_before_machine_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "one-shot-dji"
            header = bytearray(20)
            header[:6] = b"\x7fELF\x02\x01"
            header[18:20] = (62).to_bytes(2, "little")
            binary.write_bytes(header + b"DJI_VLA_SOCKET\0DJI_VLA_URL")
            binary.chmod(0o755)
            config = Path(directory) / "navigation.json"
            config.write_text('{"image_width":1920}')
            args = installer.parse_args(["--torch-profile", "cpu", "--dji-binary", str(binary),
                                         "--navigation-config", str(config)])
            runner = RecordingRunner()
            runner.dry_run = False
            with patch.object(installer.platform, "system", return_value="Linux"), \
                 patch.object(installer.shutil, "which", return_value="/usr/bin/apt-get"), \
                 patch.object(installer.Path, "is_dir", return_value=True), \
                 patch.object(installer.os, "geteuid", return_value=0), \
                 patch.object(installer.sys, "version_info", (3, 10, 0)), \
                 patch.object(installer.subprocess, "run", side_effect=AssertionError("No services or packages before binary guard")), \
                 self.assertRaisesRegex(RuntimeError, "lacks DJI_VLA_CONTINUOUS_V2"):
                installer.install(args, runner)
            self.assertEqual(runner.commands, [])
            self.assertEqual(runner.files, {})

    def test_vla_only_navigation_does_not_require_local_dji_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "navigation.json"
            config.write_text('{"image_width":1920}')
            args = installer.parse_args(["--torch-profile", "cpu", "--vla-only", "--navigation-config", str(config)])
            runner = RecordingRunner()
            with patch.object(installer, "has_vla_bridge", side_effect=AssertionError("No DJI binary on perception host")), \
                 contextlib.redirect_stdout(io.StringIO()):
                installer.install(args, runner)
            self.assertIn("/etc/drone-vla/navigation.json", runner.files)
            self.assertFalse(any("dji-lte.service" in path for path in runner.files))

    def test_rejects_binary_built_for_another_machine(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "dji_lte"
            header = bytearray(20)
            header[:6] = b"\x7fELF\x02\x01"
            header[18:20] = (183).to_bytes(2, "little")
            binary.write_bytes(header)
            self.assertTrue(installer.compatible_binary(binary, "aarch64"))
            self.assertFalse(installer.compatible_binary(binary, "x86_64"))
            binary.write_bytes(b"#!/bin/sh\necho test")
            self.assertFalse(installer.compatible_binary(binary, "aarch64"))

    def test_full_plan_preserves_dji_service_settings(self):
        runner = RecordingRunner()
        args = installer.parse_args(["--torch-profile", "jetson-cu126"])
        with patch.object(installer.platform, "machine", return_value="aarch64"), \
             patch.object(installer.shutil, "which", return_value=None), contextlib.redirect_stdout(io.StringIO()):
            installer.install(args, runner)
        commands = runner.commands
        self.assertIn(("apt-get", "install", "-y", "--no-remove", "libcudnn9-cuda-12"), commands)
        self.assertLess(commands.index(("ldconfig",)), next(i for i, c in enumerate(commands) if "--index-url" in c))
        self.assertIn(("python3", "-m", "venv", "--copies", "/opt/drone-vla/.venv"), commands)
        wheels = [c for c in commands if "--index-url" in c]
        self.assertEqual(len(wheels), 1)
        self.assertIn(installer.TORCH_INDEXES["jetson-cu126"], wheels[0])
        self.assertIn("torch==2.8.0", wheels[0])
        self.assertIn("torchvision==0.23.0", wheels[0])
        dropin = runner.files["/etc/systemd/system/dji-lte.service.d/drone-vla.conf"][0]
        self.assertNotIn("ExecStart", dropin)
        self.assertIn("RuntimeDirectoryMode=0700", dropin)
        self.assertIn("Requires=drone-vla.service", dropin)
        service = runner.files["/etc/systemd/system/drone-vla.service"][0]
        self.assertIn("User=drone-vla", service)
        self.assertIn("--preload", service)
        self.assertIn("ExecStartPost=", service)
        self.assertIn("VLA_DEPTH_DEVICE=cuda:0", service)
        self.assertNotIn(("systemctl", "start", "dji-lte.service"), commands)
        self.assertIn(("systemctl", "enable", "dji-lte.service"), commands)

    def test_vla_only_cpu_plan_has_no_dji_deployment(self):
        runner = RecordingRunner()
        args = installer.parse_args(["--torch-profile", "cpu", "--vla-only"])
        with contextlib.redirect_stdout(io.StringIO()):
            installer.install(args, runner)
        self.assertFalse(any("dji-lte.service" in path for path in runner.files))
        self.assertFalse(any("libcudnn9-cuda-12" in command for command in runner.commands))
        self.assertIn("VLA_DEPTH_DEVICE=cpu", runner.files["/etc/systemd/system/drone-vla.service"][0])

    def test_explicit_start_and_custom_binary(self):
        runner = RecordingRunner()
        args = installer.parse_args(["--torch-profile", "cpu", "--dji-binary", "/tmp/new-dji", "--start-dji"])
        with contextlib.redirect_stdout(io.StringIO()):
            installer.install(args, runner)
        self.assertIn(("install", "-m", "0755", "/tmp/new-dji", "/usr/local/bin/dji_lte"), runner.commands)
        self.assertIn(("systemctl", "start", "dji-lte.service"), runner.commands)

    def test_generated_launcher_is_valid_shell(self):
        for vla_only in (False, True):
            text = installer.launcher_text(Path("/opt/drone-vla"), "dji-lte.service", vla_only)
            result = subprocess.run(["bash", "-n"], input=text, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"$@"', text)

    def test_rejects_unsafe_configuration(self):
        for arguments in (["--install-dir", "/"], ["--install-dir", "/opt/a b"],
                          ["--dji-service", "a.service\nExecStart=/tmp/evil"], ["--vla-only", "--start-dji"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                installer.parse_args(arguments)

    def test_dry_run_performs_no_subprocess_or_writes(self):
        args = installer.parse_args(["--torch-profile", "cpu", "--dry-run"])
        with patch.object(installer.subprocess, "run", side_effect=AssertionError("Unexpected real command")), \
             patch.object(installer.Path, "write_text", side_effect=AssertionError("Unexpected file write")), \
             contextlib.redirect_stdout(io.StringIO()):
            installer.install(args, installer.Runner(dry_run=True))

    def test_old_binary_is_rejected_before_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "old-dji"
            binary.write_text("old executable")
            binary.chmod(0o755)
            args = installer.parse_args(["--torch-profile", "cpu", "--dji-binary", str(binary)])
            runner = RecordingRunner()
            runner.dry_run = False
            with patch.object(installer.platform, "system", return_value="Linux"), \
                 patch.object(installer.shutil, "which", return_value="/usr/bin/apt-get"), \
                 patch.object(installer.Path, "is_dir", return_value=True), \
                 patch.object(installer.os, "geteuid", return_value=0), \
                 patch.object(installer.sys, "version_info", (3, 10, 0)), \
                 self.assertRaisesRegex(RuntimeError, "prebuilt executable"):
                installer.install(args, runner)
            self.assertEqual(runner.commands, [])

    def test_running_dji_is_rejected_before_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "new-dji"
            header = bytearray(20)
            header[:6] = b"\x7fELF\x02\x01"
            header[18:20] = (62).to_bytes(2, "little")
            binary.write_bytes(header + b"DJI_VLA_SOCKET\0DJI_VLA_URL")
            binary.chmod(0o755)
            args = installer.parse_args(["--torch-profile", "cpu", "--dji-binary", str(binary)])
            runner = RecordingRunner()
            runner.dry_run = False
            with patch.object(installer.platform, "system", return_value="Linux"), \
                 patch.object(installer.platform, "machine", return_value="x86_64"), \
                 patch.object(installer.shutil, "which", return_value="/usr/bin/apt-get"), \
                 patch.object(installer.Path, "is_dir", return_value=True), \
                 patch.object(installer.os, "geteuid", return_value=0), \
                 patch.object(installer.sys, "version_info", (3, 10, 0)), \
                 patch.object(installer.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)), \
                 self.assertRaisesRegex(RuntimeError, "Stop dji-lte.service"):
                installer.install(args, runner)
            self.assertEqual(runner.commands, [])

    def test_health_endpoint(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from vla_service import VlaHandler
        handler = SimpleNamespace(path="/health", reply=Mock(), server=SimpleNamespace(commands_only=False))
        VlaHandler.do_GET(handler)
        handler.reply.assert_called_once_with(200, {"ok": True, "mode": "vision"})

    def test_cpu_check_rejects_nonworking_cuda_build(self):
        import prepare_runtime
        from types import SimpleNamespace
        fake_torch = SimpleNamespace(__version__="2.8.0+cu130", version=SimpleNamespace(cuda="13.0"))
        fake_vision = SimpleNamespace(__version__="0.23.0")
        with patch.dict(sys.modules, {"torch": fake_torch, "torchvision": fake_vision}), self.assertRaises(RuntimeError):
            prepare_runtime.check_torch("cpu")

    @unittest.skipUnless(importlib.util.find_spec("pydantic") and importlib.util.find_spec("ollama") and importlib.util.find_spec("PIL"), "Planner dependencies unavailable")
    def test_depth_pipeline_uses_selected_device(self):
        import vision_test
        sentinel = object()
        with patch.object(vision_test, "_DEPTH_PIPELINE", None), patch.object(vision_test, "pipeline", return_value=sentinel) as pipeline, \
             patch.dict(os.environ, {"VLA_DEPTH_DEVICE": "cuda:0"}):
            self.assertIs(vision_test._get_depth_pipeline(), sentinel)
            self.assertEqual(pipeline.call_args.kwargs["device"], "cuda:0")


if __name__ == "__main__":
    unittest.main()
