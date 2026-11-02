#!/usr/bin/env python3
"""Install drone-vla and connect it to an existing VLA-enabled dji_lte binary."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from wait_http import wait_http

SOURCE = Path(__file__).resolve().parents[1]
TORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda126": "https://download.pytorch.org/whl/cu126",
    "jetson-cu126": "https://pypi.jetson-ai-lab.io/jp6/cu126",
}
CUSPARSELT_NAME = "libcusparse_lt-linux-aarch64-0.7.1.0-archive"
CUSPARSELT_URL = f"https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/{CUSPARSELT_NAME}.tar.xz"


def detect_profile(machine, tegra_release, requested="auto"):
    if requested != "auto":
        if requested == "jetson-cu126" and machine not in ("aarch64", "arm64"):
            raise ValueError("Jetson wheels require ARM64")
        return requested
    if tegra_release:
        match = re.search(r"# R(\d+).*?REVISION:\s*(\d+)\.(\d+)", tegra_release)
        if machine in ("aarch64", "arm64") and match and tuple(map(int, match.groups()))[:2] == (36, 4) and int(match[3]) >= 3:
            return "jetson-cu126"
        raise ValueError("Automatic Jetson setup supports L4T R36.4.3+ (JetPack 6.2.x). Use --torch-profile cpu for another release.")
    if machine not in ("x86_64", "aarch64", "arm64"):
        raise ValueError(f"Unsupported architecture: {machine}")
    return "cpu"


def safe_path(value):
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", value) or ".." in Path(value).parts:
        raise argparse.ArgumentTypeError("Use an absolute path without spaces or '..'")
    return Path(value)


def has_vla_bridge(binary, continuous=False):
    # Scan in bounded chunks; embedded environment names survive normal binary stripping.
    required = {b"DJI_VLA_SOCKET", b"DJI_VLA_URL"}
    if continuous:
        required.add(b"DJI_VLA_CONTINUOUS_V2")
    found = set()
    previous = b""
    with binary.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            combined = previous + chunk
            for marker in required:
                if marker in combined:
                    found.add(marker)
            previous = chunk[-64:]
    return found == required


def compatible_binary(binary, machine):
    with binary.open("rb") as stream:
        header = stream.read(20)
    expected = 183 if machine in ("aarch64", "arm64") else 62 if machine == "x86_64" else None
    return len(header) == 20 and header[:6] == b"\x7fELF\x02\x01" and int.from_bytes(header[18:20], "little") == expected


def service_launches_binary(existing_start, binary):
    """Recognize a direct executable or a shell wrapper without executing it."""
    match = re.search(r"\bpath=([^ ;]+)\s*;", existing_start)
    if not match:
        return False
    executable = Path(match[1])
    if executable == binary:
        return True
    try:
        source = executable.read_text()
    except (OSError, UnicodeError):
        return False
    if not re.match(r"^#!\s*(?:/bin/(?:ba|da|z)?sh|/usr/bin/env\s+(?:ba|da|z)?sh)\s*(?:\n|$)", source):
        return False
    for line in source.splitlines()[1:]:
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue
        if words and words[0] == "exec":
            words = words[1:]
        if words and words[0] == str(binary):
            return True
    return False


def remove_legacy_cuda13_packages(python, runner):
    """Clean CUDA 13 wheels left by an earlier PyTorch install in our venv."""
    if runner.dry_run:
        print("+ remove leftover NVIDIA CUDA 13 packages from the VLA virtual environment")
        return
    installed = subprocess.run([str(python), "-m", "pip", "list", "--format=json"],
                               capture_output=True, text=True, check=True)
    names = [package["name"] for package in json.loads(installed.stdout)]
    legacy = sorted(name for name in names if re.fullmatch(r"nvidia-.+-cu13", re.sub(r"[-_.]+", "-", name.lower())))
    if legacy:
        runner.run(python, "-m", "pip", "uninstall", "-y", *legacy)


class Runner:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run

    def run(self, *command, env=None):
        print("+ " + shlex.join(map(str, command)), flush=True)
        if not self.dry_run:
            subprocess.run(list(map(str, command)), check=True, env=env)

    def write(self, path, text, mode=0o644):
        print(f"+ write {path}", flush=True)
        if not self.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Avoid needless changes to existing units on repeat installs.
            if not path.exists() or path.read_text() != text:
                temporary = path.with_name(path.name + ".install-new")
                temporary.write_text(text)
                temporary.chmod(mode)
                temporary.replace(path)
            else:
                path.chmod(mode)


def service_text(directory, profile, port=8766, navigation=False, planner="legacy"):
    device = "cpu" if profile == "cpu" else "cuda:0"
    extra = (" --navigation-config /etc/drone-vla/navigation.json" if navigation else "")
    extra += " --planner single-pass" if planner == "single-pass" else ""
    return f"""[Unit]
Description=Drone VLA camera inference service
Requires=ollama.service
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
User=drone-vla
Group=drone-vla
WorkingDirectory={directory}
Environment=PYTHONUNBUFFERED=1
Environment=HF_HOME=/var/lib/drone-vla/huggingface
Environment=OLLAMA_HOST=http://127.0.0.1:11434
Environment=VLA_DEPTH_DEVICE={device}
Environment=LD_LIBRARY_PATH={directory}/.cuda/lib:/usr/local/cuda/lib64
ExecStartPre={directory}/.venv/bin/python {directory}/scripts/wait_http.py http://127.0.0.1:11434/api/tags --timeout 60
ExecStart={directory}/.venv/bin/python {directory}/vla_service.py --host 127.0.0.1 --port {port} --preload{extra}
ExecStartPost={directory}/.venv/bin/python {directory}/scripts/wait_http.py http://127.0.0.1:{port}/health --timeout 300
Restart=on-failure
RestartSec=5
TimeoutStartSec=360
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
"""


def dji_dropin_text(port=8766):
    return f"""[Unit]
Requires=drone-vla.service
After=drone-vla.service

[Service]
Environment=DJI_VLA_SOCKET=/run/dji-vla/control.sock
Environment=DJI_VLA_URL=http://127.0.0.1:{port}/plan
RuntimeDirectory=dji-vla
RuntimeDirectoryMode=0700
"""


def launcher_text(directory, dji_service, vla_only):
    units = "ollama.service drone-vla.service" + ("" if vla_only else " " + dji_service)
    start = "drone-vla.service" if vla_only else dji_service
    command = f'exec {directory}/.venv/bin/python {directory}/dji_control.py --socket /run/dji-vla/control.sock "$@"'
    return f"""#!/usr/bin/env bash
set -euo pipefail
case "${{1:-}}" in
  start) exec systemctl start {start} ;;
  stop) exec systemctl stop {start} drone-vla.service ;;
  status) exec systemctl --no-pager status {units} ;;
  logs) shift; exec journalctl -u ollama.service -u drone-vla.service {'' if vla_only else '-u ' + dji_service} -f "$@" ;;
  check) exec env LD_LIBRARY_PATH={directory}/.cuda/lib:/usr/local/cuda/lib64 {directory}/.venv/bin/python {directory}/scripts/prepare_runtime.py --profile "$(cat {directory}/.torch-profile)" ;;
  command) shift; {command} ;;
  navigate) shift; exec {directory}/.venv/bin/python {directory}/navigation_control.py --socket /run/dji-vla/control.sock "$@" ;;
  '') echo 'Usage: drone-vla start|stop|status|logs|check|navigate "destination"|"motion instruction"'; exit 2 ;;
  *) {command} ;;
esac
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", type=safe_path, default=Path("/opt/drone-vla"))
    parser.add_argument("--dji-path", type=safe_path, default=Path("/usr/local/bin/dji_lte"))
    parser.add_argument("--dji-binary", type=safe_path, help="Install this prebuilt VLA-enabled binary at --dji-path")
    parser.add_argument("--dji-service", default="dji-lte.service")
    parser.add_argument("--torch-profile", choices=("auto", *TORCH_INDEXES), default="auto")
    parser.add_argument("--vla-port", type=int, default=8766, help="Local VLA HTTP port (default: 8766; DJI exit-status service uses 8765)")
    parser.add_argument("--vla-only", action="store_true", help="Install perception on a computer without the DJI runtime")
    parser.add_argument("--start-dji", action="store_true", help="Start the DJI service after setup; run on the ground")
    parser.add_argument("--ollama-version", help="Optional Ollama version passed to the official installer")
    parser.add_argument("--dry-run", action="store_true", help="Print deployment commands without changing the machine")
    parser.add_argument("--navigation-config", type=safe_path, help="Install calibrated JSON and enable continuous navigation (requires a DJI_VLA_CONTINUOUS_V2 binary)")
    parser.add_argument("--planner", choices=("legacy", "single-pass"), default="legacy", help="One-shot planner (legacy preserves existing behavior)")
    args = parser.parse_args(argv)
    if not 1 <= args.vla_port <= 65535:
        parser.error("--vla-port must be between 1 and 65535")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_@.-]*\.service", args.dji_service):
        parser.error("Invalid --dji-service name")
    if args.install_dir in (Path("/"), Path("/opt"), Path("/usr")):
        parser.error("Use a dedicated installation directory")
    if args.start_dji and args.vla_only:
        parser.error("--start-dji cannot be combined with --vla-only")
    return args


def install(args, runner):
    dry = runner.dry_run
    navigation_config = None
    if args.navigation_config:
        # Dependencies might not yet exist on a fresh system. Full schema validation
        # happens with the installed venv before any service unit is changed.
        navigation_config = args.navigation_config.read_text()
        if not isinstance(json.loads(navigation_config), dict):
            raise ValueError("Navigation config must be a JSON object")
    if not dry:
        if platform.system() != "Linux" or not shutil.which("apt-get") or not Path("/run/systemd/system").is_dir():
            raise RuntimeError("Supported targets: Ubuntu 22.04/24.04 or Debian 12 with systemd")
        if os.geteuid() != 0:
            raise RuntimeError("Run with sudo bash install.sh")
        if not (3, 10) <= sys.version_info[:2] <= (3, 12):
            raise RuntimeError("Use system Python 3.10–3.12 (Jetson CUDA wheels require Python 3.10)")
    release_path = Path("/etc/nv_tegra_release")
    release = release_path.read_text() if release_path.exists() else ""
    profile = detect_profile(platform.machine(), release, args.torch_profile)
    if profile == "jetson-cu126" and not dry and sys.version_info[:2] != (3, 10):
        raise RuntimeError("Jetson CUDA 12.6 wheels require Python 3.10")
    directory = args.install_dir
    if not dry and directory.is_symlink():
        raise RuntimeError("Installation directory must not be a symlink")
    if not args.vla_only and not dry:
        binary = args.dji_binary or args.dji_path
        if not binary.is_file() or not os.access(binary, os.X_OK) or not has_vla_bridge(binary):
            raise RuntimeError(f"{binary} must be a prebuilt executable containing DJI_VLA_SOCKET and DJI_VLA_URL. Rebuild the VLA-enabled DJI integration or pass --dji-binary.")
        if navigation_config is not None and not has_vla_bridge(binary, continuous=True):
            raise RuntimeError(f"{binary} lacks DJI_VLA_CONTINUOUS_V2. Continuous navigation requires rebuilding dji_lte with the current native VLA bridge; pass that executable with --dji-binary. The older one-shot binary is supported only without --navigation-config.")
        if not compatible_binary(binary, platform.machine()):
            raise RuntimeError(f"{binary} must be a Linux ELF executable built for {platform.machine()}")
        if subprocess.run(["systemctl", "is-active", "--quiet", args.dji_service]).returncode == 0:
            raise RuntimeError(f"Stop {args.dji_service} while the drone is on the ground, then rerun the installer.")
        existing_start = ""
        if subprocess.run(["systemctl", "cat", args.dji_service], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            existing_start = subprocess.run(["systemctl", "show", args.dji_service, "--property=ExecStart", "--value"], capture_output=True, text=True, check=True).stdout.strip()
        if existing_start and not service_launches_binary(existing_start, args.dji_path):
            raise RuntimeError(f"Cannot verify that {args.dji_service} launches {args.dji_path}. Set --dji-path to the actual DJI binary; a shell wrapper must invoke that absolute path directly. ExecStart: {existing_start}")
    print(f"Installing {directory}; PyTorch profile: {profile}", flush=True)
    runner.run("apt-get", "update")
    runner.run("apt-get", "install", "-y", "python3-venv", "python3-pip", "curl", "ca-certificates", "rsync", "zstd", "xz-utils", "libopenblas-dev", "libjpeg-dev")
    if profile == "jetson-cu126":
        # Jetson PyTorch wheels use the host cuDNN runtime rather than bundling it.
        # Install from the configured JetPack repository; avoid a full JetPack upgrade.
        runner.run("apt-get", "install", "-y", "--no-remove", "libcudnn9-cuda-12")
        runner.run("ldconfig")
    # Stop only the service owned by this installer before updating its files/venv.
    if not dry and subprocess.run(["systemctl", "is-active", "--quiet", "drone-vla.service"]).returncode == 0:
        runner.run("systemctl", "stop", "drone-vla.service")
    runner.run("install", "-d", "-m", "0755", directory)
    if SOURCE.resolve() != directory.resolve():
        runner.run("rsync", "-a", "--exclude=.git", "--exclude=.venv", "--exclude=__pycache__", "--exclude=.DS_Store", "--exclude=.cuda", str(SOURCE) + "/", str(directory) + "/")
    if args.dji_binary and not args.vla_only:
        if args.dji_binary.resolve() != args.dji_path.resolve():
            if not dry and args.dji_path.exists() and not Path(str(args.dji_path) + ".before-vla").exists():
                runner.run("cp", "-p", args.dji_path, str(args.dji_path) + ".before-vla")
            runner.run("install", "-m", "0755", args.dji_binary, args.dji_path)
    if not dry and subprocess.run(["id", "-u", "drone-vla"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        runner.run("useradd", "--system", "--user-group", "--create-home", "--home-dir", "/var/lib/drone-vla", "--shell", "/usr/sbin/nologin", "drone-vla")
    elif dry:
        runner.run("useradd", "--system", "--user-group", "--create-home", "--home-dir", "/var/lib/drone-vla", "--shell", "/usr/sbin/nologin", "drone-vla")
    runner.run("install", "-d", "-m", "0755", "-o", "drone-vla", "-g", "drone-vla", "/var/lib/drone-vla/huggingface")
    for group in ("video", "render"):
        if dry or subprocess.run(["getent", "group", group], stdout=subprocess.DEVNULL).returncode == 0:
            runner.run("usermod", "-a", "-G", group, "drone-vla")
    python = directory / ".venv/bin/python"
    runner.run("python3", "-m", "venv", "--copies", directory / ".venv")
    runner.run(python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    remove_legacy_cuda13_packages(python, runner)
    # Install the platform wheels first. The pinned requirements cannot replace them with CUDA 13 wheels.
    matching_torch = False
    if not dry:
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = f"{directory}/.cuda/lib:/usr/local/cuda/lib64"
        matching_torch = subprocess.run([str(python), str(directory / "scripts/prepare_runtime.py"), "--profile", profile],
                                        env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not matching_torch:
        runner.run(python, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", "--index-url", TORCH_INDEXES[profile], "torch==2.8.0", "torchvision==0.23.0")
    runner.run(python, "-m", "pip", "install", "-r", directory / "requirements.txt")
    runner.run(python, "-m", "pip", "check")
    runner.write(directory / ".torch-profile", profile + "\n")
    if profile == "jetson-cu126":
        # Keep the NVIDIA cuSPARSELt redistributable isolated from the system CUDA installation.
        if dry:
            print(f"+ install NVIDIA cuSPARSELt 0.7.1 to {directory}/.cuda/lib when libcusparseLt.so.0 is missing")
        elif not (directory / ".cuda/lib/libcusparseLt.so.0").exists():
            host_libraries = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, check=True).stdout
            if "libcusparseLt.so.0" not in host_libraries:
                with tempfile.TemporaryDirectory(prefix="vla-cusparselt-") as temporary:
                    archive = Path(temporary) / "library.tar.xz"
                    runner.run("curl", "--fail", "--location", "--retry", "3", CUSPARSELT_URL, "--output", archive)
                    runner.run("tar", "-xJf", archive, "--no-same-owner", "-C", temporary)
                    libraries = Path(temporary) / CUSPARSELT_NAME / "lib"
                    if not libraries.is_dir():
                        raise RuntimeError("Unexpected NVIDIA cuSPARSELt archive layout")
                    runner.run("install", "-d", directory / ".cuda/lib")
                    runner.run("rsync", "-a", str(libraries) + "/", str(directory / ".cuda/lib") + "/")
    if navigation_config is not None:
        runner.run(python, "-c", "from navigation import NavigationConfig; import pathlib,sys; NavigationConfig.model_validate_json(pathlib.Path(sys.argv[1]).read_text())", args.navigation_config,
                   env=dict(os.environ, PYTHONPATH=str(directory)))
    runner.run("runuser", "-u", "drone-vla", "--", "env", "HF_HOME=/var/lib/drone-vla/huggingface", f"LD_LIBRARY_PATH={directory}/.cuda/lib:/usr/local/cuda/lib64", python, directory / "scripts/prepare_runtime.py", "--profile", profile, "--preload",
               *(("--navigation",) if navigation_config is not None else ()))
    ollama = shutil.which("ollama")
    if not ollama:
        if dry:
            runner.run("curl", "--fail", "--location", "https://ollama.com/install.sh", "--output", "<temporary>/ollama-install.sh")
            runner.run("sh", "<temporary>/ollama-install.sh")
        else:
            with tempfile.TemporaryDirectory(prefix="vla-ollama-") as temporary:
                script = Path(temporary) / "install.sh"
                runner.run("curl", "--fail", "--location", "--retry", "3", "https://ollama.com/install.sh", "--output", script)
                environment = os.environ.copy()
                if args.ollama_version:
                    environment["OLLAMA_VERSION"] = args.ollama_version
                runner.run("sh", script, env=environment)
                ollama = shutil.which("ollama")
                if not ollama:
                    raise RuntimeError("Official Ollama installer did not install an executable on PATH")
    ollama = ollama or "/usr/local/bin/ollama"
    # Existing manual Ollama installations may have a binary but no service.
    if dry or not subprocess.run(["systemctl", "cat", "ollama.service"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        if not dry and subprocess.run(["id", "-u", "ollama"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            runner.run("useradd", "--system", "--user-group", "--create-home", "--home-dir", "/usr/share/ollama", "--shell", "/usr/sbin/nologin", "ollama")
        runner.write(Path("/etc/systemd/system/ollama.service"), f"[Unit]\nDescription=Ollama model server\nAfter=network-online.target\n\n[Service]\nUser=ollama\nGroup=ollama\nExecStart={ollama} serve\nRestart=on-failure\nRestartSec=3\n\n[Install]\nWantedBy=multi-user.target\n")
    for group in ("video", "render"):
        if dry or subprocess.run(["getent", "group", group], stdout=subprocess.DEVNULL).returncode == 0:
            runner.run("usermod", "-a", "-G", group, "ollama")
    runner.write(Path("/etc/systemd/system/ollama.service.d/drone-vla.conf"), "[Service]\nEnvironment=OLLAMA_HOST=127.0.0.1:11434\n")
    runner.run("systemctl", "daemon-reload")
    runner.run("systemctl", "enable", "--now", "ollama.service")
    runner.run("systemctl", "restart", "ollama.service")
    if not dry:
        wait_http("http://127.0.0.1:11434/api/tags")
    runner.run("env", "OLLAMA_HOST=http://127.0.0.1:11434", ollama, "pull", "gemma3:4b")
    if not dry:
        # Warm the actual model so its first download/load does not consume dji-lte's 30s request timeout.
        request = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=json.dumps({"model": "gemma3:4b", "prompt": "Reply READY", "stream": False}).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as reply:
            result = json.load(reply)
            if "error" in result or not result.get("done"):
                raise RuntimeError(f"Ollama model warmup failed: {result}")
    if navigation_config is not None:
        runner.write(Path("/etc/drone-vla/navigation.json"), navigation_config)
    runner.write(Path("/etc/systemd/system/drone-vla.service"), service_text(directory, profile, args.vla_port,
                 navigation=navigation_config is not None, planner=args.planner))
    if not args.vla_only:
        if dry or not subprocess.run(["systemctl", "cat", args.dji_service], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            runner.write(Path("/etc/systemd/system") / args.dji_service, f"[Unit]\nDescription=DJI LTE flight-control application\nAfter=network-online.target\nWants=network-online.target\n\n[Service]\nUser=root\nExecStart={args.dji_path}\nRestart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=multi-user.target\n")
        runner.write(Path("/etc/systemd/system") / (args.dji_service + ".d/drone-vla.conf"), dji_dropin_text(args.vla_port))
    runner.write(Path("/usr/local/bin/drone-vla"), launcher_text(directory, args.dji_service, args.vla_only), 0o755)
    runner.run("systemctl", "daemon-reload")
    runner.run("systemctl", "enable", "--now", "drone-vla.service")
    if not dry:
        wait_http(f"http://127.0.0.1:{args.vla_port}/health", timeout=300)
    if not args.vla_only:
        runner.run("systemctl", "enable", args.dji_service)
        if args.start_dji:
            runner.run("systemctl", "start", args.dji_service)
    print("\nInstallation plan complete." if dry else "\nInstallation complete.")
    print("sudo drone-vla start\nsudo drone-vla status\nsudo drone-vla 'Hover'\nsudo drone-vla logs")


def main():
    args = parse_args()
    try:
        install(args, Runner(args.dry_run))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Installation failed: {error}\nFix the reported step and rerun; completed downloads and model caches are retained.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
