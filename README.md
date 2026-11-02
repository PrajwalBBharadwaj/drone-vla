# drone-vla

For continuous camera navigation, native bridge build/deployment, and timing
instrumentation, see [Persistent camera navigation](NAVIGATION.md). It grounds a
destination once and uses fresh-frame tracking/depth for subsequent commands.
`drone-vla navigate "Approach that blue building"` submits a goal once; DJI's
background worker drives subsequent updates independently of terminal polling.
This requires the rebuilt protocol-v2 DJI binary and calibrated navigation JSON.
Ordinary `drone-vla "motion instruction"` requests retain the one-shot path below.

The DJI integration uses this flow:

```text
Instruction -> dji-lte + its decoded camera frame -> VLA service
            <- final body velocity/yaw command  <-
               standby/manual flight-control loop
```

The camera belongs to dji-lte. Its decoded RGB callback supplies the latest wide
camera frame. A dji-lte background worker sends those pixels and the natural-language
instruction to this repository's VLA service. The service runs the existing
`integrated_control.plan_motion()` and returns the applied command. dji-lte
validates the result and applies it through its existing manual loop, including
geofence, altitude, and terrain controls.

Install on a new Linux companion computer with the setup script. Supported
targets are Ubuntu 22.04/24.04 or Debian 12 with systemd and Python 3.10–3.12.
JetPack 6.2.x is detected automatically and uses Python 3.10 CUDA 12.6 wheels.
Other Linux machines default to CPU PyTorch; an NVIDIA desktop with a compatible
driver can select `--torch-profile cuda126`.

Copy this checkout to the target and run, with the drone on the ground and the
existing DJI service stopped:

```sh
sudo systemctl stop dji-lte.service
cd /opt/drone-vla
sudo bash install.sh
```

You can also run the installer from another checkout directory; it copies the
runtime to `/opt/drone-vla` and excludes the source virtual environment. Internet
access is needed for packages and the initial model downloads. After installation,
the cached models run locally.

The script:

- Installs Linux prerequisites, creates `.venv` with copies, and upgrades pip.
- Installs pinned PyTorch/torchvision wheels from the appropriate CPU, CUDA, or
  Jetson index before installing the pinned Python requirements.
- Tests a real GPU operation for CUDA profiles. On Jetson, it supplies missing
  cuSPARSELt libraries in `/opt/drone-vla/.cuda/lib`, without changing system CUDA.
- Installs the Jetson cuDNN 9 runtime (`libcudnn9-cuda-12`) from the configured
  JetPack APT repository before checking PyTorch. Removes leftover NVIDIA CUDA 13
  Python packages from the VLA virtual environment when reusing an earlier setup.
- Installs the Ollama server through its official installer when missing, starts
  its service, downloads `gemma3:4b`, and warms the model.
- Downloads the depth model into `/var/lib/drone-vla/huggingface` using the
  dedicated `drone-vla` service account.
- Creates `drone-vla.service`, with depth-model preloading and HTTP readiness
  checks, and configures the DJI service to wait for it.
- Adds the DJI socket/URL environment through a systemd drop-in, preserving the
  existing DJI service's executable arguments and hardware configuration.
- Installs the `/usr/local/bin/drone-vla` command launcher.

The installer requires a prebuilt **VLA-enabled** DJI executable, normally at
`/usr/local/bin/dji_lte`. It checks for the bridge's environment markers before
installing anything. A stock dji-lte binary needs the camera/VLA bridge built in
first; the Python installer cannot add that code to an existing executable.
When `--navigation-config` enables continuous goals, the binary must also contain
`DJI_VLA_CONTINUOUS_V2`. The installer rejects the older one-shot bridge before
changing packages or services. The native source and standalone test build now
live in the DJI repository; deploying it still requires rebuilding the complete
DJI executable with its SDK/toolchain.
An existing device's DJI credentials, UART settings, camera setup, and vendor
CUDA/JetPack installation remain prerequisites for the flight application.

To deploy a freshly built DJI executable as well:

```sh
sudo bash install.sh --dji-binary /path/to/build/bin/dji_lte
```

For continuous navigation, first complete the measured calibration fields in
`navigation.example.json`, then deploy while grounded:

```sh
sudo bash install.sh --dji-binary /path/to/build/bin/dji_lte \
  --navigation-config /absolute/path/navigation.json --start-dji
```

See [the navigation guide](NAVIGATION.md) for camera alignment, protocol details,
HITL validation, and the goal CLI. The example JSON intentionally fails
validation until camera and vehicle measurements have been supplied.

The original executable is backed up once as `dji_lte.before-vla`. Use
`--dji-path` or `--dji-service` if your executable/service has another location
or name. The installer preserves existing unit settings and creates a minimal
root DJI service only if no unit exists.

Ollama and VLA start during installation. The DJI service is enabled for subsequent
boots, but is started immediately only with `--start-dji`, or with the launcher
after installation:

```sh
sudo drone-vla start
sudo drone-vla status
```

In the simulator first, enable airborne standby/manual mode, set the operator
joystick to zero, and point the wide camera forward. Submit text:

```sh
sudo drone-vla "Hover"
sudo drone-vla "Move forward at 1 m/s for 1 second"
```

The launcher also supports:

```sh
sudo drone-vla logs       # Follow Ollama, VLA, and DJI logs
sudo drone-vla check      # Verify PyTorch/GPU
sudo drone-vla stop       # Stop DJI and VLA; leave the shared Ollama service running
```

The services use these paths:

| Component | Location |
| --- | --- |
| Python application and virtual environment | `/opt/drone-vla` |
| DJI executable | `/usr/local/bin/dji_lte` |
| Depth-model cache | `/var/lib/drone-vla/huggingface` |
| DJI instruction socket | `/run/dji-vla/control.sock` |
| VLA HTTP endpoint | `http://127.0.0.1:8766/plan` (override with `--vla-port`) |
| DJI service drop-in | `/etc/systemd/system/dji-lte.service.d/drone-vla.conf` |

Socket presence alone does not establish flight eligibility. The bridge accepts
movement only during airborne Standby/manual control with the operator idle;
mode/session changes invalidate pending movement. A missing socket means the
native bridge is unavailable or has not initialized.

VLA logs the received camera dimensions and the elapsed time for instruction,
vision, depth, and the entire HTTP request. Model output is bounded to 256 tokens
for instruction/scene JSON and 128 tokens for the image description; truncated
responses are rejected. Model calls keep Gemma loaded for subsequent requests.
A disconnected DJI client discards the result. These limits do not extend DJI's
inference deadline: measure the stages on the target Jetson before flight use.
For a communications-only Hover test, `sudo drone-vla "Hover" --timeout 0`
disables the CLI wait limit. This still respects DJI's separate compiled HTTP
deadline; changing that deadline requires the VLA-enabled DJI bridge source.
Exact Hover/Stop/Hold/Hold position requests produce zero motion directly,
without a model call. The new native bridge can handle that hold locally without
a camera frame. Compound instructions still use the model. A fast Hover test
checks command acceptance, not camera transfer, visual assessment, SDK execution,
or movement-inference latency.

For temporary commands-only operation, set `VLA_COMMANDS_ONLY=1` in
`/run/systemd/system/drone-vla.service.d/commands-only.conf` under a `[Service]`
section, then reload systemd and restart the services while grounded. This runtime
override is removed automatically on reboot. To restore vision earlier, delete
that override, reload systemd, and restart while grounded. Do not put this
temporary setting in `/etc/systemd/system`. Vision remains the default mode.
The service also accepts `--commands-only` for direct execution.
This mode uses only the natural-language command interpreter. It skips image
assessment, depth assessment, and depth-model preloading. DJI still sends and
validates its camera payload; the existing SDK/flight limits and the 30-second
DJI inference deadline still apply. `/health` reports `mode: commands-only`, and
each result reports `Visual checks bypassed (commands-only mode)`. Requests must
include a numerical speed for movement; results over the control limits are
rejected.

Installation can be rerun after a failed download; matching PyTorch wheels and
model caches are reused. Stop DJI first. To preview all commands without changing
the machine, or install just VLA on a computer without a DJI executable:

```sh
bash install.sh --dry-run --torch-profile jetson-cu126
sudo bash install.sh --vla-only --torch-profile cpu
```

The one-shot CLI sends only text to dji-lte and prints the result once dji-lte accepts it
for the manual loop. The existing flight constraints can further limit movement;
SDK execution failures are reported in dji-lte's logs.
It has no camera-device or test-image option. Frames come only from dji-lte's
live camera callback. Movement needs the calibrated wide RGB stream; the new
bridge rejects a missing frame or a decoded frame older than 750 ms at
submission. The old one-shot bridge used a 500 ms limit. Existing standalone
image prototypes remain available for offline development.

Each one-shot instruction performs one inference and one bounded movement. While planning,
dji-lte holds zero VLA motion. HTTP inference runs outside the flight thread with
a 30-second timeout. Movement lasts one second by default, at most two seconds.
Limits are 2 m/s combined horizontal speed, 1 m/s vertical speed, and 30 degrees/s
yaw. Commands above those limits are rejected. Operator joystick movement
cancels VLA motion and invalidates pending inference. In the new native bridge,
routine centered-stick messages do not cancel an owned goal. A session/mode change also
discards the result. Motion returns to zero at expiry, and the integration never
initiates takeoff or enables manual mode.

The one-shot path receives a frame taken before inference and does not replan
that movement. Continuous `navigate` instead lets the native worker request
fresh-frame corrections for the same stored goal and refresh an expiring
setpoint while the manual-control loop runs. Correlated session/goal/frame IDs,
frame-age checks, command expiry, and operator takeover prevent an old result
from extending movement. This changes scheduling, not model speed: Jetson
inference timing and full SDK behavior still require profiling/HITL validation.
See [the navigation guide](NAVIGATION.md) and the DJI README for native tests.

Run Python tests:

```sh
.venv/bin/python -m unittest -v \
  test_install_linux test_dji_control test_navigation test_single_pass \
  test_bridge_protocol test_continuous_control
```

Build the standalone native bridge with `-DDJI_VLA_STANDALONE_TESTS=ON` as
described in [NAVIGATION.md](NAVIGATION.md). Then run actual C++/Python HTTP
round-trip tests with that executable (no model calls or SDK commands):

```sh
VLA_TEST_BINARY=/tmp/dji-vla-build/test-vla-bridge \
  .venv/bin/python -m unittest -v test_vla_http test_continuous_http
```

The `/plan` request body is a four-byte big-endian JSON-header length, the JSON
header (`request_id`, `text`, `width`, `height`, `format=RGB`, `default_duration`),
and packed RGB bytes. The response carries the request ID, final signed control
axes, duration, and the planner's override reason. Continuous protocol-v2 updates
also carry and echo `bridge_protocol`, `goal_id`, `session`, `frame_id`, and
`frame_monotonic_ns`; the last value timestamps the decoded-frame callback using
explicit `clock_gettime(CLOCK_MONOTONIC)` in both processes, not sensor exposure.
DJI continuously supplies those updates after an accepted
`navigate` request. Grounding is asynchronous and returns zero motion until a
fresh-frame target is ready. The continuous CLI's default/max movement lease is
0.35 seconds; legacy steps default to 0.2 seconds. Initial grounding defaults to
a 60-second deadline, configurable up to 120 seconds, with bounded 5 Hz frame
history. Those limits do not establish measured inference latency.

Installation sources: [Ollama Linux installer](https://docs.ollama.com/linux),
[NVIDIA Jetson CUDA 12.6 wheels](https://forums.developer.nvidia.com/t/looking-for-pytorch-compatible-with-cuda-and-jetpack-6-2/348897),
[PyTorch version/index mapping](https://pytorch.org/get-started/previous-versions/),
and [NVIDIA cuSPARSELt redistributables](https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/).
