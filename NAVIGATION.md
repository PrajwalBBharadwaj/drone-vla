# Persistent camera navigation

`drone-vla navigate` submits a destination once. The native DJI bridge then
supplies fresh camera frames and receives corrected velocity/yaw commands for
that goal until completion or cancellation. The terminal monitors the goal; it
does not drive each camera update.

The initial image and language instruction go to Gemma through Ollama once.
While grounding runs, all navigation commands are zero. Afterwards, optical-flow
tracking, a resident small metric-depth model, and a local geometric planner
produce subsequent commands without another language-model call.

## Processes and continuous updates

```mermaid
sequenceDiagram
    participant CLI as Natural-language terminal
    participant DJI as dji_lte native bridge
    participant VLA as Python navigation service
    participant LLM as Ollama / Gemma 3
    CLI->>DJI: navigate (instruction, session, ticket, goal ID)
    DJI-->>CLI: Accepted goal
    DJI->>VLA: Goal start + timestamped RGB frame (HTTP /plan)
    VLA->>LLM: Ground destination in the image
    VLA-->>DJI: Zero axes; grounding pending
    loop Native worker receives a fresh decoded frame
        DJI->>VLA: Same goal/session + next frame
        VLA->>VLA: Track destination, estimate depth, plan local motion
        VLA-->>DJI: Correlated frame/goal/session + expiring setpoint
        DJI->>DJI: Manual loop applies latest valid setpoint
        CLI->>DJI: Read goal status
        DJI-->>CLI: State, axes, reason and timings
    end
    CLI->>DJI: Cancel owned goal on exit
```

These are four OS processes: the terminal CLI, `dji_lte`, the Python VLA service,
and Ollama. Only DJI owns the camera and flight SDK. HTTP inference runs in a
native background worker; it does not block the flight-control loop. The native
worker replaces the old frame with the latest available frame rather than
building a queue of stale inference requests. It permits one inference at a time.

The manual-control loop keeps dispatching the latest unexpired command between
perception updates. Starting another update does not itself zero that command.
No command survives its lease or the maximum age of the frame supporting it.
The continuous CLI defaults to a 0.35-second movement lease, which is also the
native maximum. Legacy steps default to 0.2 seconds. The continuous bridge
rejects supporting frame ages over 0.75 seconds. Consequently,
inference slower than the lease can still produce holds between updates. These
are rejection limits, not achieved latency or update-rate measurements.

Each update carries a goal ID, native session, sequence, and frame identifier.
The Python response must echo correlation data before DJI accepts its axes.
Results from an older session/goal/frame cannot overwrite the current command.
Native frame timestamps are taken at the decoded RGB callback using explicit
`clock_gettime(CLOCK_MONOTONIC)` in both processes, **not camera exposure time**.
Python checks frame age on arrival and after processing; native dispatch also
checks supporting-frame age. Capture/decoder buffering needs separate measurement.

An operator movement, explicit cancellation, eligibility loss, stale frame,
expired command, failed request, or session change clears navigation motion.
Routine centered-stick messages do not constitute a takeover in continuous
navigation; deliberate operator input always takes priority. The terminal cancels
only the goal/session that it owns. Existing SDK flight limits still apply.

## What the local behavior does

For `Approach the blue building while avoiding the tree`, grounding selects the
building. Following frames use optical flow with forward/backward, RANSAC, and
appearance checks. Depth Anything V2 Metric Outdoor Small estimates depth. The
controller checks an inflated vehicle footprint and stopping distance, turns
toward the tracked destination, and then approaches at constant altitude. Target
bearings to the left produce negative yaw; bearings to the right produce positive
yaw. Outside the configured `yaw_deadband_deg` (default 3 degrees), it turns in
place. Once aligned, it moves forward while applying small yaw corrections. Every
fresh image supplies a new correction; an initial command is never replayed as a
complete route. It stops at the configured stand-off distance.
Stand-off is estimated from depth inside the tracked target box. An occluder or
depth error can cause a premature hold; this is not independent verification that
the intended destination has been reached.

This is a bounded local approach behavior. It does not provide a global map,
long-lived detour waypoints, arbitrary motion sequences, landing, or guaranteed
navigation around an occluding tree. An immediate obstacle in the current forward
corridor stops all axes, including yaw. It does not choose a side detour in order
to keep moving. A blocked goal ends and requires a new instruction.

Grounding and tracker initialization always return zero axes. Buffered frames
are sampled at 5 Hz during grounding and tracked in bounded batches while new
frames arrive; `acquiring` holds until tracking catches up. Buffer capacity is
tied to the grounding deadline, which defaults to 60 seconds and can be configured
up to 120 seconds. A subsequent fresh frame is required before movement.
Ambiguous targets, uncertain tracking, invalid depth, processing overruns,
interrupted updates, and goal expiry hold motion. Partly visible targets can
continue tracking: crossing an image edge does not by itself mean that the
destination disappeared. Failed optical flow attempts conservative appearance
recovery using the existing target, without another VLM call. Ambiguous matches
are rejected rather than used to select a different object.

Tracking uncertainty reports `target.visibility="unconfirmed"` and a zero-command
`stale` state. `out_of_view` is reserved for reliable tracking geometry showing
that the entire target box left the image; weak features or occlusion alone do
not establish absence. The service retries on fresh frames for
`target_recovery_seconds` (default 5 seconds). A recovered target first returns a
zero `ready` response; movement needs another fresh successful update. Recovery
expiry ends the goal as `lost`. These labels describe visual evidence, not a
guarantee that a camera-only system can always determine whether an object exists
outside the view. Lost/blocked goals require a new instruction.

Manual stick input always takes over and cancels the active native goal. After
manually yawing until the destination is visible, center the stick and submit the
destination again. The service does not automatically resume motion after an
operator takeover. Processing overruns can retry on the next fresh frame. Cancelled
grounding jobs cannot produce motion, and another grounding job is rejected
until the earlier worker finishes.

### Live destination location and distance

Every destination update includes `navigation.target` in native status. The
terminal and Python service print a compact command/target line by default:

```text
aligning cmd(f=+0.00 r=+0.00 u=+0.00 yaw=+3.64) bbox=[661,366,1450,933] center=[1055,649] conf=0.935 range=8.80m bearing=+4.55deg visible 216ms
```

Full observations remain in `--metrics` JSON; `--verbose-json` prints them in the
terminal. Start the service with `--log-level DEBUG` or `VLA_LOG_LEVEL=DEBUG` to
inspect detailed bridge/request traces. The wire correlation and freshness checks
are unchanged. An example full target observation is:

```json
{
  "instruction": "Approach the blue building",
  "description": "One blue building",
  "visibility": "visible",
  "fresh": true,
  "image_size_px": [1920, 1080],
  "center_px": [1440.0, 540.0],
  "box_px": [1248.0, 270.0, 1632.0, 810.0],
  "bearing_deg": 21.801,
  "forward_distance_m": 19.5,
  "distance_m": 21.002,
  "position_body_m": {"forward": 19.5, "right": 7.8, "up": 0.0},
  "distance_source": "monocular_metric_estimate"
}
```

Pixels refer to the **original calibrated camera frame**, even though perception
uses a smaller image. The origin is the upper-left corner; x increases right and
y increases down. The normalized box is also included. `forward_distance_m` is
the depth-scaled median estimated optical-axis depth inside the current target
box. `distance_m` is its approximate slant range along the box-center camera ray;
the body position assumes the configured forward-aligned camera. These are learned
monocular estimates, not measured robot-to-object range or world coordinates.
An occluder or a large background-containing box can affect them.

The observation also carries its sequence, decoded-frame ID/timestamp, and initial
grounding confidence. `fresh=false` marks an observation that cannot support
current motion. Unconfirmed tracking has null current pixel/distance fields,
so a last-known coordinate is not presented as a current observation.

```sh
sudo journalctl -u drone-vla.service -f -o cat
sudo drone-vla navigate "Approach that building on the left" \
  --max-seconds 90 --metrics /tmp/destination-tracking.json
```

### Target box on the operator video

The Python HTTP service exposes `GET /overlay`, a small latest-only display cache
updated after validating each navigation response. DJI LTE polls it in a background
worker and draws the target box, its geometric center, confidence, estimated range,
and observation age on the RGB video copy before conversion to YUV and writing to
`/dev/video0`. The existing video-streaming service then carries those annotations.
This does not invoke the model again or alter the pixels used for perception.

The center is the middle of the selected bounding box, not a separately detected
roof or facade point. Seeing the rectangle lets you check whether the VLM selected
too much ground/background. The overlay applies normalized coordinates to the
original image dimensions; it does not introduce an arbitrary vertical correction.

Polling uses `DJI_VLA_URL` with `/plan` replaced by `/overlay` (default
`http://127.0.0.1:8766/overlay`); `DJI_VLA_OVERLAY_URL` can override it. The camera
callback only reads the local cache, so network/inference work does not block video.
Requests retrieve metadata only, never another camera frame. Full-HD input and
video dimensions must match. Thermal/zoom views hide the box. Clear/invalid/old
observations and transport failures clear it; repeated reads do not renew its
750 ms lifetime from the original decoded-frame timestamp. On the VLA-enabled
native branch, operator takeover and control/source epoch changes also invalidate
the overlay, including late responses from an old goal.

The annotation follows the latest completed perception update, so it can trail
the live video by the reported observation age; it is not an exact exposure-time
annotation of every video frame. No display-frame buffering is added.

```sh
curl http://127.0.0.1:8766/overlay
```

Before a valid visible target exists, this returns
`{"protocol":1,"action":"clear"}`. The video overlay requires rebuilding DJI LTE
with the new camera overlay code; Python-only installation still supplies compact
logs and the metadata endpoint. Keep the native VLA control bridge when rebuilding.

## Build and deploy

Deploy or restart services **with the drone on the ground**. Continuous navigation
requires rebuilding `dji_lte` with the current native VLA source. An older
one-shot binary cannot become continuous through a Python update.

The installer checks for `DJI_VLA_SOCKET`, `DJI_VLA_URL`, and
`DJI_VLA_CONTINUOUS_V2` in the selected executable whenever
`--navigation-config` is requested. It rejects an old binary before installing
packages or changing services. One-shot-only installations do not require the
new marker. `--vla-only` can configure perception without a local DJI binary.

Build the full DJI executable using the target's normal SDK/toolchain and device
configuration. To build the standalone bridge tests from the DJI repository:

```sh
cmake -S . -B /tmp/dji-vla-build -DDJI_VLA_STANDALONE_TESTS=ON
cmake --build /tmp/dji-vla-build
ctest --test-dir /tmp/dji-vla-build --output-on-failure
```

That option bypasses the vendor SDK build. It exercises transport and command
arbitration; it does not build the full flight application or prove SDK dispatch
or flight behavior. The bridge core and native/HTTP tests have been compiled and
run locally. The complete SDK application and Jetson build remain unverified in
this macOS workspace.

1. Copy this updated Python checkout and the newly built DJI executable to the
   drone.
2. Copy `navigation.example.json` to a separate configuration file. Fill the
   `null` values with measured camera intrinsics, vehicle radius including rotors,
   and a conservatively measured braking deceleration. The example deliberately
   fails validation until completed.
3. Match `image_width`/`image_height` to the uncropped, rectified wide RGB camera
   frame. `fx`, `fy`, `cx`, and `cy` use that frame's pixel coordinates. The
   controller rescales them internally without cropping.
4. Lock the camera facing along the drone's forward body axis. Set
   `camera_forward_aligned` to `true` only then. Moving gimbals, camera rotation,
   or distorted images require a camera transform/rectification implementation.

From the transferred Python checkout, with the completed configuration:

```sh
sudo systemctl stop dji-lte.service
sudo bash install.sh \
  --dji-binary /absolute/path/new-dji_lte \
  --navigation-config /absolute/path/navigation.json \
  --start-dji
curl http://127.0.0.1:8766/health
```

If the rebuilt executable is already at `/usr/local/bin/dji_lte`, omit
`--dji-binary`. The health response must include `"mode":"vision"` and
`"navigation":"enabled"`. Inspect `sudo systemctl cat drone-vla.service` and
remove any earlier runtime commands-only override before restarting. Navigation
refuses to operate with visual checks bypassed; vision remains the default.

The installer preloads the small depth model and writes
`/etc/drone-vla/navigation.json`. Code/venv remain in `/opt/drone-vla`, the DJI
executable in `/usr/local/bin/dji_lte`, and model caches in
`/var/lib/drone-vla/huggingface`. It preserves the existing DJI service wrapper
and separate video-stream service. It does not upgrade JetPack/CUDA.

### Update an already deployed continuous bridge

Destination steering and compact logging can use an already deployed
`DJI_VLA_CONTINUOUS_V2` executable. Drawing the target on the operator video also
requires the native camera overlay changes and a rebuilt executable that still
contains `DJI_VLA_CONTINUOUS_V2`. Copy the updated checkout to the EXT using your
working SCP address, then, with the physical aircraft grounded:

```sh
sudo systemctl stop dji-lte.service drone-vla.service
sudo install -m 0644 \
  ~/drone-vla-src/navigation.py \
  ~/drone-vla-src/navigation_perception.py \
  ~/drone-vla-src/navigation_control.py \
  ~/drone-vla-src/dji_control.py \
  ~/drone-vla-src/bridge_protocol.py \
  ~/drone-vla-src/navigation_logging.py \
  ~/drone-vla-src/target_overlay.py \
  ~/drone-vla-src/vla_service.py \
  /opt/drone-vla/
sudo systemctl start drone-vla.service dji-lte.service
curl http://127.0.0.1:8766/health
```

For the normal calibrated destination behavior, remove the temporary Python
forward-stop runtime drop-in and optional native simulator-speed drop-in using
the cleanup instructions below before restarting. They adjust the diagnostic
clearance/speed profile. An object destination now tracks and steers even if the
Python forward-test switch remains set, and never receives its diagnostic output
speed or the forward-test marker used by the optional native speed override.

## Run and observe

Validate in DJI simulation/HITL first. With airborne Standby/manual eligibility,
the operator joystick centered, and the calibrated camera facing forward:

```sh
sudo drone-vla navigate "Approach that blue building while avoiding the tree" \
  --max-seconds 90 --metrics /tmp/navigation-timing.json
```

The CLI submits `op=navigate` once, then monitors native status. Frames and
updates continue inside DJI. Ctrl-C cancels the owned goal. Neither this CLI nor
the Python navigation service opens a camera or drives the DJI SDK directly.

Without the installed launcher:

```sh
sudo /opt/drone-vla/.venv/bin/python /opt/drone-vla/navigation_control.py \
  --socket /run/dji-vla/control.sock "Approach the blue building"
```

The new CLI rejects an old binary for the default continuous mode. The explicit
`--legacy-steps` option retains the older repeated-instruction transport for
backward-compatibility work. That path has stop/start gaps between short
movements; operator-message cancellation depends on the binary version. It is not the
continuous controller, and the installer does not select it automatically.

## Timing and limits

### Temporary hybrid-HITL forward/stop test

When DJI simulation moves the simulated vehicle but the real camera stays still,
the normal heading controller cannot see its turns taking effect. Lowering
clearance alone does not fix that missing feedback.

For this diagnostic, set `VLA_SIM_FORWARD_STOP_TEST=1` in a **runtime-only**
`drone-vla.service` drop-in (or pass `--sim-forward-stop-test`). Navigation and
visual checks must remain enabled. The service derives an in-memory profile:

- By default speed is capped at 0.2 m/s, preserving any already slower limit.
- Additional vehicle clearance margin becomes 0.2 m.
- Minimum forward lookahead becomes 2 m; the calculated stopping horizon can
  still require more room.
- Only the current forward corridor is checked. Clear means forward motion with
  zero yaw; blocked means all-zero axes, even when a side route is clear.
- Destination bearing, vertical target alignment and arrival stand-off do not
  steer or finish this diagnostic. This is a forward/stop test, not goal navigation.

Measured vehicle radius, braking deceleration, depth scale, camera intrinsics,
source-frame freshness and native command leases stay unchanged. No saved
calibration is rewritten. Learned depth remains an estimate, so the configured
lookahead is not a guaranteed physical stopping distance.

For literal commands such as `move forward`, `go forward`, or `forward`, this
diagnostic needs no object target, LLM grounding, or tracker. It accepts the intent
with a zero `ready` response, then requires another fresh frame and a successful
depth check before moving. It recognizes complete phrases, so a negation or
compound instruction cannot accidentally match a forward substring.

If an object destination is supplied instead, normal destination grounding,
tracking, arrival checks and steering run even when this switch is enabled.
Only the literal forward intents use the target-free corridor diagnostic. Object
goals do not emit the `sim-forward-stop-test` marker and do not use
`VLA_SIM_FORWARD_SPEED_MPS`. The in-memory low-speed/clearance profile still exists
while the switch is enabled; remove the runtime drop-in to restore the original
calibrated profile for destination testing. With a static real camera and an
independent simulator, an off-center target will keep requesting a turn until you
physically pan the camera toward the target. Simulated yaw alone cannot supply
that visual feedback.
For the obstacle check, look for `blocked` with
`SIM forward-stop: obstacle in the current forward corridor`. Source-stream gaps,
bad depth and expired commands still hold. Blocked/lost goals end and require a new
instruction to repeat the test. For object destinations the brief selection log
reports the model explanation and confidence; DEBUG logs also include the initial
normalized box and bridge identifiers.

With the physical aircraft secured in DJI simulation, create the runtime drop-in
while the control processes are stopped:

```sh
sudo systemctl stop dji-lte.service drone-vla.service
sudo install -d /run/systemd/system/drone-vla.service.d
sudo tee /run/systemd/system/drone-vla.service.d/sim-forward-stop.conf >/dev/null <<'EOF'
[Service]
Environment=VLA_SIM_FORWARD_STOP_TEST=1
EOF
sudo systemctl daemon-reload
sudo systemctl start drone-vla.service dji-lte.service
curl http://127.0.0.1:8766/health
```

Health reports `"navigation_behavior":"approach"` and
`"sim_forward_stop_test_enabled":true`, with
`"directional_behavior":"sim-forward-stop-test"`. Literal-forward CLI motion
reasons start with `SIM forward-stop`. For a target-free corridor test, run:

```sh
sudo drone-vla navigate "move forward" --max-seconds 90
```

Change the real camera view after the forward command appears. No object needs to
stay in view for this temporary diagnostic.

To restore normal behavior immediately:

```sh
sudo systemctl stop dji-lte.service drone-vla.service
sudo rm -f /run/systemd/system/drone-vla.service.d/sim-forward-stop.conf
sudo systemctl daemon-reload
sudo systemctl start drone-vla.service dji-lte.service
```

The runtime drop-in also disappears at reboot. Health then reports
`"navigation_behavior":"approach"` and `"sim_forward_stop_test_enabled":false`.
This option does not activate DJI simulation
or verify simulator state; it is exclusively a diagnostic for an externally
configured HITL setup, not a real-flight obstacle-avoidance profile.

#### Python-only simulator output speed

The existing deployed bridge accepts up to 2 m/s. To increase this diagnostic's
forward setpoint without rebuilding DJI, update `navigation.py` and
`vla_service.py`, then add the second environment setting to the same runtime
Python drop-in:

```ini
[Service]
Environment=VLA_SIM_FORWARD_STOP_TEST=1
Environment=VLA_SIM_FORWARD_SPEED_MPS=2
```

The equivalent service argument is `--sim-forward-speed-mps 2` alongside
`--sim-forward-stop-test`. Speeds must be finite, positive and at most 2 m/s.
The speed setting alone cannot enable the test mode. Keep the normal saved
`max_speed_mps` unchanged; its 1 m/s navigation limit still applies outside this
diagnostic. Leave the native `DJI_VLA_SIM_FORWARD_SPEED_MPS` override unset or zero
when using the Python-only option.

Restart the stopped control services after `systemctl daemon-reload`. Health
reports `sim_forward_speed_mps: 2` and `planning_speed_mps: 0.2` (or the previously
configured lower planning speed). A clear corridor returns `forward: 2` and zero
other axes; a blocked corridor still returns all zeros. Source freshness, leases
and operator takeover remain enforced; literal forward intents need no initial
grounding. Removing the Python runtime
drop-in and restarting restores normal navigation; reboot also removes it.

This changes only the output for the independent-camera HITL diagnostic. The
perception corridor still uses the low-speed planning profile; it does not
establish real-flight stopping distance at the selected output speed. Python-only
changes cannot produce 15 m/s through a native bridge that rejects above 2 m/s.

#### Optional HITL velocity override

To exercise a faster simulator setpoint while using the independent real camera,
the native bridge supports `DJI_VLA_SIM_FORWARD_SPEED_MPS` in `(0, 15]`; zero or
unset disables it. This requires rebuilding and deploying the updated `dji_lte`
binary and updating `navigation.py`. Keep the calibrated `max_speed_mps` unchanged.

Both opt-ins are required: Python must run the forward-stop test above, and the
DJI process must have the native override configured. Python labels its responses
`navigation.behavior="sim-forward-stop-test"`. Only a fresh, correlated,
`approaching` response with positive forward and zero other axes gets the native
speed override. Ordinary navigation, one-shot commands, and normal wire validation
retain their original limits. Grounding, blockage, stale frames, cancellation,
authority loss and command expiry cannot trigger the override.

This is an actual SDK velocity setpoint override, not a simulator graphics setting.
The operator must ensure DJI simulation is active: the bridge cannot verify that
state from the available SDK interface. The camera still evaluates the existing
low-speed test corridor. This hybrid test demonstrates command delivery and a
zero-command transition; it does **not** validate obstacle avoidance, braking or
stopping distance at 15 m/s. Simulator braking can continue after a zero setpoint.

With the physical aircraft secured in DJI simulation and services stopped, create
an additional runtime drop-in for the rebuilt DJI executable:

```sh
sudo install -d /run/systemd/system/dji-lte.service.d
sudo tee /run/systemd/system/dji-lte.service.d/sim-forward-speed.conf >/dev/null <<'EOF'
[Service]
Environment=DJI_VLA_SIM_FORWARD_SPEED_MPS=15
EOF
sudo systemctl daemon-reload
sudo systemctl start drone-vla.service dji-lte.service
```

The native status response reports `sim_forward_speed_mps` (configured value),
`sim_forward_override_active` (current setpoint), and the actual `command.forward`.
Navigation status also retains `original_planner_forward_mps` and
`sim_forward_override_applied`. For a clear corridor in this test, expect original
planner speed 0.2 m/s and native forward speed 15 m/s. Changing the environment
requires restarting DJI, since the bridge reads it once at initialization.

Remove this override after testing while the services are stopped:

```sh
sudo systemctl stop dji-lte.service
sudo rm -f /run/systemd/system/dji-lte.service.d/sim-forward-speed.conf
sudo systemctl daemon-reload
sudo systemctl start dji-lte.service
```

Remove the Python forward-stop drop-in as described above to restore normal goal
navigation. Both runtime drop-ins disappear at reboot.

### Performance measurements

The changes remove repeated language inference after grounding, keep the depth
model resident, use bounded processing dimensions and CUDA FP16 when available,
and prepare images in memory. The depth backend is injectable for later measured
optimization; TensorRT export/runtime integration is not implemented.

Initial grounding can still take seconds or longer. Background grounding holds
the drone; it does not make the model faster. The 350 ms processing budget is a
rejection threshold, **not measured inference performance**. Repeated stale/expiry
holds indicate a perception/transport budget problem that requires profiling.
Raising deadlines alone does not improve latency.

Keep these measurements separate:

- Instruction acceptance to first valid movement, including semantic grounding.
- Decoded-frame timestamp to native acceptance/dispatch of a supported setpoint.
- Service processing time after its HTTP body has arrived.
- Physical response, which requires vehicle telemetry and separate validation.

CLI polling is not a perception timing source. Likewise, a decoded-frame
timestamp is not a sensor exposure timestamp. Report percentiles and maximum
frame age on the target Jetson before claiming subsecond performance.

Learned monocular metric depth has scale and domain errors. Configured depth
discounts and margins are heuristics rather than calibrated guarantees. Thin
obstacles, motion, occlusion, blind regions, and incorrect semantic grounding
remain limitations. Tracking does not independently verify semantic identity.
This experimental local behavior requires camera/depth and HITL validation;
on-drone timing and flight performance remain unverified.

## One-shot compatibility and local verification

Ordinary `drone-vla "motion instruction"` requests still use the one-shot path.
The planner defaults to `legacy`. Optional `--planner single-pass` combines
instruction interpretation and scene assessment into one multimodal call using
an in-memory JPEG. It supports forward/yaw/hold and rejects unobserved
backward/lateral/vertical motion, ambiguous commands, and explicit durations.
It is separate from the persistent goal loop.

```sh
.venv/bin/python -m unittest \
  test_dji_control test_install_linux test_navigation test_single_pass \
  test_bridge_protocol test_continuous_control test_destination_navigation \
  test_target_tracking test_target_telemetry_cli
```

The destination endpoint can also be exercised with real OpenCV tracking and
RGB/JSON HTTP requests, without a native executable:

```sh
.venv/bin/python -m unittest test_destination_http test_http_upload test_target_overlay test_navigation_logging
```

After compiling the standalone native target above, enable the actual
C++/Python HTTP tests with its executable path:

```sh
VLA_TEST_BINARY=/tmp/dji-vla-build/test-vla-bridge \
  .venv/bin/python -m unittest test_vla_http test_continuous_http
```

Tests use synthetic frames, injected model backends, and installer recording
runners; they do not download models or issue flight commands. They cover target
tracking, asynchronous grounding, obstacle rejection, stale results, protocol
correlation, ownership, and deployment guards. The HTTP tests exercise the
compiled bridge against the Python service, including continued updates and
stale/correlation rejection. They do not measure Jetson inference latency or
dispatch through the SDK. Validate the full SDK in HITL separately.

Backend references: [Depth Anything V2 metric outdoor small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf),
[Ollama structured vision output](https://docs.ollama.com/capabilities/structured-outputs),
and [OpenCV tracking APIs](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html).
