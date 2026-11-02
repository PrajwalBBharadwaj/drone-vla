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
controller checks an inflated vehicle footprint and stopping distance, selects
an observed free heading, turns first, and then approaches at constant altitude.
It stops at the configured stand-off distance.
Stand-off is estimated from depth inside the tracked target box. An occluder or
depth error can cause a premature hold; this is not independent verification that
the intended destination has been reached.

This is a bounded local approach behavior. It does not provide a global map,
long-lived detour waypoints, arbitrary motion sequences, landing, or guaranteed
navigation around an occluding tree. A reactive heading correction can be
insufficient to complete a complicated detour. If no local path passes the
checks, it holds and ends the goal rather than crossing a known obstruction.

Grounding and tracker initialization always return zero axes. Buffered frames
are sampled at 5 Hz during grounding and tracked in bounded batches while new
frames arrive; `acquiring` holds until tracking catches up. Buffer capacity is
tied to the grounding deadline, which defaults to 60 seconds and can be configured
up to 120 seconds. A subsequent fresh frame is required before movement.
Ambiguous targets, lost tracking, invalid depth, processing overruns, interrupted
updates, and goal expiry hold motion. Lost/blocked goals require a new
instruction. Processing overruns can retry on the next fresh frame. Cancelled
grounding jobs cannot produce motion, and another grounding job is rejected
until the earlier worker finishes.

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
  test_bridge_protocol test_continuous_control
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
