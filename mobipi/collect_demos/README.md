# RBY1 Microwave Door Demo Collection

## Microwave fix

In `microwave.py`, replace the old direct `qpos` indexing:

```python
hinge_qpos = sim.data.qpos[sim.model.joint_name2id(f"{self.name}_microjoint")]
```

with:

```python
joint_name = f"{self.name}_microjoint"
hinge_qpos = float(sim.data.get_joint_qpos(joint_name))
```

This is needed because a MuJoCo joint ID is not guaranteed to be the same as its index in the `qpos` vector. Using `get_joint_qpos()` ensures RoboCasa reads the actual microwave hinge position correctly when checking the door state and task success.

## Run one microwave-closing demo

```bash
cd ~/mobipi_EP/mobipi/mobipi/collect_demos

python auto_close_microwave_door.py \
  --environment CloseSingleDoor \
  --layout 1 \
  --style 10 \
  --robot-x 0.8 \
  --robot-y -1.45 \
  --push-x-direction 1 \
  --push-lead 0.03 \
  --keep-open \
  --contact-z-offset 0 \
  --steps-per-waypoint 4 \
  --handle-clearance 0.053 \
  --arm left
```

## Collect multiple successful demonstrations

```bash
cd ~/mobipi_EP/mobipi/mobipi/collect_demos

python collect_close_door_demos.py \
  --count 10 \
  --layout 1 \
  --arm left \
  --robot-x 0.8 \
  --robot-y -1.45 \
  --push-x-direction 1 \
  --push-lead 0.03 \
  --steps-per-waypoint 4 \
  --handle-clearance 0.053 \
  --finish-angle 0.12
```
## RB-Y1 right-arm TurnOnSinkFaucet pipeline

**Status (2026-10-22): implementation and contract tests are available; no
successful faucet demonstration has been verified. Milestone A is blocked by
Cartesian tracking / approach initialization. Do not treat the commands below as
an experimentally validated collection recipe.**

Development branch: `rby1_turn_on_sink_faucet_demos`, based on the remotely
verified default branch `release`, commit `e0ae2b205e0bacabcc508a5af935ca7b845ad6af`.
The isolated checkout is `EngineeringProject/mobipi-faucet`. The original
`mobipi` checkout and its modified RoboMimic dependency were left untouched.
Nothing was committed, pushed, merged, or trained.

### Existing implementation reused

The microwave collector supports `CloseSingleDoor` and records raw robosuite
`DataCollectionWrapper` episodes. Its batch runner counts child-process successes
and varies styles and seeds. It has no general scripted-policy class, HDF5 image
recorder, source preparation workflow, or faucet-specific safety checks. Its
batch runner can remove output directories, so it is not used for faucet output.

The faucet collector imports the existing controller initialization, inactive-arm
hold, full-action construction, TCP-to-reference transform, smoothstep waypoint,
and rotation interpolation helpers from `auto_close_microwave_door.py`. Existing
door scripts are unchanged. A common faucet recorder adds pre-action states,
images, exact executed commands, rewards, dones, diagnostics, and exclusive-create
HDF5 output. Only completed successful rollouts enter episode groups.

MimicGen already supplies `MG_TurnOnSinkFaucet`, sink task specs, source extraction,
source selection, segmentation, and articulated-scene trajectory transforms.
The default branch's newer MimicGen dependency also supplies
`MG_RBY1_CloseSingleDoor` and a multistyle preparation script. Its TCP/wrist
conversion and controller-split conventions were inspected and followed; its
hardcoded left-arm / 31-D composer cannot be used unchanged for this right-arm
24-D controller. Action dimensions are read from live controller splits.
The standard `prepare_src_dataset.py` function is reused, writing a separate file.

### Files and behavior

| File | Purpose |
| --- | --- |
| `collect_faucet_demos.py` | Right-arm scripted phases, shared recorder, success-counted seed collection, per-attempt videos and JSONL logs |
| `faucet_runtime.py` | Task-local robot/gripper registration, fixture geometry, safety checks, addressed IK, `MG_RBY1TurnOnSinkFaucet` |
| `faucet_config.json` | Fixed layout/style/seed, cameras, approach tolerances, three MimicGen subtasks, target/cap settings |
| `faucet_dataset.py` | RoboMimic HDF5 writer and structural / subtask validation |
| `faucet_pipeline.py` | Preparation, validation, command replay, and MimicGen generation CLI |
| `tests/test_faucet_*.py` | Structural fixtures and real MuJoCo interface checks; fixtures do not establish faucet task success |

The task-local environment `TurnOnSinkFaucetRBY1` subclasses the original
`TurnOnSinkFaucet`. It preserves the actual water-on criterion, adds an overview
camera, and corrects `Sink.get_handle_state` to use named joint qpos access.
The installed reader incorrectly indexes qpos by joint ID. Water is on when the
handle angle modulo `2*pi` is strictly between `0.40` and `pi`. No rollout code
writes faucet joint positions, water state, or success flags into the simulator.
The original task's reset still initializes the faucet to off.

Sink assets include vertical side levers and wide top levers with different
hinge axes. The controller reads the live `handle_joint` world axis, anchor,
limits and `handle_main` collision geometry. It approaches with open fingers,
closes on the handle, follows its positive hinge arc toward 0.48 rad, verifies
water-on through the task predicate, releases, and retreats. This is a grasp/arc
strategy, not a fixed-direction activation command. Other faucet geometries
remain unverified; absent landmarks, missing bounds and unsupported joints fail
closed.

The repository gripper adapter is loaded under `RBY1FaucetGripper`, without
installing or changing host packages. The checked repository frames are
`gripper0_right_grip_site` and `gripper0_right_ft_frame`, separated by 0.2501 m.
The locally installed gripper originally used a null adapter with coincident
frames. The task adapter uses the repository XML and verifies that separation.

The installed WholeBody IK also confuses joint IDs, qpos addresses and velocity
addresses. RB-Y1 has a free root, so these differ. A task-local solver binding
retains its damped least-squares / nullspace formulation with the proper
addresses, joint limits and velocity limit. It affects only this environment's
WholeBody IK instance. The original installed solver and other tasks are not
modified. Right-arm TCP targets are converted to absolute world wrist targets;
the inactive arm and gripper are held. Optional
`environment.faucet_initial_arm_qpos` supplies seven reset arm angles, using
named joints; no alternative initial posture has yet passed the rollout gate.

Safety checks cover all right-arm collision geometries descended from the
shoulder, including elbow, forearm, wrist and fingers. Conservative geom AABBs
with an 8 mm margin are checked against oriented basin interiors and the water
site volume, including when water is off. Current state and the next IK joint
configuration are checked before each command, and actual state is checked after
each control step. Kitchen penetration is rejected except finger/handle contact.
Nonfinite actions, invalid joint results, unreachable waypoints and persistent
tracking errors abort the attempt. This is a simulation collection gate, not
continuous-time collision avoidance; it can conservatively reject feasible paths.
Robot self-contacts are excluded from the kitchen-contact gate.

### MimicGen segmentation and action compatibility

Three segments use the **stable sink fixture root frame** as `object_ref=faucet`:

1. Approach and establish physical finger-to-handle contact (`contact` rising edge).
2. Articulate until the task reports water on (`water_on` rising edge).
3. Complete release and retreat (final segment has no termination signal).

The moving handle pose is recorded as diagnostics in `object_poses/handle` but
is not used as a rigid reference for the entire articulation. MimicGen transforms
each segment once from its source fixture frame; the demonstrated arc supplies
the within-segment handle motion. A source demonstration is selected once per
trajectory by default. Contact and activation boundaries must be ordered, with
remaining samples for retreat. Boundaries are measured from physics, not time.

Prepared source episodes contain the installed MimicGen schema:
`datagen_info/{eef_pose,target_pose,gripper_action,object_poses,subtask_term_signals}`,
including the fork's `base_pos` and `base_rot`. Preparation validates array
lengths, finite values, SE(3) poses, ordered boundaries, a stationary fixture
reference, and equality between decoded targets and recorded commanded TCP poses.
Original seeds are preserved.

Generation uses the installed `DataGenerator` for source parsing, selection,
interpolation and object-centric transforms. A context-scoped waypoint executor
builds RB-Y1 actions and applies the same recorder and safety gates. It restores
the original executor afterward, including on failure. This avoids the fork's
hardcoded mobile-base action suffix, left-only composer, zero-noise clipping of
absolute coordinates, and generation script's hardcoded layout override. No
MimicGen checkout is edited. Action noise must currently be zero; nonzero noise
is explicitly rejected. Interpolation, fixed steps and offsets are configurable.

Both source and generated HDF5 files preserve executed `actions`, pre-action
`states`, `obs` (including overview/wrist RGB and right-arm proprioception),
`rewards`, `dones`, exact model XML, episode metadata, environment/controller
configuration, success/origin attributes and diagnostics. Generated episodes
also retain MimicGen metadata and selected source indices. Commanded TCP poses
and achieved TCP poses are distinct arrays; commands are never replaced by
next-state observations. This preserves the information needed for later
absolute or relative Cartesian policy preprocessing. No policy conversion or
training is performed here.

### Commands and milestone gates

Use the existing Python/Conda environment, separate from ROS / `colcon_ws`:

```bash
cd /home/david-isfc/KTH_Univ/DD2414_Engineerins/EngineeringProject/mobipi-faucet
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mobipi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR=/tmp/numba-faucet
export MPLCONFIGDIR=/tmp/mpl-faucet
export MUJOCO_GL=egl
```

The Python environment must already expose robosuite, RoboCasa, RoboMimic and
MimicGen. The separate checkout's submodule directories are not initialized;
these commands use installed dependency paths. Import `faucet_runtime` before
recreating this environment from another program, so the custom task, interface
and gripper are registered. No new dependencies are required.

A. One deterministic trial (currently fails safely), then command replay after
an accepted success:

```bash
python -m mobipi.collect_demos.collect_faucet_demos \
  --output data/faucet_one --count 1 --max-attempts 1
python -m mobipi.collect_demos.faucet_pipeline validate \
  --source data/faucet_one/seeds.hdf5
python -m mobipi.collect_demos.faucet_pipeline replay \
  --source data/faucet_one/seeds.hdf5 --output data/faucet_one/replay.mp4
```

B. After A succeeds and replay passes, collect ten successful seeds. Seed increments
vary supported reset sampling, including distractor identity/placement. The layout
and style stay fixed; this does not establish broad motion diversity.

```bash
python -m mobipi.collect_demos.collect_faucet_demos \
  --output data/faucet_seeds --count 10 --max-attempts 30
python -m mobipi.collect_demos.faucet_pipeline validate \
  --source data/faucet_seeds/seeds.hdf5
```

C. Prepare a new source file, preserving seeds:

```bash
python -m mobipi.collect_demos.faucet_pipeline prepare \
  --source data/faucet_seeds/seeds.hdf5 --output data/faucet_source_prepared.hdf5
python -m mobipi.collect_demos.faucet_pipeline validate --prepared \
  --source data/faucet_source_prepared.hdf5
```

D. Generate one new trajectory with MimicGen, then replay its commands:

```bash
python -m mobipi.collect_demos.faucet_pipeline generate \
  --source data/faucet_source_prepared.hdf5 --output data/faucet_mg_smoke \
  --target-successes 1 --max-attempts 3
python -m mobipi.collect_demos.faucet_pipeline replay \
  --source data/faucet_mg_smoke/generated.hdf5 --output data/faucet_mg_smoke/replay.mp4
```

E. Evaluate ten attempts. Exit status is nonzero if fewer than ten succeed, while
`summary.json` reports attempts, successes, rate, failure reasons and source choices:

```bash
python -m mobipi.collect_demos.faucet_pipeline generate \
  --source data/faucet_source_prepared.hdf5 --output data/faucet_mg_small \
  --target-successes 10 --max-attempts 10
```

For controlled pose variation after deterministic smoke/replay passes, copy the
config and set `generation_environment.randomize_base_init_pose` to a small
value such as `0.01`, then pass `--config your_config.json`. The supported RoboCasa
reset parameter perturbs x, y and yaw. Re-evaluate safety, success and actual
motion diversity before increasing randomization; do not change faucet families
or layouts without new validation. Seeds and effective environment overrides are
saved with every run. Default smoke configuration has no extra pose/noise changes.

F. Only after reliable small-batch results, explicitly run this command to target
**100 successes**, capped at 300 attempts. This job was **not launched**:

```bash
python -m mobipi.collect_demos.faucet_pipeline generate \
  --source data/faucet_source_prepared.hdf5 --output data/faucet_mg_100 \
  --target-successes 100 --max-attempts 300
```

All output directories/files must be new. Attempt caps prevent infinite failure
loops; an incomplete success target returns nonzero. Failed attempts retain
videos/logs but never become demonstration groups. An empty HDF5 from a completely
failed run is rejected as a source. CLI generation defaults to one target / one
attempt; the configuration documents full-generation target/cap values, which
must be requested explicitly with the CLI flags above.

### Validation evidence and remaining work

The existing `mobipi` Conda environment was used with robosuite 1.5.0,
RoboCasa `3683fb0`, installed MimicGen `05c723b`, and the pre-existing RoboMimic
checkout `03b728c`. Its environment wrapper is unchanged relative to the default
branch's pinned RoboMimic `a6706a5`. Tests were also run against the Python modules
at the default branch's pinned MimicGen `e0d55fc`, extracted into a temporary
dependency snapshot without altering the original checkout.

Six short development trials produced zero accepted seeds. The final guarded
baseline failed in `safe_preapproach` with persistent TCP error of approximately
0.077 m. A trial with base placement `[2.75, -1.5, 0]`, yaw `-pi/2`, still failed
at approximately 0.075 m. A seven-joint bent-arm reset
`[0.3, -0.2, 0, -1.2, 0, 0.9, 0]` progressed until finger/countertop penetration
and was rejected. These changes did not produce a working manipulation recipe
and are not enabled in the default config. The baseline and clearer overview
videos / logs / configs / summaries are saved under ignored `data/faucet_debug/`.

The next required experiment is a stable, collision-free initial arm posture and
base/approach placement that tracks the lift and pre-approach waypoints. Do not
relax the safety gate to bypass the tracking or countertop failures. Contact,
handle activation, successful command replay, ten-seed collection, real source
preparation, MimicGen smoke, and small-batch evaluation remain unverified. No
MimicGen attempts were run; generated count is zero and its success rate and
motion diversity are **not measured**.

Run the focused tests:

```bash
python -m unittest discover -s mobipi/collect_demos/tests -v
python -m compileall -q mobipi/collect_demos
```

These checks cover real model frames, action decoding/encoding, addressed IK
hold behavior with a free root, actual faucet-off state, installed metadata and
spec schemas, arm collision coverage, executor restoration, preservation of
absolute coordinates, HDF5 writer keys, nonfinite/misaligned arrays, commanded
versus achieved poses, and articulated-frame / subtask-boundary rejection.
The final focused suite passed all 16 checks against the pinned MimicGen modules;
compilation and whitespace checks passed. Passing these checks does not establish
a successful rollout or validated MimicGen dataset.
