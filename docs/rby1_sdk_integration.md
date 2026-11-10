# RB-Y1 SDK backend: follower co-simulation

[`run_rby1_pipeline.py`](../mobipi/scripts/run_rby1_pipeline.py) can drive the
RB-Y1 through the Rainbow Robotics **rby1-sdk** (`--backend sdk`) instead of
moving the RoboCasa robot directly. All robot motion (navigation, task
posture, policy, arm return) goes through SDK calls, which are the same for the
official `rby1-sim` simulator and the real robot.

The SDK has no cameras and `rby1-sim` contains only the robot on a floor, so a
RoboCasa env runs as a **follower**. Every control tick it copies the SDK
robot's odometry and joint positions, steps its physics (the follower arm
pushes the microwave door), renders the policy's head and wrist cameras, checks
environment contacts and reports task progress. Base-pose selection
(3DGS/DINO/BO) and path planning are unchanged.

```text
 run_rby1_pipeline.py --backend sdk
   select_base_pose (3DGS/DINO/BO, unchanged)
   move_to_pose ──► SDKBaseDriver ──── SE(2) velocity ───► rby1-sim / robot (gRPC :50051)
   policy ───────► SDKPolicyEnv ────── left arm joints ──►        │
   arm return ───► SDKSession ──────── joint move ───────►        │
                        ▲                                         │
                        └── Follower (RoboCasa) ◄── odometry, joints
                            cameras, contacts, door progress
```

The coupling is one-way: contacts in the follower don't push back on the SDK
robot, so this tests frames, timing, streaming and the controller path, not
contact forces.

## 1. Reproducing the setup

**No changes were made to the `rby1-sdk` or `rby1-ros2` repositories.** The
SDK is the PyPI package, and the simulator is Rainbow's Docker image. ROS is not
used. Everything else is in this repository.

Starting from the `mobipi` conda env described in the README:

```bash
conda activate mobipi

# 1. SDK client. pip pulls numpy 2.x, which the env can't use, so pin it back.
#    rby1-sdk works with numpy 1.23 even though `pip check` complains.
pip install rby1-sdk==0.10.0
pip install numpy==1.23.3

# 2. RoboSuite RB-Y1 model files must match this repository (RBY1MobileBase).
#    The installer keeps a .pre-rby1 backup of each patched RoboSuite file.
./assets/robosuite/install_rby1.sh ~/anaconda3/envs/mobipi

# 3. gsplat with prebuilt CUDA kernels. The source package compiles them on
#    first use and needs nvcc; without a CUDA toolkit, rendering fails with
#    "'NoneType' object has no attribute 'CameraModelType'".
pip install --no-deps --force-reinstall "gsplat==1.4.0+pt22cu121" \
    --index-url https://docs.gsplat.studio/whl/pt22cu121

# 4. A 3DGS scene: choose close_single_door (1), scene 1, confirm.
#    It extracts to SCENE_MODEL_ROOT_DIR (macros/scene_data/ with our macros_private.py).
python mobipi/scripts/download_scene_models.py

# 5. The simulator (A v1.2 matches rby1a_1.2.xml in RoboSuite). Without --rm the
#    controller log survives a crash: docker logs rby1-sim
xhost +local:docker
docker run -d --name rby1-sim --net=host -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw rainbowroboticsofficial/rby1-sim:0.10.6-a_v1.2
```

Versions used: rby1-sdk 0.10.0 (client) against the 0.9.1 server in
`rby1-sim:0.10.6-a_v1.2` (image `356e077d963d`), torch 2.2.0+cu121,
robosuite 1.5.0, gsplat 1.4.0+pt22cu121.

## 2. Running

Check the connection and frame conventions first (moves the simulated base a
little):

```bash
python -m mobipi.scripts.rby1_sdk_checks --checkpoint ckpts/model_115_cartesian.pth
```

It prints PASS/FAIL for:
- connect: 24 joints read;
- tcp: SDK FK against the RoboSuite model at random joint vectors, < 1 mm / 0.5°;
- odometry: a turn in place, checking that odometry tracks the base link.

Full pipeline:

```bash
python -m mobipi.scripts.run_rby1_pipeline --task close_microwave \
    --checkpoint ckpts/model_115_cartesian.pth \
    --dataset data/mimicgen/CloseDoorLeft_RBY1_115mg/demo_im128_head_wrist_clean37.hdf5 \
    --scene-dir scene_data/close_single_door/layout1_style1 \
    --layout-id 1 --style-id 1 --seed 1 --skip-detection \
    --backend sdk --base-deadband sim
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--backend {sim,sdk}` | `sim` | `sim` is the unchanged RoboCasa pipeline |
| `--robot-address` | `localhost:50051` | SDK address (simulator or robot) |
| `--base-deadband {none,sim}` | `none` | Wheel deadband the base driver compensates. Use `sim` with `rby1-sim`, **never on the robot** ([4.2](#42-base)). |
| `--arm-control {joint,cartesian}` | `joint` | How the policy's TCP targets reach the SDK ([4.3](#43-policy)) |

`result.json` gets an `sdk` block:
- `control_ticks`, `tick_overruns`;
- `follower_max_joint_error_rad`;
- `clipped_joint_commands` and `clipped_joints` (targets clipped to SDK limits);
- `arm_reference_resyncs` and `max_arm_reference_error_rad`.

## 3. Code changes

| File | Change |
| --- | --- |
| [`mobipi/utils/rby1_sdk_utils.py`](../mobipi/utils/rby1_sdk_utils.py) (new) | `SDKRobot`: connection, state, odometry, FK, command stream, joint moves, clipping, stop. `Follower`: the RoboCasa mirror. `SDKBaseDriver`: `_BaseDriver` on SDK odometry and SE(2) commands. `SDKPolicyEnv`/`SDKCartesianAdapter`: stand-ins for robomimic's env and adapter in `execute_loop`. `SDKSession`: attach, policy interfaces, arm return, statistics. |
| [`mobipi/scripts/rby1_sdk_checks.py`](../mobipi/scripts/rby1_sdk_checks.py) (new) | Connection, TCP-frame and odometry checks |
| [`mobipi/scripts/run_rby1_pipeline.py`](../mobipi/scripts/run_rby1_pipeline.py) | The options above. `run_policy`, `navigate_to_pose` and the arm return take an optional `SDKSession`; with the SDK backend the pipeline aligns the SDK robot with the follower after the start pose is sampled. |
| [`mobipi/utils/rby1_utils.py`](../mobipi/utils/rby1_utils.py) | `move_to_pose(driver_factory=…)` swaps in another base driver; `_BaseDriver.hold()` |

With `--backend sim` the pipeline runs exactly as before.

## 4. How it works

### 4.1 Frames

Fitted by comparing SDK FK with the RoboSuite model over random joint vectors
(residual < 0.001 mm / 0.0001°, so torso and arm joint axes and signs match):

```text
T_rsbase_tcp = Trans(0, 0, 0.003) · T_sdk(base → ee_left) · Trans(0, 0, −0.124) · Rz(−90°)
```

- The RoboSuite base origin (`mobilebase0_center`) is 3 mm above the SDK
  `base` link.
- The policy TCP (`grip_site`) is 0.124 m along `ee_left`'s z axis, rotated
  −90° about it.
- Odometry tracks the `base` link, which is the point `get_base_vec` reports.
  The robot turns about the wheel axle 0.228 m ahead of it, the same yaw anchor
  as RoboSuite's virtual base.
- After the start pose is sampled, `reset_odometry` makes odometry read the
  follower's world pose.

The constants are `T_RSBASE_SDKBASE` and `T_EE_LEFT_TCP`; `rby1_sdk_checks`
re-verifies them. The policy's low-dimensional observations come from SDK FK,
as they would on the robot. Images come from the follower.

### 4.2 Base

`SDKBaseDriver` subclasses the existing closed-loop unicycle driver. It reads
the pose from odometry and sends `SE2VelocityCommandBuilder` commands on a
stream (0.3 s hold time). The follower is synced kinematically while driving
(about 1 ms per tick), and its contacts feed the existing collision checks.

`rby1-sim`'s wheels reach `0.55 · (command − 0.24 rad/s)` per wheel. Without
compensation the driver crawls near its goals and reports stalls.
`--base-deadband sim` adds the deadband to each nonzero wheel speed. The 0.55
gain is deliberately not inverted: the SDK ramps turn rates at about 1 rad/s²,
and doubled commands overshoot the heading by about 0.4 rad. On the robot,
measure the response first; the `sim` profile would overspeed the base.

### 4.3 Policy

**`--arm-control joint` (default).** `SDKPolicyEnv` reproduces the training
controller as a reference arm `q_ref`:

1. Solve the WholeBody IK step that RoboSuite would take for the policy's TCP
   target. It runs on a scratch copy of the follower state with the arm at
   `q_ref`.
2. Move `q_ref` by 22 % of that step: `q_ref ← q_ref + 0.22 (q_ik − q_ref)`.
   RoboSuite's joint servos complete exactly this fraction of each IK step per
   control step (fit R² = 1.00 for all left-arm joints, constant
   `TRAINING_SERVO_RESPONSE`).
3. Clip `q_ref` to the SDK joint limits and stream it with
   `JointPositionCommandBuilder`. This uses the URDF velocity limits and 30× the
   URDF acceleration limits, as in Rainbow's `35_leader_arm_teleop_with_monitor.py`.
   The default limits make each streamed target start too slowly.
4. If the measured arm falls more than 0.1 rad behind, `q_ref` restarts from
   the measured joints (`arm_reference_resyncs`).

Replaying recorded sim actions open loop, `q_ref` matches the sim joints within
0.006 rad and the SDK arm trails it by less than 0.02 rad. The reference needs
only the IK model, so the same code runs on the robot without a follower.

**`--arm-control cartesian`** streams the target to the SDK Cartesian
controller instead. It faults with "Singularity" at the training start
posture (left arm straight down), so it isn't usable for these checkpoints
without a different start posture.

### 4.4 Posture and arm return

Before navigating, `SDKSession.attach` sets odometry to the follower's start
pose and moves the SDK torso, arms and head to the follower's posture with one
joint move. The arm return is a single SDK joint move of the left arm. The
follower keeps checking that the door stays closed and that nothing collides,
with the same result fields as the sim version.

### 4.5 Follower

Each tick the follower does three things:
- teleports its base to the SDK pose;
- gives the SDK torso, head and right-arm joints to its joint-position servos;
- overrides the WholeBody IK so that it returns the SDK left-arm joints (the
  trick `return_arm_to_start` uses).

Its arm therefore tracks the SDK arm with the training servos and pushes the
door through MuJoCo contact. A physics step takes about 40 ms, so a policy step
takes about 80 ms of wall-clock time (the sim-only pipeline takes about 65 ms).
The reference arm advances per step, so the motion the policy sees doesn't
depend on this.

## 5. SDK and simulator behavior found on the way

| Behavior | Handling |
| --- | --- |
| The Cartesian controller faults ("Singularity") with a straight arm. The control manager then stays in `MinorFault` and drops every command; the client gets no error. | Joint control by default. A stream that ends unexpectedly checks the control manager and raises. |
| A joint command with one target outside the SDK limits is rejected as a whole, silently (visible only in `docker logs`). | Targets are clipped (`clipped_joints`). |
| The RoboSuite RB-Y1 joint ranges differ from the SDK URDF for most joints, e.g. `left_arm_4` ±6.283 vs ±3.142, `left_arm_5` ≤ 2.007 vs ≤ 1.920, `left_arm_1` ≥ −0.05 vs ≥ −0.017, `left_arm_6` ±2.967 vs ±2.705, `torso_0` ±0.349 vs ±0.262. | See [section 6](#6-results): the current policy depends on the extra range. |
| Default joint acceleration limits make streamed targets start slowly (~10 % of each step per period). | Raised limits ([4.3](#43-policy)) |
| `rby1-sim` wheels respond at `0.55 · (command − 0.24 rad/s)`, and turn rates are ramped at ~1 rad/s². | `--base-deadband sim` ([4.2](#42-base)) |
| About 1 in 1000 `GetRobotState` calls fails in `rby1-sim` (protobuf "invalid UTF-8" in the self-collision link names). | State reads are retried. |
| The SDK 0.10.0 client works with the 0.9.1 server in the simulator image. | — |

## 6. Results

`close_microwave`, layout 1 / style 1, scene `close_single_door/layout1_style1`,
checkpoint `model_115_cartesian.pth`, seed 1, `--skip-detection`. The dataset
argument was `datasets/demo_im128_head_wrist.hdf5`, the RGB dataset available
on this machine for DINO scoring.

| Stage | `--backend sim` | `--backend sdk --base-deadband sim` |
| --- | --- | --- |
| Mobilization | pose [0.829, −1.439, 1.613], score 0.775 | identical |
| Navigation | 1.9 cm / 0.008 rad, 180 steps, no contacts | 1.7 cm / 0.007 rad, 321 steps, no contacts |
| Policy | door closed at step 1305 | timeout after 2000 steps, door untouched |
| Arm return | 0.0004 rad error, door stayed closed | not reached |

Mobilization and navigation through the SDK behave like the sim. The policy
doesn't, and the reason is the joint limits rather than the SDK path:

- In the SDK run, targets were clipped to the SDK limits in 1598 of 2000 policy
  steps (`left_arm_5` 1591, `left_arm_4` 1582, `left_arm_1` 7). The reference arm
  never had to re-sync (`max_arm_reference_error_rad` 0.029), so the SDK
  followed every target it was allowed to.
- The **successful sim rollout leaves the SDK limits too**: `left_arm_4`
  reaches 3.95 rad (SDK limit 3.142) for 430 steps, and `left_arm_5` sits at
  RoboSuite's 2.007 limit (SDK 1.920) for 287 steps. The policy's way of
  closing the door relies on a wrist range the real RB-Y1 doesn't have, so it
  can't be reproduced on the real controller either.

## 7. Open items

- **Joint limits (blocking for the robot):** give the RoboSuite RB-Y1 model the
  SDK URDF limits (`rby1a_1.2.xml`), then re-collect demonstrations and retrain.
  Until then, policies can learn motions the robot can't execute. The SDK
  backend's `clipped_joints` count is a quick check for new checkpoints.
- Run more seeds in both backends once a policy within the SDK limits exists.
- On the robot:
  - measure the base response (no `--base-deadband sim`);
  - localize the robot in the 3DGS frame instead of resetting odometry;
  - replace the follower's cameras and task progress with real cameras and a
    success check.
