# Mobi-π to RB-Y1 utility map

## Target pipeline

```text
RB-Y1 observations -> Mobi-π base-pose score -> navigate to pose
                    -> RB-Y1 manipulation policy -> task success
```

RoboCasa is the first target. ROS 2 and hardware execution come after the simulated loop works.

## Utility mapping

| Utility | Mobi-π currently expects | RB-Y1 equivalent / required change |
|---|---|---|
| Robot model | A RoboSuite mobile manipulator with known body, arm, gripper, camera, and controller names. | Register the selected RB-Y1 MJCF in RoboSuite. Define base, torso, arm, gripper, end-effector sites, collision geoms, cameras, default pose, and controller configuration. Confirm A/M/UB and version first. |
| Environment | RoboCasa reset, robot placement, fixture map, object lookup, contacts, and task success. | Make the selected RoboCasa tasks accept RB-Y1. Start with one arm and one task; verify scripted success before learning. |
| DINO view score | Training images and rendered candidate images for each policy observation key. Images are RGB tensors, preprocessed by `timm`; masked scoring is only checked at 224×224. Mobi-π normally scores `robot0_agentview_right`, plus the left view for `CloseDrawer`. | Map physical cameras to semantic roles instead of old names. Resize/normalize consistently, preserve RGB order, and provide intrinsics plus camera-to-base transforms. Only score cameras present in both demonstrations and candidate rendering. |
| RB-Y1 cameras | Fixed simulated cameras attached to the old robot, with exact MuJoCo calibration and segmentation. Navigation separately uses `navview`/`fisheye`. | The workspace does not currently define RB-Y1 camera topics or MJCF cameras, so the reported four-camera setup is not yet verifiable here. Treat them as `camera_1..4` until their topics, frames, intrinsics, distortion, and roles are confirmed. We probably use one or two policy cameras for DINO and reserve wide/front cameras for navigation. DINO does not require all four. |
| 3D scene model | A Gaussian Splat model renders the policy camera view at any candidate `(x, y, yaw)`. | Retain the renderer. Supply RB-Y1 camera intrinsics/extrinsics and robot masks, then regenerate scene data for the selected RoboCasa scenes. |
| Manipulation policy | RoboMimic observation keys, action shape, controller semantics, and normalization from the original robot dataset. | Collect RB-Y1 demonstrations and train a new policy. Define whether actions are Cartesian deltas or joint commands, which torso/arm joints participate, gripper convention, control rate, and camera keys. Existing checkpoints are not directly compatible. |
| Base-pose optimizer | Candidate SE(2) poses scored by training-view similarity, object visibility, and point-cloud vacancy. | Keep Bayesian optimization. Add RB-Y1 footprint, camera geometry, and arm reachability/IK so a visually good pose is also executable. |
| Collision utility | Circular footprint from `horizontal_radius` plus point-cloud/floor checks. | Use a validated RB-Y1 footprint and whole-body checks near counters. SDK dynamics/state can provide link distances on hardware; MuJoCo supplies contacts in RoboCasa. |
| Navigation | Hard-coded `mobilebase0_base`, workspace bounds, gains, and a 12-D action with base commands in `action[7:10]`. | Replace with `get_base_pose()` and `command_base_velocity(vx, vy, wz)`. RB-Y1 M supports lateral motion; A ignores `vy`. Use the internal controller in simulation and Nav2 for hardware. |
| Arm/torso control | Original RoboSuite controller and action indexing. | Mirror RB-Y1 joint groups in RoboSuite. Hardware can reuse `JointPositionCommandBuilder`, `CartesianCommandBuilder`, and command streams from the SDK, or the equivalent ROS actions. |
| Gripper | Original RoboSuite gripper action and grasp state. | Add the exact RB-Y1 gripper actuator/contact model. Hardware command support still needs confirmation: finger geometry and Dynamixel code exist, but no clear general gripper action was found in the ROS driver. |
| Robot state / FK | MuJoCo joint/base state and controller-specific base-pose access. | Normalize SDK/ROS joint state, odometry, end-effector transforms, collision state, and command feedback. Reuse SDK `get_state()`, state streaming, dynamics FK, and transformation utilities on hardware. |
| Demonstrations | MimicGen/RoboMimic HDF5 containing synchronized images, states, actions, language, and metadata. | Extend RB-Y1 record/replay with timestamps, selected camera frames, joint/base/EEF/gripper state, policy action, and task metadata; validate replay before training. |
| Evaluation | RoboCasa success plus Mobi-π logs and videos. | Preserve task success where robot-independent. Add final base-pose error, reachability rejection, collision, timeout, and manipulation failure reason. |
| ROS bridge | None; Mobi-π directly steps MuJoCo. | Later create a separate `colcon_ws` adapter using `/rby1/cmd_vel`, `/rby1/odom`, TF, joint/Cartesian actions, stream control, camera topics, and stop/cancel. Keep ROS out of the Mobi-π Conda environment. |

## Camera contract for DINO

DINO itself accepts batches shaped `(N, 3, H, W)` after model preprocessing. The Mobi-π scorer adds a view dimension for candidate poses: `(candidate, view, 3, 224, 224)`.

For an RB-Y1 with four cameras, use a configuration like this after the hardware layout is confirmed:

| RB-Y1 camera role | Feed to DINO? | Purpose |
|---|---:|---|
| Primary manipulation view | Yes | Must match the primary image key used to train the manipulation policy. |
| Secondary manipulation view | Optional | Use only if demonstrations, policy, and scene renderer all include it. |
| Wide/front navigation view | Usually no | Navigation or LeLaN input; different distortion/viewpoint would shift DINO similarity. |
| Wrist, rear, or second navigation view | Task-dependent | Use for policy/scoring only if trained as its own observation key. |

The important rule is **camera correspondence**, not camera count. Each DINO comparison must use the same semantic camera, preprocessing, resolution, intrinsics, and approximate robot visibility as its training images.

## Control functions worth reusing

- `initialize_robot()`: connect, power, servo, reset faults, enable control manager. Adapt it to return structured errors instead of exiting.
- `movej()`: grouped torso and arm posture control.
- `robot.model()`: discover model name, DOF, joint names, and group indices.
- `get_state()` / `start_state_update()`: joints, targets, odometry, and collision feedback.
- `SE2VelocityCommandBuilder`: base-frame `(vx, vy, wz)` commands.
- `JointPositionCommandBuilder`: posture, replay, or joint-space policy commands.
- `CartesianCommandBuilder` and Cartesian streams: likely hardware backend for end-effector policies.
- SDK dynamics FK and `compute_transformation()`: EEF observations and reachability validation.
- `cancel_control()` / stream cancellation: common stop path.
- ROS stream manager: stream activation and zero-velocity keepalive.
- Nav2: hardware `navigate_to_pose()` implementation.

## Small adapter boundary

Mobi-π should call a robot-neutral interface:

```python
get_base_pose()
command_base_velocity(vx, vy, wz)
navigate_to_pose(target)
move_to_posture(name)
get_robot_state()
get_camera_image(name)
get_camera_calibration(name)
apply_policy_action(action)
command_gripper(arm, position)
get_collision_state()
stop(reason)
```

Implement it twice:

- `RoboCasaRBY1Control`: MuJoCo/RoboSuite backend for Phase 1.
- `RBY1ROSControl`: ROS 2/SDK backend for later hardware execution.

## Implementation order

1. Confirm RB-Y1 variant, gripper, four camera roles/calibration, first arm, and first task.
2. Initialize the pinned Mobi-π RoboCasa/RoboMimic submodules.
3. Register and control RB-Y1 in RoboSuite; complete one scripted RoboCasa task.
4. Add the adapter and replace hard-coded base/action/camera names.
5. Record demonstrations and train a fixed-base RB-Y1 manipulation policy.
6. Adapt DINO/scene rendering and add reachability to base-pose scoring.
7. Close the simulation loop: select pose -> navigate -> manipulate -> evaluate.
8. Add the separate ROS 2 adapter.
