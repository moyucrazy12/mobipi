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
# RBY1 Drawer Demo Collection

`CloseDrawerLeft` (`mobipi/envs`) always places the robot to the right of the
drawer and resets into a squat with the head turned toward it. When a sampled
scene has no room on that side it samples another one; if none of 20 has room,
the demo script skips the scene (exit code 3).

## Run one drawer-closing demo

```bash
cd ~/mobipi_EP/mobipi/mobipi/collect_demos

python auto_close_drawer.py --layout 1 --style 0 --seed 0 --record-dir drawer_raw
```

## Collect multiple successful demonstrations

```bash
cd ~/mobipi_EP/mobipi/mobipi/collect_demos

python collect_close_drawer_demos.py \
  --count 50 \
  --layouts 0 2 3 5 6 \
  --output rby1_drawer_demos_50
```

Successful demos are saved as `demo_NN_layout_LL_style_SS_seed_XXXX/ep_*` and
listed in `manifest.csv`; `attempts.csv` lists every attempt with its outcome
(success / failed / skipped) and its sampled motion parameters.

The drawer opening and the scripted arm motion are randomized by default, so the
demos are not all the same trajectory:

| Option | Default | Meaning |
|---|---|---|
| `--base-pos-noise` | `0` | std of the base x / y placement noise [m] |
| `--base-yaw-noise` | `0` | std of the base yaw placement noise [deg] |
| `--drawer-open MIN MAX` | `0.5 1.0` | initial drawer opening (1.0 = RoboCasa's fully open) |
| `--push-edge-inset` | `0.06 0.14` | push point distance from the drawer's robot-side edge [m] |
| `--contact-z-offset` | `-0.03 0.01` | push point height relative to the handle [m] |
| `--tool-pitch` | `0 6` | gripper pitch below horizontal [deg] |
| `--push-lead` | `0.02 0.04` | how far ahead of the gripper the push target is [m] |

The motion options take one value (fixed) or `MIN MAX` (sampled uniformly per
attempt). Base placement noise is off by default: a few cm / deg already puts
the end of the push out of the left arm's reach (most attempts fail with
0.05 m / 5 deg). The left arm already works near its reach limit, which is also
why the pitch and height ranges are narrow: negative pitch, pitch near 10 deg or
a push point 3 cm above the handle make the gripper miss the pre-contact pose
and catch on the cabinet trim. Pass `--drawer-open 0.9 1.0 --push-edge-inset 0.10
--contact-z-offset 0 --tool-pitch 0 --push-lead 0.03` for the old fixed setup.

## Convert the demos to a robomimic HDF5

MimicGen, robomimic's `dataset_states_to_obs.py` and the RB-Y1 diffusion policy
pipeline read one robomimic HDF5 instead of the raw `ep_*` folders:

```bash
python gather_demos_hdf5.py \
  --input rby1_drawer_demos_10 \
  --output ../../datasets/close_drawer/demo.hdf5
```

It takes the successful demos from each input's `manifest.csv` and writes
`states`, `actions` and `actions_abs` (the same absolute commands) per demo, with
`model_file` / `ep_meta`, and `env_args` that recreate the collection
environment (from the `env_kwargs.json` saved with each recording, so the base
noise and drawer opening range carry over; all gathered demos must share them). Render image observations from it with robomimic's
`dataset_states_to_obs.py`, e.g. `--camera_names robot0_head_camera
robot0_left_eye_in_hand --camera_height 128 --camera_width 128`.
