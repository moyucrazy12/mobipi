# RB-Y1 microwave source demonstrations

Run commands from the repository root:

## Run one source demo

```bash
python -m mobipi.collect_demos.auto_close_microwave_door \
    --layout 1 --style 1 --seed 1 --initial-door-angle-deg 60 \
     --output-dir logs/rby1_source/single60 \
    --record-dir logs/rby1_source/single60/raw
```

Omit `--headless` for the live simulator. The sequence is approach, sustained
active-hand contact, hinge-following close, stable-close detection, unload,
retreat, base-relative stow, and a short hold. 

Approach and stow use reachable FK paths
but still executes and records normal Cartesian actions, without teleporting the
robot or overriding the low-level IK action stream.

The fixed `RBY1Mount` base is placed using the selected microwave's stationary fixture frame and
RoboCasa's placement helper, with an RB-Y1 setback. It does not face the moving
door or move when the starting door angle changes. Optional `--base-local X Y
YAW_DEG` sets an explicit microwave-relative base pose; the floor height is kept.
`--spawn-offset-x/y` adjust the nominal placement. Do not force a world-X push.

For layouts 2, 4, 5, and 7,move the base 7 cm closer by changing
`--spawn-offset-y` from `-0.35` to `-0.28`. Use `--pre-lift-deg 30` for a small
shoulder lift before the normal approach, and reuse that clearance waypoint on
the return to stow. This raises the gripper about 10 cm in this scene. The
waypoint uses the existing FK path and Cartesian commands. Moving closer
without a clearance waypoint can make the lift or stow sweep through the
stove/counter.

```bash
python -m mobipi.collect_demos.auto_close_microwave_door \
    --layout 4 --style 1 --seed 1 --initial-door-angle-deg 60 \
    --spawn-offset-y -0.28 --push-radius-fraction 0.5 \
    --pre-lift-deg 30 --direct-approach \
    --output-dir logs/rby1_source/layout4_direct60 \
    --record-dir logs/rby1_source/layout4_direct60/raw
```

`--direct-approach` removes the intermediate alignment stop. After the small
clearance lift, the TCP follows a straight reach to the pre-contact point while
the gripper aligns, then advances gently until sustained panel contact.
The return to stow uses the raised clearance posture.

The angle-grid collector also accepts these motion/placement flags.
`--pre-lift-deg` sets the shoulder angle between 0 and 90 degrees. The original
`--lift-before-turn` flag still selects a 90-degree lift; an explicit
`--pre-lift-deg` overrides it. 

The optional angle is applied inside environment reset, before the recording
wrapper caches the initial state. The microwave state reader preserves the
negative-opening hinge convention: `hinge_qpos = -float(sim.data.get_joint_qpos(joint_name))`.

## Validate the four starting angles

```bash
python -m mobipi.collect_demos.collect_close_door_demos \
    --layouts 1 --styles 1 --angles 20 40 60 80 --seed 1 --headless \
    --output logs/rby1_source/four_angles
```

Failure reasons distinguish `unreachable_contact`, `collision`, `lost_contact`,
`stalled_closing`, `rebound`, `failed_release`, `failed_stow`, and setup or
interruption failures. A sweep also reports timeout or inconsistent base poses.
Use the logged seed and command to reproduce a failed case separately.

## Collect a balanced dataset

The collector chooses these fixture-side Y offsets automatically:

| Layouts | Y offset |
| --- | --- |
| 2, 4, 5, 7 | -0.28 m |
| 1, 3, 6, 8, 9 | -0.35 m |

An explicit `--spawn-offset-y` overrides the rule for every selected layout.
The command below uses the automatic rule, so omit the old global offset flag.

```bash
mkdir -p logs/rby1_source
demo_run="logs/rby1_source/microwave_balanced_$(date +%Y%m%d_%H%M%S)"

nohup python -u -m mobipi.collect_demos.collect_close_door_demos \
    --count 300 --max-attempts 1500 \
    --layouts 1 2 3 4 7 \
    --styles 0 1 2 3 4 5 6 7 8 9 10 11 \
    --angles 20 30 40 50 60 70 80 \
    --push-radius-fraction 0.5 --pre-lift-deg 30 --direct-approach \
    --headless --output "$demo_run" \
    > "${demo_run}.log" 2>&1 < /dev/null &

echo "PID: $! | Results: $demo_run | Log: ${demo_run}.log"
```

`--count 300` targets 300 complete successes, reserving 60 per selected layout.
Attempts interleave layouts, explore new layout/style/angle combinations, and
favor underrepresented successful styles and angles. Style/angle balance is
best effort: an unreachable condition cannot supply a valid demonstration.
`stats.json` reports the actual saved distribution, layout shortfalls, and cases
that exhausted their retry limit. The run does not fill one layout's quota with
extra demonstrations from an easier layout.

`collection_config.json` records the Python executable, motion arguments,
placement rules, sampler seed, and source hashes. `--dry-run` prints the settings
without creating files or launching simulations. This collector records source
demos; it does not launch MimicGen, render a training image dataset, or train.

To export only successful recordings to a small source HDF5 (states/actions,
without running MimicGen or collecting RGB observations):

```bash
python mobipi/collect_demos/gather_demos_hdf5.py \
    --input logs/rby1_source/four_angles \
    --output logs/rby1_source/four_angle_sources.hdf5 \
    --environment CloseSingleDoor
```

The 31-D source action layout is retained. Requested opening angles are captured
in the raw initial states and retained as per-demo metadata in the HDF5.

# RBY1 Drawer Demo Collection

The full pipeline from these demos to a trained and evaluated policy is in
[docs/rby1_close_drawer_policy.md](../../docs/rby1_close_drawer_policy.md).

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
| `--robot-front-offset` | `-0.18` | robot offset in front of the drawer [m] (RoboCasa: `-0.23`) |
| `--robot-side-gap` | `0.3` | gap between the drawer's right edge and the robot [m] |
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
0.05 m / 5 deg). The left arm works near its reach limit: at RoboCasa's
placement (23 cm in front of the drawer) the end of the push is just within
reach, and varying the motion leaves the drawer a few cm open, so the robot
stands 5 cm closer by default. The pitch and height ranges are narrow for the
same reason: negative pitch, pitch near 10 deg or a push point 3 cm above the
handle make the gripper miss the pre-contact pose and catch on the cabinet trim.

For the old fixed setup, pass `--robot-front-offset=-0.23 --drawer-open 0.9 1.0
--push-edge-inset 0.10 --contact-z-offset 0 --tool-pitch 0 --push-lead 0.03`.

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

## Generate more demos with MimicGen

MimicGen turns the scripted source demos into more demos: it splits each one
at the first robot-drawer contact (approach / push), transforms the segments to
the drawer handle in newly sampled scenes, executes them and keeps the
successful ones. Interface: `MG_RBY1_CloseDrawerLeft`
(`external/mimicgen/.../mg_rby1_close_drawer.py`); config:
`configs/mimicgen/CloseDrawerLeft_RBY1_300mg.json` (300 demos, layouts
0/2/3/5/6, all styles). Run from the repository root:

```bash
# 1. scripted source demos -> source HDF5
python mobipi/collect_demos/collect_close_drawer_demos.py --count 50 --layouts 0 2 3 5 6 \
  --output mobipi/collect_demos/rby1_drawer_demos_50
python mobipi/collect_demos/gather_demos_hdf5.py --input mobipi/collect_demos/rby1_drawer_demos_50 \
  --output datasets/close_drawer_mg/source.hdf5

# (1.5. optional: to check the demos, you can use robomimic's `playback_dataset.py`)
cd external/robomimic
MUJOCO_GL=egl python robomimic/scripts/playback_dataset.py \
  --dataset ../../datasets/close_drawer_mg/source.hdf5 \
  --use-actions --render_image_names robot0_head_camera \
  --video_path ../../datasets/drawer_actions.mp4 --n 5 --video_skip 1
cd ../..

# 2. add MimicGen datagen_info and the approach / push boundary (modifies the file in place)
(cd external/mimicgen/mimicgen/scripts && python prepare_src_dataset_multistyle.py \
  --dataset ../../../../datasets/close_drawer_mg/source.hdf5 \
  --manifest ../../../../mobipi/collect_demos/rby1_drawer_demos_50/manifest.csv \
  --env-interface MG_RBY1_CloseDrawerLeft --contact-signal stage_contact_drawer)

# 3. generate (writes generated_datasets/mimicgen/CloseDrawerLeft_RBY1_300mg/demo.hdf5)
python external/mimicgen/mimicgen/scripts/generate_dataset.py \
  --config configs/mimicgen/CloseDrawerLeft_RBY1_300mg.json
```

`--num_demos`, `--source` and `--folder` override the config. The generated
`demo.hdf5` has the same format as the gathered one (states, 25-D actions,
model_file, ep_meta, env_args; no `actions_abs`, which `dataset_states_to_obs.py`
fills in from the absolute WholeBody IK actions), so render its camera
observations with `dataset_states_to_obs.py` as above. About a third of the
generation attempts succeed (~25 s each), so 300 demos take roughly 6 hours.
