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

### Setup

Use the existing `mobipi` Conda environment with robosuite, RoboCasa, RoboMimic
and MimicGen installed. Run all faucet commands from the repository root:

```bash
cd /home/david-isfc/KTH_Univ/DD2414_Engineerins/EngineeringProject/mobipi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mobipi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export NUMBA_CACHE_DIR=/tmp/numba-faucet
export MPLCONFIGDIR=/tmp/mpl-faucet
```

The scripts automatically load the faucet fixes and right-arm configuration from
`mobipi/collect_demos/faucet_config.json`. No dependency source edits are needed.
Use a new output path for each run; existing outputs are not overwritten.

### Run one faucet demonstration

Verified with the default layout 1, style 10 and seed 0:

```bash
python -m mobipi.collect_demos.collect_faucet_demos \
  --output data/faucet_one --count 1 --max-attempts 1
```

### Collect 10 successful seed demonstrations

Collection stops at 10 successes or 30 attempts:

```bash
python -m mobipi.collect_demos.collect_faucet_demos \
  --output data/faucet_seeds --count 10 --max-attempts 30
python -m mobipi.collect_demos.faucet_pipeline validate \
  --source data/faucet_seeds/seeds.hdf5
```

### Prepare and expand with MimicGen

Source preparation and deterministic single-source generation are verified.
The 100-demo generation batch remains unverified. After collecting the seeds:

```bash
python -m mobipi.collect_demos.faucet_pipeline prepare \
  --source data/faucet_seeds/seeds.hdf5 --output data/faucet_source_prepared.hdf5
python -m mobipi.collect_demos.faucet_pipeline validate --prepared \
  --source data/faucet_source_prepared.hdf5
python -m mobipi.collect_demos.faucet_pipeline execute-source \
  --source data/faucet_source_prepared.hdf5 --demo demo_0 \
  --output data/faucet_source_execution
python -m mobipi.collect_demos.faucet_pipeline generate \
  --source data/faucet_source_prepared.hdf5 --output data/faucet_mg_100 \
  --target-successes 100 --max-attempts 300
```

Generation targets 100 successful demonstrations, capped at 300 attempts.
Collection and generation return a nonzero exit status if the target is not met.

### Output files

Seed demonstrations are saved as `seeds.hdf5` inside the requested output directory.
Each collection attempt also saves a video and logs; `summary.json` reports results.
Preparation creates `data/faucet_source_prepared.hdf5` and preserves the seeds.
MimicGen saves accepted episodes to `data/faucet_mg_100/generated.hdf5`, with its
summary, logs and videos of the first two attempts in the same directory.

## Reusable collection pipeline

From the repository root, in the existing `mobipi` environment:

```bash
python -m mobipi.collect_demos.pipeline --task TurnOnSinkFaucet \
  --seeds 10 --augment 100 --output data/faucet_pipeline_run
```

Stages are `collect`, `prepare`, `augment`, `merge`, `validate`, and optional
`convert`. The default runs all except conversion; add `--convert` to include it.
Conversion uses the existing task converter (faucet: relative-DP-compatible 9D
TCP commands plus a scalar gripper channel); it does not start training.
The existing faucet converter currently requires 100 episodes and at least two
source IDs. It does not create a training config or change existing experiments.

Run one stage, then resume with the saved configuration:

```bash
python -m mobipi.collect_demos.pipeline --task TurnOnSinkFaucet \
  --stage collect --seeds 10 --augment 100 --output data/faucet_pipeline_run
python -m mobipi.collect_demos.pipeline --task TurnOnSinkFaucet \
  --stage prepare --output data/faucet_pipeline_run --resume
python -m mobipi.collect_demos.pipeline --task TurnOnSinkFaucet \
  --output data/faucet_pipeline_run --resume
```

Resume skips completed stages and recovers complete episodes from interrupted
attempts. Every new attempt has a separate directory. Default finite attempt
budgets are three times each success target; `--seed-attempts` and
`--augment-attempts` override them and may be increased on resume.
Other saved settings cannot change during resume.

Faucet defaults use a fixed base and a shuffled grid of ±2 cm XY translations
and ±2° yaw. Set `--translation 0.02 --rotation-deg 2` on a new run to override
the bounds (task limit: 3 cm / 2°; 3 cm is not validated).
Use `--seed-config` / `--augmentation-config` for task JSON overrides.
Existing recordings can be supplied read-only with `--seed-dataset`,
`--prepared-dataset`, or `--augmented-dataset`. For example, validate an existing
dataset without collecting or generating:

```bash
python -m mobipi.collect_demos.pipeline --task TurnOnSinkFaucet --stage validate \
  --augmented-dataset data/faucet_mg_diverse_100/generated.hdf5 \
  --augment 100 --output data/faucet_existing_check
```

Outputs: `manifest.json` contains settings, attempt outcomes, seeds and fixture
offsets; `summary.json` and `validation.json` contain results. Recordings are
`seeds.hdf5`, `source.hdf5`, `generated.hdf5`, and optional `policy.hdf5` plus
`conversion.json`. `attempts/{collect,augment}/attempt_*/` preserves videos,
logs, failures and original accepted episodes; `artifacts/` preserves builds.

### Register another task

Add a `Task` entry to `pipeline_tasks.TASKS`, supplying its active arm, default
JSON configs, and `module:function` callbacks. Reuse the task's working planner,
controller, environment adapter, success check and MimicGen subtask config.
Callback signatures match the existing faucet tools:

- `collect(config, output_dir, count, max_attempts)`
- `prepare(source, output_file)`
- `augment(source, output_dir, config, target, max_attempts)`
- `validate(dataset, prepared=False)` → per-episode report list
- `attempt_config(config, stage, index, source, translation, rotation_deg)`
  → `(config_copy, metadata)`; implement task-specific perturbation limits here.
- Optional `convert(source, output_file)` → JSON-serializable report.

Callbacks must save accepted complete episodes in the existing RoboMimic schema;
augmentation must retain source IDs and all manipulation/release phases.
Only TurnOnSinkFaucet is registered initially; CloseSingleDoor can use the same
interface without changes to its current collector.

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
