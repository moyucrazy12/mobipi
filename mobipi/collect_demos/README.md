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
