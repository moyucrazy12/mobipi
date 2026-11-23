# CubeInCup comparison experiments

Use `rby1_mobipi`, the current Python package name. No `rby1` compatibility package
is provided. This guide covers offline training; it does not control WBC or a robot.

| Experiment | Action | Dataset |
|---|---|---|
| A `baseline_7d` | Achieved world XYZ + spatial XYZ Euler delta + width (m) | Original HDF5; already training |
| B `binary_7d` | Same six motion values as A + binary gripper | `binary_7d.hdf5` |
| C `rot6d_10d` | Achieved world XYZ + spatial rotation6D delta + binary gripper | `rot6d_10d.hdf5` |
| D `relative_10d` | Future measured pose relative to one observation anchor + binary gripper | `relative_10d.hdf5` |

D is **future-pose imitation**, not accepted-command imitation. Cached requested
`*_target` fields are not used. At action row i the target is measured pose i+1
(clamped at episode end), matching A's transition alignment. Every D window uses
one anchor at its final observation; native padded windows and decoder are reused.
C uses `R_next = R_delta @ R_current`, with world-axis translation increments.
Rotation6D concatenates the first two rotation columns; it is never six Euler angles.
All poses use the WBC `end_effector_r` site/world frame, not the faucet TCP offset.

## Data and binary gripper

The source has 53 demos, 22,880 samples; the unchanged episode masks contain
47/20,693 training demos/windows and 6/2,187 validation demos/windows. Both RGB
streams (head/right wrist, 224×224) and all five proprioceptive observations remain.
Timestamps remain irregular; conversion does **not** resample or change splits.
Validation contains sessions overlapping training: it is not unseen-session testing.

Training-only width inspection: 14,593 labels ≤5 mm, 5,863 ≥75 mm, 237 in between.
Intermediate runs span at most 7 samples (95th percentile 4). This supports a
40 mm midpoint for this comparison, but does not prove intermediate openings are
unnecessary for task success. Labels are **−1=open, +1=closed**, width ≥40 mm open.
Inference helper `GripperHysteresis` closes at +0.2, opens at −0.2, holds between;
it outputs only 0 or 80 mm. Confirm hardware opening limits before future deployment.

Obtain the prepared files privately from the team's OneDrive; ask a teammate for
access. No public links. Distribute the three HDF5 files and their SHA-256 values
once (including `comparison_datasets.sha256`); teammates do not need to regenerate
them. Check with `sha256sum -c comparison_datasets.sha256` from the dataset folder.
Each file is standalone (~5.9 GB),
with copied images for portability; no fragile external HDF5 links or VDS paths.
Do not commit datasets, checkpoints, evaluation JSON or logs.

## Install

Use a separate Python/Conda environment, not ROS. From the repository root:

```bash
git switch feat/cube-in-cup-experiments
git submodule update --init external/robomimic
conda create -n cube-dp python=3.10
conda activate cube-dp
# Tested: torch 2.7.1 / torchvision 0.22.1. Select CUDA wheels for your machine.
python -m pip install torch==2.7.1 torchvision==0.22.1
python -m pip install -r rby1_mobipi/policies/configs/requirements-cube.txt
python -m pip install --no-deps -e external/robomimic -e .
python -m rby1_mobipi.policies.cube_in_cup_experiments --help
```

The parent Git submodule pin is authoritative; do not switch it to a moving branch.
Required RoboMimic revision: `102c3ae57037879b3b2541097dbda3c0d2c813f7`,
published on the team fork branch `feat/cube-experiment-contracts`.
Its review is stacked on the existing canonical-package integration branch (PR #7).
Dependency changes enable null step caps with constant LR, opt-in partial batches,
and configurable native
relative-policy anchor observation keys/quaternion order. Defaults are unchanged.
For existing environments, preserve the working PyTorch/CUDA installation.
No simulator assets or robot connection are needed for these offline commands.

## Prepare once (dataset maintainer)

```bash
export SOURCE_DATASET=/private/path/cube_in_cup_right_delta_head_wrist.hdf5
export DATASET_DIR=/private/path/cube-comparison
for EXP in binary_7d rot6d_10d relative_10d; do
  python -m rby1_mobipi.policies.cube_in_cup_experiments prepare \
    --experiment "$EXP" --source "$SOURCE_DATASET" --output "$DATASET_DIR/$EXP.hdf5"
done
sha256sum "$DATASET_DIR/"*.hdf5
```

Preparation refuses overwrites, preserves source data, records source checksum and
semantics, and validates splits, observations, binary labels and reconstructed poses.
Conversions are deterministic for a fixed input/software stack; distributed file
checksums identify the exact artifacts, independently of local HDF5 library versions.

## Train manually

Each block requires only dataset/result paths. Result roots must **not exist**;
choose a fresh root for another seed/run. No command runs baseline A.

Teammate 1 — binary gripper:

```bash
export DATASET_DIR=/private/path/cube-comparison
export RESULTS_DIR=/private/path/new-cube-results
python -m rby1_mobipi.policies.cube_in_cup_experiments train \
  --experiment binary_7d --dataset "$DATASET_DIR/binary_7d.hdf5" \
  --output "$RESULTS_DIR/binary_7d"
```

Teammate 2 — achieved rotation6D:

```bash
export DATASET_DIR=/private/path/cube-comparison
export RESULTS_DIR=/private/path/new-cube-results
python -m rby1_mobipi.policies.cube_in_cup_experiments train \
  --experiment rot6d_10d --dataset "$DATASET_DIR/rot6d_10d.hdf5" \
  --output "$RESULTS_DIR/rot6d_10d"
```

Teammate 3 — fixed-anchor future poses:

```bash
export DATASET_DIR=/private/path/cube-comparison
export RESULTS_DIR=/private/path/new-cube-results
python -m rby1_mobipi.policies.cube_in_cup_experiments train \
  --experiment relative_10d --dataset "$DATASET_DIR/relative_10d.hdf5" \
  --output "$RESULTS_DIR/relative_10d"
```

B–D: batch 8, 15 epochs, LR 1e-4, seed 2026, horizons 2/16/8, DDIM 100/10, EMA.
Same UNet/image encoders; only action dimensions differ. Each uses its own training
normalizer. Step caps are null; `drop_last=false` gives 2,587 optimizer updates per
epoch, including the final 5 windows (38,805 total), and 274 validation batches.
A remains unchanged: its historical caps/drop-last omit partial batches.
Eight recorded transitions do not always span 0.8 s; compare all variants with the
same timing limitation. Continuous-time or resampled training is a separate study.

## Monitor and evaluate

```bash
python -m tensorboard.main --logdir "$RESULTS_DIR" --port 6006
find "$RESULTS_DIR" -name 'model_epoch_*.pth' -o -name log.txt
```

Native results live under `OUTPUT/cube_EXPERIMENT/TIMESTAMP/`: `models/`, `logs/`,
`config.json`. Logs/TensorBoard contain training and validation diffusion losses.

```bash
export EXP=relative_10d  # or binary_7d / rot6d_10d
export CHECKPOINT=/private/path/model_epoch_15.pth
python -m rby1_mobipi.policies.cube_in_cup_experiments validate \
  --experiment "$EXP" --dataset "$DATASET_DIR/$EXP.hdf5"
python -m rby1_mobipi.policies.cube_in_cup_experiments evaluate-offline \
  --experiment "$EXP" --dataset "$DATASET_DIR/$EXP.hdf5" \
  --checkpoint "$CHECKPOINT" --output "$RESULTS_DIR/${EXP}_offline.json"
```

Evaluation is CPU-only; default stride 8, optional `--max-windows` for a quick check.
It checks checkpoint semantics, observations, action dimensions, dataset checksum
and training-derived normalization. It reports decoded XYZ/orientation errors by
execution slot, gripper accuracy with hysteresis, transition slots/time differences,
and predicted/demonstrated XYZ traces. Padded future slots are excluded.
Each chunk starts from a demonstrated
observation: these are teacher-forced replans and open-loop chunk comparisons,
**not closed-loop rollouts, task success, or proof of safe robot execution**.
Transition timing uses recorded intervals, with explicit missing-transition cases.
Do not compare normalized diffusion loss directly across representations.

Share Git/submodule SHAs, dataset checksums, saved config, software/GPU details,
checkpoint epoch, validation curves and offline JSON privately using identical
stride/seed settings. Keep unsuccessful/missing gripper transitions in comparisons.
See [README.md](README.md) for the wider MobiPi workflows.
