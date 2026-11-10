# RB-Y1 contributions

Team additions to MobiPi: scripted demonstrations, task adapters, bounded faucet
IK, recording/validation, Cartesian-policy preparation/evaluation, and RB-Y1
mobilization/controller adaptations. The implementation is here; existing
`mobipi.*` imports and script paths remain compatible.

| Directory | Contents |
|---|---|
| `demos/` | Door, drawer and faucet collection; HDF5 recording and MimicGen faucet execution |
| `robot/` | Simulator access/pose helpers, bounded IK, faucet/drawer task variants, base movement/posture |
| `policies/` | Faucet dataset preparation, evaluation and reporting; imports native RoboMimic policy utilities |
| `pipeline/` | Resumable demonstration collection and separate mobilization/task orchestration |
| `configs/` | Original task/policy JSON, including `mimicgen/` door/drawer configs |
| `tests/` | Simulation, dataset, policy, compatibility and numerical regression tests |
| `paths.py` | Repository/config discovery and existing experiment-root behavior |

## Models and tasks

The simulation model is named `RBY1`; tracked assets include
`assets/robosuite/models/assets/robots/rby1a/rby1a_1.2.xml`. This filename does
not confirm the physical robot's hardware revision.

- `CloseSingleDoor`: microwave door, left arm. The alternative
  `auto_close_door_general` planner remains experimental.
- `TurnOnSinkFaucetRBY1`: RoboCasa faucet variant, right arm, overview plus
  right-wrist images; relative pose9 plus one gripper channel.
- `CloseDrawerLeft`: RoboCasa drawer variant, left arm with task-specific
  torso/head posture and head/left-wrist images.

The mobilization registry currently supports `close_microwave` and
`close_drawer`. Faucet supports collection, MimicGen, training and evaluation;
this cleanup does not add a faucet mobilization task or an SDK backend.

## Setup

Use the existing Python/Conda simulation environment, separate from ROS:

```bash
git submodule update --init --recursive
conda activate mobipi
export PYTHONPATH="$PWD:$PWD/external/robomimic:$PWD/external/mimicgen:$PWD/external/robocasa${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export NUMBA_CACHE_DIR=/tmp/rby1-numba
export MPLCONFIGDIR=/tmp/rby1-mpl
export RBY1_EXPERIMENT_ROOT="$PWD/../experiment_results"
```

Run commands from the repository root. RoboCasa meshes/textures must be
downloaded; see the root installation instructions. Robot/gripper/base assets
and RoboSuite registration patches remain under `assets/robosuite/` to preserve
MJCF and installation paths.

The validated faucet environment has an older installed fixed-mount/floating-root
model. Its RoboSuite registry lacks `RBY1MobileBase`, which the mobile pipeline
requires. Prepare the mobile profile in a **separate Conda environment** using
`assets/robosuite/install_rby1.sh /path/to/that/environment`; do not silently
replace the working faucet profile. Mobile execution remains unverified here.
The tested historical model revisions, package versions and file hashes are
recorded in [environment.json](../docs/codebase_cleanup/environment.json).

## Demonstrations and MimicGen

Choose new output directories; the faucet tools refuse to overwrite outputs.

```bash
python -m rby1.demos.collect_close_door_demos --count 1 --max-attempts 1 --output data/door_seeds
python -m rby1.demos.collect_close_drawer_demos --count 1 --max-attempts 1 --output data/drawer_seeds
python -m rby1.demos.gather_demos_hdf5 --input data/drawer_seeds --output data/drawer_source.hdf5

python -m rby1.demos.collect_faucet_demos --output data/faucet_seed --count 1 --max-attempts 1
python -m rby1.demos.faucet_pipeline prepare --source data/faucet_seed/seeds.hdf5 --output data/faucet_source.hdf5
python -m rby1.demos.faucet_pipeline generate --source data/faucet_source.hdf5 --output data/faucet_mg_one --target-successes 1 --max-attempts 1
python -m rby1.demos.faucet_pipeline validate --prepared --source data/faucet_mg_one/generated.hdf5
```

For a resumable faucet collection/augmentation batch:

```bash
python -m rby1.pipeline.collect --task TurnOnSinkFaucet --seeds 10 --augment 100 --output data/faucet_batch
```

Add `--resume` to continue an interrupted batch with the saved settings.
Door/drawer use the native MimicGen interfaces `MG_RBY1_CloseSingleDoor` and
`MG_RBY1_CloseDrawerLeft`. Their configs are in `rby1/configs/mimicgen/`;
historical absolute paths inside them are unchanged. Supply `--source`,
`--folder` and `--num_demos` overrides to native `generate_dataset.py`.
See the existing [drawer procedure](../docs/rby1_close_drawer_policy.md) for
source preparation, segmentation, rendering and policy conversion. Native
multistyle preparation modifies its input: use a separate working copy.

## Policies

Faucet conversion expects the original 100-episode, multi-source dataset; a
one-demo smoke dataset cannot be used to train the historical policy.

```bash
python -m rby1.policies.prepare_faucet_dp --source data/faucet_batch/generated.hdf5 --output data/faucet_policy.hdf5
python -m robomimic.scripts.train_rby1 --config rby1/configs/faucet_relative_cartesian_dp.json --dataset data/faucet_policy.hdf5
python -m rby1.policies.evaluate_faucet_dp --check-only --epochs 9 --device cpu --output /tmp/faucet_checkpoint_check
```

The checkpoint check uses the historical experiment location. For another
experiment, pass `--checkpoint` (or `--run-dir`) and `--dataset` explicitly.
For a bounded fresh rollout, replace `--check-only` with `--epochs 9 --seeds 0
--max-steps 3`; output must be a new directory. Export a report with:

```bash
python -m rby1.policies.report_faucet_evaluation --source /path/to/model_epoch_directory --output /tmp/faucet_report
```

Absolute and relative **door** policies continue to use native
`robomimic.scripts.train_rby1` with configs in
`external/robomimic/robomimic/exps/rby1_close_single_door*.json`.
Drawer policies use native `robomimic.scripts.train` and its drawer configs;
the custom path launcher deliberately handles only door/faucet experiments.
See [native policy documentation](../external/robomimic/docs/rby1_cartesian_diffusion_policy.md).
Algorithms, normalization, window anchors, EMA and Cartesian adapters remain
in RoboMimic for standalone use and checkpoint compatibility.

## Mobilization

In the mobile-model environment, supply actual matching artifacts:

```bash
python -m rby1.pipeline.run --task close_drawer --checkpoint "$CHECKPOINT" --dataset "$DATASET" --scene-dir "$SCENE_DIR" --skip-detection
```

`SCENE_DIR` must contain `pc.ply` and the matching trained 3DGS model. DINO/3DGS
still require CUDA even with CPU policy inference. Existing task defaults can
refer to historical artifact paths, so explicit paths are recommended.
`rby1.pipeline.evaluate` runs bounded seed batches; see
[pipeline documentation](../docs/rby1_pipeline.md). Existing evaluation manifests
fingerprint code: start a new output directory after this refactor.

## Reused frameworks and verification

Generic MobiPi scene rendering, scoring, BO, collision/path planning and policy
utilities stay in `mobipi/`. RoboCasa supplies environments/fixtures; MimicGen
supplies segmentation/transforms/generation; RoboMimic supplies training and
inference. No upstream implementations were copied into this package.

Unavoidable dependency modifications: installed RoboSuite model/base/gripper
registration and base controller adapter; native MimicGen door/drawer interfaces;
native RoboMimic Cartesian/relative dataset/algorithm hooks and inference fixes.
Their locations and history are in the [audit](../docs/codebase_cleanup/audit.md).
This refactor's RoboMimic change is **test-only** (follow legacy redirects during
source inspection); production dependency code is unchanged.

Run the [regression matrix](../docs/codebase_cleanup/regression.md). It distinguishes
numerical equivalence, smoke checks and manual full-pipeline/statistical work.
Historical policy success rates and the faucet camera contract are preserved;
no physical deployment readiness is implied.
