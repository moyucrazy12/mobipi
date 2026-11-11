# RB-Y1 simulation workflows

For existing feature branches and coding agents, see [MIGRATION.md](MIGRATION.md).

`demos/` contains door, drawer and faucet planners and dataset tools; `robot/`
contains shared controllers and simulation/task adapters; `policies/` contains
faucet preparation/evaluation; `pipeline/` connects these to MobiPi. RoboCasa
supplies tasks, MimicGen supplies augmentation, and RoboMimic supplies Cartesian
Diffusion Policy training/inference. Generic scene scoring, base-pose selection
and navigation remain in `mobipi/`.

## Setup and configurations

Follow the root [installation guide](../README.md) for Conda, RoboCasa assets,
MimicGen, RoboMimic and CUDA dependencies. Keep this environment separate from ROS.
From the repository root, with simulation dependencies installed:

```bash
git submodule update --init --recursive
conda activate mobipi
python -m pip install --no-deps -e .
export PYTHONPATH="$PWD:$PWD/external/robomimic:$PWD/external/mimicgen:$PWD/external/robocasa${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export RBY1_EXPERIMENT_ROOT="$PWD/../experiment_results"
export CONTROLLER="$PWD/assets/robosuite/controllers/config/robots/default_rby1_whole_body_ik.json"
export CONFIG_DIR="/path/to/private/team/configs"
```

Obtain experiment JSON files from the team's private OneDrive folder; ask a team
member for access. Keep their contents unchanged and pass local paths explicitly:

| File | Use |
|---|---|
| `faucet_config.json` | Faucet environment, motion, safety limits and MimicGen task specification |
| `faucet_mimicgen_diverse_config.json` | Faucet batch augmentation |
| `faucet_relative_cartesian_dp.json` | Faucet dataset checks and relative pose9 + gripper training |
| `faucet_relative_cartesian_dp_extended.json` | Optional extended faucet training |
| `CloseSingleDoor_RBY1_surface_layout1_115mg.json` | Native door MimicGen generation |
| `CloseDrawerLeft_RBY1_300mg.json` | Native drawer MimicGen generation |

Also obtain the matching door/drawer Diffusion Policy JSON for the selected
absolute or relative Cartesian experiment. Match observation keys, cameras,
controller and action type to the dataset/checkpoint. Override historical
artifact paths explicitly; moving JSON does not convert stored artifacts.

Robot, gripper and controller definitions remain in
[`assets/robosuite/`](../assets/robosuite/README.md) as installation/runtime assets.
The model name `RBY1` does not establish the physical hardware revision. The
working faucet profile uses a fixed mount and floating root. Mobile execution
needs `RBY1MobileBase` and matching assets: install that profile in a separate
environment with `assets/robosuite/install_rby1.sh /path/to/environment`.

## 1. Scripted demonstrations

Use new output directories. Door/drawer use the left arm and require the mobile
model profile (`mobilebase0_center`); faucet uses the right arm and its fixed-mount
profile. Run these task groups in their matching environments. The door planner
opens a MuJoCo viewer and needs a working display.

```bash
python -m rby1.demos.collect_close_door_demos --controller "$CONTROLLER" --count 1 --max-attempts 1 --output data/door_seeds
python -m rby1.demos.collect_close_drawer_demos --controller "$CONTROLLER" --count 1 --max-attempts 1 --output data/drawer_seeds
python -m rby1.demos.gather_demos_hdf5 --input data/drawer_seeds --output data/drawer_source.hdf5
python -m rby1.demos.collect_faucet_demos --config "$CONFIG_DIR/faucet_config.json" --output data/faucet_seed --count 1 --max-attempts 1
```

## 2. MimicGen

Faucet preparation adds task boundaries while preserving executed targets:

```bash
python -m rby1.demos.faucet_pipeline prepare --source data/faucet_seed/seeds.hdf5 --output data/faucet_source.hdf5
python -m rby1.demos.faucet_pipeline generate --config "$CONFIG_DIR/faucet_config.json" --source data/faucet_source.hdf5 --output data/faucet_mg_one --target-successes 1 --max-attempts 1
python -m rby1.demos.faucet_pipeline validate --prepared --source data/faucet_mg_one/generated.hdf5
```

For resumable batches, supply both configs initially; `--resume` reuses settings
stored in the output manifest:

```bash
python -m rby1.pipeline.collect --task TurnOnSinkFaucet --seed-config "$CONFIG_DIR/faucet_config.json" --augmentation-config "$CONFIG_DIR/faucet_mimicgen_diverse_config.json" --seeds 10 --augment 100 --output data/faucet_batch
```

Native drawer multistyle preparation modifies its input. Work on a copy:

```bash
cp data/drawer_source.hdf5 data/drawer_prepared.hdf5
python -m mimicgen.scripts.prepare_src_dataset_multistyle --dataset data/drawer_prepared.hdf5 --manifest data/drawer_seeds/manifest.csv --env-interface MG_RBY1_CloseDrawerLeft --contact-signal stage_contact_drawer
python -m mimicgen.scripts.generate_dataset --config "$CONFIG_DIR/CloseDrawerLeft_RBY1_300mg.json" --source data/drawer_prepared.hdf5 --folder data/drawer_mg --num_demos 1
```

Door uses `MG_RBY1_CloseSingleDoor`, `stage_contact_door`, and the door config.
Use `--help` for layout, style and collection options.

## 3. Training datasets

Faucet preparation needs the original 100-episode, multi-source collection:

```bash
python -m rby1.policies.prepare_faucet_dp --source "$FAUCET_GENERATED" --output data/faucet_policy.hdf5 --config "$CONFIG_DIR/faucet_relative_cartesian_dp.json"
```

For door/drawer, render observations, then add Cartesian commands and train/valid
masks. The rendered file is written beside the input dataset:

```bash
python -m robomimic.scripts.dataset_states_to_obs --dataset "$DRAWER_GENERATED" --output_name drawer_im128.hdf5 --camera_names robot0_head_camera robot0_left_eye_in_hand --camera_height 128 --camera_width 128 --done_mode 2 --num_procs 1
python -m robomimic.scripts.prepare_rby1_dp_dataset --dataset "$DRAWER_RENDERED" --output data/drawer_policy.hdf5 --valid-ratio 0.1
```

## 4. Diffusion Policy training

Use matching external configs. Native RoboMimic retains the absolute/relative
Cartesian algorithms, normalization, gripper prediction and EMA:

```bash
python -m robomimic.scripts.train_rby1 --config "$CONFIG_DIR/faucet_relative_cartesian_dp.json" --dataset data/faucet_policy.hdf5
python -m robomimic.scripts.train_rby1 --config "$DOOR_DP_CONFIG" --dataset "$DOOR_POLICY_DATASET"
python -m robomimic.scripts.train --config "$DRAWER_DP_CONFIG" --dataset data/drawer_policy.hdf5
```

## 5. Checkpoint evaluation

Set artifact variables to matching files. Check inference, then run a bounded
faucet rollout in a new output directory:

```bash
python -m rby1.policies.evaluate_faucet_dp --checkpoint "$CHECKPOINT" --dataset "$DATASET" --check-only --device cpu --output /tmp/faucet_check
python -m rby1.policies.evaluate_faucet_dp --checkpoint "$CHECKPOINT" --dataset "$DATASET" --task-config "$CONFIG_DIR/faucet_config.json" --seeds 0 --max-steps 3 --device cpu --output /tmp/faucet_rollout
python -m robomimic.scripts.evaluate_rby1_rollouts --checkpoint "$DRAWER_CHECKPOINT" --episodes 1 --horizon 3 --output-dir /tmp/drawer_rollout
```

See [native Cartesian policy options](../external/robomimic/docs/rby1_cartesian_diffusion_policy.md)
for door/drawer evaluation.

## 6. Complete MobiPi pipeline

In the mobile-model environment, supply matching policy and scene artifacts:

```bash
python -m rby1.pipeline.run --task close_drawer --checkpoint "$CHECKPOINT" --dataset "$DATASET" --scene-dir "$SCENE_DIR" --skip-detection
```

`SCENE_DIR` needs `pc.ply` and its trained 3DGS model. Scene scoring requires CUDA.
Mobilization supports `close_drawer` and `close_microwave`. Faucet supports
collection, augmentation, training and evaluation. The fixed-mount faucet
environment cannot validate mobile navigation.
`python -m rby1.pipeline.evaluate --help` lists batch evaluation options.

Essential dataset/control guards:

```bash
RBY1_TEST_CONFIG="$CONFIG_DIR/faucet_config.json" python -m unittest discover -s tests
```

Without `RBY1_TEST_CONFIG`, the two simulator checks are skipped; dataset and
termination guards still run.
