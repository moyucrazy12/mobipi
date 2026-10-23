# RB-Y1 CloseDrawerLeft: from demos to a trained Diffusion Policy

End-to-end pipeline for the drawer task: scripted demos → (optional) MimicGen →
rendered dataset → Cartesian Diffusion Policy → rollout evaluation.

Run everything in the `mobipi` conda env. Paths are relative to the repository
root unless a step says `cd external/robomimic`.

```text
collect_close_drawer_demos.py      scripted demos (raw ep_* folders + manifest.csv)
        |
gather_demos_hdf5.py               one robomimic HDF5 (states, 25-D actions, scene, env_args)
        |
        +-- MimicGen (optional): prepare_src_dataset_multistyle.py -> generate_dataset.py
        |
dataset_states_to_obs.py           render head + wrist cameras (128x128)
        |
prepare_rby1_dp_dataset.py         9-D actions_eef_cmd + train / valid masks
        |
train.py                           Diffusion Policy (robomimic)
        |
evaluate_rby1_rollouts.py          success rate in simulation
        |
run_rby1_pipeline.py --task close_drawer   mobilization + navigation + policy
```

## The task

`CloseDrawerLeft` (`mobipi/envs/kitchen_drawer.py`) is RoboCasa's `CloseDrawer`
with the drawer always on the robot's left, so the left arm can reach it:

- the robot stands to the right of the drawer, 18 cm in front of it in the
  batch collector (RoboCasa's 23 cm leaves the end of the push at the edge of
  the arm's reach);
- it resets into a squat with the head turned toward the drawer
  (`rby1_utils.TORSO_POSES`, `HEAD_POSES`);
- scenes without room on the right are re-sampled (up to 20 times);
- success: the drawer is at most 5 % open (RoboCasa's check).

The policy controls only the left arm. Its action is the left TCP target in the
robot-base frame (`actions_eef_cmd`, 9-D: position + rotation6d);
`RBY1CartesianAdapter` turns it into the 25-D WholeBody IK action and holds the
right arm, torso and head.

## 1. Collect scripted demos

```bash
python mobipi/collect_demos/collect_close_drawer_demos.py \
  --count 50 --layouts 0 2 3 5 6 \
  --output mobipi/collect_demos/rby1_drawer_demos_50
```

- Layouts 0, 2, 3, 5, 6 are the Mobi-π training kitchens; 1, 4, 7, 8, 9 are
  held out for testing.
- The drawer opening (50–100 %) and the arm motion (push point, height, gripper
  pitch, push lead) are randomized per attempt; see
  `mobipi/collect_demos/README.md` for the options. Robot placement noise is off:
  a few cm already put the end of the push out of reach.
- About 70 % of attempts succeed, ~40 s each (50 demos ≈ 50 min). Output:
  `demo_NN_*/ep_*` per demo, `manifest.csv`, `attempts.csv` (outcome and sampled
  motion parameters of every attempt), `logs/`.

## 2. Gather into one HDF5

```bash
python mobipi/collect_demos/gather_demos_hdf5.py \
  --input mobipi/collect_demos/rby1_drawer_demos_50 \
  --output datasets/close_drawer_mg/source.hdf5
```

Writes `states`, `actions` (25-D absolute WholeBody IK commands), `actions_abs`
(the same), `model_file`, `ep_meta` per demo, and `env_args` from the
`env_kwargs.json` saved with each recording. Batches collected with different
settings cannot be merged.

## 3. (Optional) Generate more demos with MimicGen

MimicGen splits each source demo at the first robot–drawer contact (approach /
push), transforms the segments to the drawer handle in new scenes and keeps the
successful executions. Interface `MG_RBY1_CloseDrawerLeft`, config
`configs/mimicgen/CloseDrawerLeft_RBY1_300mg.json` (300 demos, training layouts,
all styles).

```bash
# keep an untouched copy: the prepare step modifies the source file in place
cp datasets/close_drawer_mg/source.hdf5 datasets/close_drawer_mg/close_drawer_50.hdf5

(cd external/mimicgen/mimicgen/scripts && python prepare_src_dataset_multistyle.py \
  --dataset ../../../../datasets/close_drawer_mg/source.hdf5 \
  --manifest ../../../../mobipi/collect_demos/rby1_drawer_demos_50/manifest.csv \
  --env-interface MG_RBY1_CloseDrawerLeft --contact-signal stage_contact_drawer)

python external/mimicgen/mimicgen/scripts/generate_dataset.py \
  --config configs/mimicgen/CloseDrawerLeft_RBY1_300mg.json
```

Output: `generated_datasets/mimicgen/CloseDrawerLeft_RBY1_300mg/demo.hdf5`
(same format as step 2). The 300-demo run took 826 attempts (36 % success,
~5 h). `--num_demos`, `--source` and `--folder` override the config.

## 4. Render the camera observations

```bash
cd external/robomimic
MUJOCO_GL=egl python robomimic/scripts/dataset_states_to_obs.py \
  --dataset ../../generated_datasets/mimicgen/CloseDrawerLeft_RBY1_300mg/demo.hdf5 \
  --output_name close_drawer_300mg_im128.hdf5 \
  --camera_names robot0_head_camera robot0_left_eye_in_hand \
  --camera_height 128 --camera_width 128 --done_mode 2 --num_procs 6
```

The output is written next to the input. ~20–30 s per demo per worker. For the
scripted demos only, use `--dataset ../../datasets/close_drawer_mg/close_drawer_50.hdf5`.

## 5. Add the policy action and the train / valid split

```bash
python robomimic/scripts/prepare_rby1_dp_dataset.py \
  --dataset ../../generated_datasets/mimicgen/CloseDrawerLeft_RBY1_300mg/close_drawer_300mg_im128.hdf5 \
  --output ../../generated_datasets/mimicgen/CloseDrawerLeft_RBY1_300mg/close_drawer_300mg_im128_cmd.hdf5 \
  --valid-ratio 0.1
```

Adds `actions_eef_cmd` to every demo and `mask/train`, `mask/valid` (seed 2026).
Each command is converted back through `RBY1CartesianAdapter` and must reproduce
the recorded command, so the policy's actions execute exactly as demonstrated.
A `.report.json` next to the output lists per-demo checks and the split.

## 6. Create the training config

Needs the `_cmd` dataset from step 5 (the rendered file from step 4 has no
`actions_eef_cmd` and no train / valid masks; the script stops with a message
if you pass it). Writes `robomimic/exps/<name>.json` from the 50-demo config
with the dataset, the name and the epoch lengths set (one epoch = one pass over
the training sequences at batch size 16):

```bash
python robomimic/scripts/create_rby1_dp_config.py \
  --dataset ../../generated_datasets/mimicgen/CloseDrawerLeft_RBY1_300mg/close_drawer_300mg_im128_cmd.hdf5 \
  --name rby1_close_drawer_cartesian_cmd_300mg --epochs 20 --save-every 2
```

It prints the split and the resulting number of training steps. For 270 train
demos an epoch is ~10,800 steps (~18 min), so 20 epochs take ~6 hours.

Settings inherited from the RB-Y1 microwave config: head + wrist RGB (ResNet18,
random crop) and the TCP pose as inputs, observation horizon 2, prediction
horizon 16, action horizon 8, DDIM with 10 inference steps, EMA, batch size 16,
learning rate 1e-4, `min_max` action normalization.

## 7. Train

```bash
# quick check (a few steps; writes to /tmp/tmp_trained_models)
python robomimic/scripts/train.py --config robomimic/exps/rby1_close_drawer_cartesian_cmd_300mg.json --debug

# real run in the background
nohup python robomimic/scripts/train.py --config robomimic/exps/rby1_close_drawer_cartesian_cmd_300mg.json \
  < /dev/null > ../../logs/train_close_drawer_300mg.log 2>&1 &
tail -f ../../logs/train_close_drawer_300mg.log
```

- Checkpoints: `experiment_results/<experiment name>/<timestamp>/models/model_epoch_N.pth`.
- TensorBoard: `tensorboard --logdir experiment_results/<experiment name>`.
- If the experiment folder already exists, `train.py` asks whether to delete it.
  Answer `y` only if you no longer need the old runs; with `< /dev/null` the
  question fails instead of deleting anything.
- Speed: ~0.1 s per step on the RTX 2000 Ada (one epoch of 50 demos ≈ 3 min;
  of 270 demos ≈ 18 min).
- The validation loss is a poor guide for this policy: in the 50-demo run it was
  lowest at epoch 10 and tripled by epoch 30, while rollout success went from
  45 % to 65 %. Pick checkpoints by rollout success.

## 8. Evaluate

```bash
MUJOCO_GL=egl python robomimic/scripts/evaluate_rby1_rollouts.py \
  --run-dir experiment_results/rby1_close_drawer_cartesian_cmd_300mg/<timestamp> \
  --episodes 20 --output-dir experiment_results/eval_close_drawer_300mg
```

- Rolls out every saved checkpoint (or `--epochs 10 20`, or `--checkpoint file.pth`)
  in the training environment, the same scenes for every checkpoint (`--seed`).
- `--layouts 1 4 7 8 9` evaluates on the held-out kitchens.
- `--videos` saves a head + wrist video per episode.
- `--horizon` limits the episode length (default 1000 steps; demos take ~600).
- Output per checkpoint: `model_epoch_N.json` (every episode with its scene,
  success, steps and final drawer opening; > 1.0 means the drawer was pulled
  further open) and `summary.json`.
- ~15–25 s per episode.

## 9. Run the full Mobi-π pipeline

Base-pose selection, navigation, the policy, and the arm return run in one
pipeline with `--task close_drawer` (from the repository root). The task's default
checkpoint is epoch 10 of the `look300` run:

```bash
python -m mobipi.scripts.run_rby1_pipeline --task close_drawer \
  --scene-dir scene_data/close_drawer/layout2_style2 --layout-id 2 --style-id 2 \
  --skip-detection --vis
```

Seed sweeps, options, and the scene layouts the drawer task can use:
[Running the RB-Y1 full pipeline](rby1_pipeline.md).

## Replay a dataset

From `external/robomimic`; give the camera names without `_image`, always a
`--video_path` (on-screen `--render` does not work), and `--video_skip 1` for
real time (the default 5 plays at 5× speed):

```bash
# stored camera images, as the policy sees them (rendered datasets only)
python robomimic/scripts/playback_dataset.py --dataset <rendered>.hdf5 --use-obs \
  --render_image_names robot0_head_camera robot0_left_eye_in_hand \
  --video_path /tmp/drawer_obs.mp4 --n 5 --video_skip 1

# re-simulate the saved states
MUJOCO_GL=egl python robomimic/scripts/playback_dataset.py --dataset <dataset>.hdf5 \
  --render_image_names robot0_head_camera --video_path /tmp/drawer_states.mp4 --n 5

# replay the actions open-loop ("playback diverged" warnings are expected)
MUJOCO_GL=egl python robomimic/scripts/playback_dataset.py --dataset <dataset>.hdf5 --use-actions \
  --render_image_names robot0_head_camera --video_path /tmp/drawer_actions.mp4 --n 5
```

Do not replay `source.hdf5` while the MimicGen prepare step is writing to it.

## Results so far

50 scripted demos (45 train / 5 valid), 30 epochs, 20 episodes per checkpoint
in the training kitchens:

| Epoch | 5 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| Success | 40 % | 45 % | 55 % | 55 % | 50 % | 65 % |

Most failures pull the drawer further open: the policy comes down into the open
drawer instead of in front of it, and drags the front outward on its way to the
pre-contact pose. Next: train on the 300 MimicGen demos and evaluate the best
checkpoint on the held-out layouts.

## Fixes this pipeline depends on

- `robomimic/utils/dataset.py`: `min_max` normalized actions are clipped to
  [-1, 1]. Float32 rounding put the nearly constant rotation6d components
  ~1e-5 past ±1, which Diffusion Policy rejects.
- `robomimic/scripts/dataset_states_to_obs.py`: the writer waits for an end
  marker from every worker (it used to stop before large trajectories arrived,
  write an empty file and hang), and absolute WholeBody IK actions are used as
  `actions_abs` (RoboCasa's conversion only handles a right-arm OSC controller).
- `robomimic/envs/env_robosuite.py`: imports `mobipi.envs`, so tools that
  recreate the environment from a dataset know `CloseDrawerLeft`.
- MimicGen `generate_dataset.py`: an `and`/`or` precedence bug saved failed
  trials even with `keep_failed: false` and crashed on the first failure.
