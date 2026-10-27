# Running the RB-Y1 full pipeline

The pipeline selects a base pose with 3DGS/DINO, navigates there, runs the
Cartesian policy, and returns the left arm after confirming task success.
`--task` selects one of the supported tasks:

| `--task` | Environment | Done when | 3DGS scenes |
| --- | --- | --- | --- |
| `close_microwave` (default) | `CloseSingleDoor` (microwave) | door within 4.5° of shut | `scene_data/close_single_door/` |
| `close_drawer` | `CloseDrawerLeft` ([policy pipeline](rby1_close_drawer_policy.md)) | drawer at most 5 % open | `scene_data/close_drawer/` |

Each task sets its default `--checkpoint` and `--dataset`, which the examples
below also pass explicitly.

Run all commands from the repository root:

```bash
conda activate mobipi_pipeline
```

You need a compatible policy checkpoint, an RGB HDF5 demonstration dataset with
a `mask/train` split, and a matching 3DGS scene directory containing `pc.ply` and
`model/**/nerfstudio_models/step-000029999.ckpt`. For another checkpoint step, use
`--scene-model-step`. See [Gaussian splatting](gaussian_splatting.md) for scene collection
and training. Existing scenes can be downloaded with:

```bash
python mobipi/scripts/download_scene_models.py
```

## One run with visualization

```bash
python -m mobipi.scripts.run_rby1_pipeline --task close_microwave \
    --checkpoint ckpts/model_115_cartesian.pth \
    --dataset data/mimicgen/CloseDoorLeft_RBY1_115mg/demo_im128_head_wrist_clean37.hdf5 \
    --scene-dir scene_data/close_single_door/layout1_style1 \
    --layout-id 1 --style-id 1 --seed 1 \
    --skip-detection --vis
```

For the drawer:

```bash
python -m mobipi.scripts.run_rby1_pipeline --task close_drawer \
    --checkpoint external/robomimic/experiment_results/rby1_close_drawer_cartesian_cmd_look300/20261020114821/models/model_epoch_10.pth \
    --dataset datasets/close_drawer_look/close_drawer_look_300_im128_cmd.hdf5 \
    --scene-dir scene_data/close_drawer/layout2_style2 \
    --layout-id 2 --style-id 2 --seed 1 \
    --skip-detection --vis
```

Each drawer scene capture has one drawer open, and the drawer run makes that drawer
the target, opened as far as in the capture, so the renders match the simulation.
In layouts 1 and 9 that drawer has no room for the robot on its right, which
`CloseDrawerLeft` needs, and the run stops with an error. `--no-scene-overrides`
samples the drawer as in training instead, but the renders then show a different
drawer open.

Omit `--vis` to run without a window. Inference defaults to GPU with CPU-sampled
diffusion noise. `--policy-device cpu` moves policy inference to CPU; 3DGS/DINO
still requires CUDA. `--skip-detection` skips the optional object detector;
collision checks remain active.

Navigation drives to a staging pose up to 0.5 m behind the selected pose, turns
to its heading there, and drives straight in. The default `--policy-steps 2000` is
a maximum. Once the task stays done for 0.5 seconds, the policy stops and the arm
returns smoothly to its starting joint posture. The return stops if the task comes
undone (the door or drawer reopens). Overall success requires navigation, stable
completion, and a successful return. Each run saves `result.json`
and pose histories under a new folder in `logs/rby1_pipeline/`.

## Test multiple seeds

This uses the task's default checkpoint and dataset:

```bash
python -u -m mobipi.scripts.evaluate_rby1_pipeline --task close_microwave \
    --scene-dir scene_data/close_single_door/layout1_style1 \
    --skip-detection --start-seed 0 --num-seeds 20 \
    --output-dir logs/rby1_pipeline/seeds_0_19
```

Seeds 0–19 run sequentially in separate processes, without a viewer. Each seed
changes navigation sampling, pose optimization, and policy noise. Replace
`--start-seed 0 --num-seeds 20` with `--seeds 0 1 42 123` for an explicit selection.
The evaluator accepts the single-run options above, including `--task`,
`--checkpoint`, `--dataset`, `--layout-id`, and `--style-id`.

To keep running after closing the terminal, use a fresh output directory:

```bash
mkdir -p logs/rby1_pipeline
nohup python -u -m mobipi.scripts.evaluate_rby1_pipeline --task close_microwave \
    --scene-dir scene_data/close_single_door/layout1_style1 \
    --skip-detection --num-seeds 20 \
    --output-dir logs/rby1_pipeline/seeds_0_19 \
    > logs/rby1_pipeline/seeds_0_19.console.log 2>&1 &
```

Monitor with `tail -f logs/rby1_pipeline/seeds_0_19.console.log`. Results contain:

| File | Contents |
| --- | --- |
| `summary.json` | Successful seeds, success rate, and failures grouped by stage/status. |
| `results.csv` | Per-seed outcomes, timings, and paths to detailed logs. |
| `seed_N/attempt_XXX/result.json` | Full navigation, policy, and arm-return results. |
| `seed_N/attempt_XXX.log` | Console output, including progress and errors. |

Failures continue to the next seed. The wall-time limit is 20 minutes per seed;
change it with `--timeout-minutes`. Errors and timeouts count against the success
rate. Ctrl+C, or `kill PID` for a background run, stops the batch and saves progress.
Repeat the same command with **`--resume`** to skip finished seeds (including
failures) and retry interrupted ones. Changing settings or input files requires a
new output directory, and so does editing the pipeline or task modules.

## Different checkpoints, scenes, and tasks

For another experiment on a supported task, replace these together:

| Option | Select |
| --- | --- |
| `--task` | The task the policy was trained for. |
| `--checkpoint` | A compatible RB-Y1 Cartesian policy trained for that task. |
| `--dataset` | Matching RGB demonstrations and environment/controller metadata. |
| `--scene-dir` | The trained 3DGS scene for the target task, layout, and style. |
| `--layout-id`, `--style-id` | The layout/style corresponding to that scene. |
| `--output-dir` | A fresh directory for the experiment. |

The dataset's environment must match the task: `close_drawer` rejects a
`CloseSingleDoor` dataset, for example.

## Adding a task

Task-specific code lives in [`mobipi/pipeline/tasks/`](../mobipi/pipeline/tasks/);
the pipeline scripts are task-agnostic. To add a task:

1. Subclass `PipelineTask` ([`base.py`](../mobipi/pipeline/tasks/base.py)) in a new
   module. Set `name` (the `--task` value), `env_name`, the default checkpoint and
   dataset, and `success_definition`. Implement `progress_fn(env)`: check that the
   env is this task and return a callable giving its progress (higher is closer to
   done), which succeeds at `success_threshold`. Override `env_overrides` (robosuite
   kwargs, e.g. the fixture off the training layouts), `scene_overrides` (make the
   simulation match the 3DGS capture) and `collision_threshold_multiplier` (scoring
   radius) if needed.
2. Add an instance to `TASKS` in [`__init__.py`](../mobipi/pipeline/tasks/__init__.py).
3. Register custom environments in `mobipi/envs/__init__.py`.

The policy must have the observations, actions, and controller that robomimic's
`check_policy` and `RBY1CartesianAdapter` expect (head + left wrist 128 px RGB,
TCP pose, 9-D `actions_eef_cmd`, as for both current tasks). Validate one visible
run before starting its seed sweep.
