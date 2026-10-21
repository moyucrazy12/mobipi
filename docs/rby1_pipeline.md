# Running the RB-Y1 full pipeline

The pipeline selects a base pose with 3DGS/DINO, navigates there, runs the
Cartesian policy, and returns the left arm after confirming task success.
The current implementation supports **RB-Y1 microwave closing (`CloseSingleDoor`)**.

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
python -m mobipi.scripts.run_rby1_pipeline \
    --checkpoint ckpts/model_115_cartesian.pth \
    --dataset data/mimicgen/CloseDoorLeft_RBY1_115mg/demo_im128_head_wrist_clean37.hdf5 \
    --scene-dir scene_data/close_single_door/layout1_style1 \
    --layout-id 1 --style-id 1 --seed 1 \
    --skip-detection --vis
```

Omit `--vis` to run without a window. Inference defaults to GPU with CPU-sampled
diffusion noise. `--policy-device cpu` moves policy inference to CPU; 3DGS/DINO
still requires CUDA. `--skip-detection` skips the optional object detector;
collision checks remain active.

The default `--policy-steps 2000` is a maximum. Once the door stays within 4.5° of
shut for 0.5 seconds, the policy stops and the arm returns smoothly to its starting
joint posture. The return stops if the door reopens. Overall success requires
navigation, stable closure, and a successful return. Each run saves `result.json`
and pose histories under a new folder in `logs/rby1_pipeline/`.

## Test multiple seeds

This uses the checkpoint and dataset shown above by default:

```bash
python -u -m mobipi.scripts.evaluate_rby1_pipeline \
    --scene-dir scene_data/close_single_door/layout1_style1 \
    --skip-detection --start-seed 0 --num-seeds 20 \
    --output-dir logs/rby1_pipeline/seeds_0_19
```

Seeds 0–19 run sequentially in separate processes, without a viewer. Each seed
changes navigation sampling, pose optimization, and policy noise. Replace
`--start-seed 0 --num-seeds 20` with `--seeds 0 1 42 123` for an explicit selection.
The evaluator accepts the single-run options above, including `--checkpoint`,
`--dataset`, `--layout-id`, and `--style-id`.

To keep running after closing the terminal, use a fresh output directory:

```bash
mkdir -p logs/rby1_pipeline
nohup python -u -m mobipi.scripts.evaluate_rby1_pipeline \
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
new output directory.

## Different checkpoints, scenes, and tasks

For another supported microwave-closing experiment, replace these together:

| Option | Select |
| --- | --- |
| `--checkpoint` | A compatible RB-Y1 Cartesian policy trained for that task. |
| `--dataset` | Matching RGB demonstrations and environment/controller metadata. |
| `--scene-dir` | The trained 3DGS scene for the target task, layout, and style. |
| `--layout-id`, `--style-id` | The layout/style corresponding to that scene. |
| `--output-dir` | A fresh directory for the experiment. |

There is no `--task` flag: the environment name comes from the HDF5 metadata.
**Changing paths alone does not enable a different task**, such as opening a
door or closing a drawer. Before running those tasks, adapt
[`run_rby1_pipeline.py`](../mobipi/scripts/run_rby1_pipeline.py): the task validation
in `main`, fixture selection in `make_env`, success/return conditions in
`run_policy` and `return_arm_to_start`, and the microwave-specific final approach
in `navigate_to_pose`. The imported robomimic `check_policy` and Cartesian adapter
must also match the new policy's observations, actions, and controller. Validate
one visible run before starting its seed sweep.
