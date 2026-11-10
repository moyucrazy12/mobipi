# Regression matrix

Baseline: `cf9868d5cc4471f573a2691b8849ff13ab047344`, merged `release`.
Use the setup exports in [rby1/README.md](../../rby1/README.md), from the checkout
being tested. Output directories below are exclusive-create; use fresh names.
No training hyperparameters, models, action representations, camera keys or
task success definitions were changed.

## Automated suites

```bash
python -m unittest discover -s rby1/tests -v
python -m unittest discover -s external/robomimic/tests -p 'test_rby1*.py' -v
python -m unittest discover -s external/robomimic/tests -p 'test_diffusion_policy_inference.py' -v
python -m unittest discover -s external/robomimic/tests -p 'test_training_data.py' -v
# Compatibility with the previous discovery location:
python -m unittest discover -s mobipi/collect_demos/tests -v
python -m rby1.tests.check_clis --report /tmp/rby1-cli-check.json
git diff --check
```

To compare CLI help against a baseline, run the same `check_clis.py` file by
absolute path with the baseline checkout as cwd and its source PYTHONPATH;
then pass that report as `--reference` on the refactored checkout. Command
examples containing moved module names are normalized; options/help otherwise
must match. The selected 21 historical commands are listed in
`rby1/tests/fixtures/cli_commands.json`. Three additional tests invoke legacy
scripts from another cwd without the superproject on PYTHONPATH.

| Stage | Executed | Observed result |
|---|---|---|
| Merged release, before moves | Original targeted suites | 90 passed, no skips |
| Demo/policy/config move | Original suites + separate import/config checks | 90 passed; 2 additional checks passed; 21 CLIs passed |
| Robot/environment/pipeline move | Suites and alias checks | 92 passed; 21 CLIs passed; help differs only in two canonical command examples |
| Initial low-level consolidation | Suites + fixed seeded numeric/config checks | 97 passed |
| Test relocation | Canonical suites | 97 passed; old discovery separately passed 51 |
| Added pipeline contracts | Same test file run on baseline and refactor | Control-flow checks passed; virtual-base case unavailable in both environments |
| Final canonical/controller boundary | Canonical suites and dependency suites | 104 run: 103 passed, 1 environment-dependent skip |
| Wheel packaging | `python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/rby1-wheel .` | Built; archive contains canonical code/configs/fixtures and legacy CLI modules |

The skipped case is specifically `RBY1MobileBase` absent from installed
`robosuite.models.bases.BASE_MAPPING`. Before adding its prerequisite guard,
this check failed identically on baseline and refactor with
`TypeError: 'str' object is not callable` from `robot_base_factory`.
It is a pre-existing environment mismatch, not evidence that the complete
mobile pipeline works. The existing 90-test baseline still runs successfully.

## Workflow coverage and manual limits

| Workflow | Automated evidence | Manual validation still needed |
|---|---|---|
| CloseSingleDoor demonstrations | Left-arm environment reset/step, controller/TCP packing and CLI compatibility | Full scripted closing episode, accepted raw recording and replay in the intended model profile |
| CloseDrawer / CloseDrawerLeft | Task registration/defaults, left-arm reset/step, camera/config preservation, collection/gather CLI | Full drawer episode, collection manifest → HDF5 → native MimicGen → replay |
| Faucet demonstrations | Bounded IK/joint limits, reset, recording schema, safety failures, one successful fixed-seed episode before/after | Broader scene/style coverage |
| MimicGen | Native source preparation/segmentation, scoped executor restoration, one native generation, structural validation, all three phases retained | New layouts/styles and large augmentation batches; door/drawer generation not rerun |
| Dataset preparation | HDF5 validation, source-ID remapping/recovery, absolute/relative native dataset windows and train/valid masks | Full new 100-episode faucet conversion; large image rendering |
| Absolute/relative Diffusion Policy | Native geometry/rotation6D/quaternion tests; 9-D absolute and 10-D relative+gripper CPU forward/backward updates, model-state reload/continued updates | Long image training, optimizer-state resume and statistical policy-performance reproduction |
| Observation/inference | Real epoch-9 faucet checkpoint, preprocessing, camera contract, action normalization/queue and identical output action | Full-duration learned-policy task completion |
| Evaluation | Fixed-seed three-step real rollout; same metrics/control/actions; safety/physical restoration rejection, termination/export/resume tests | Statistical door/faucet grids and unseen-scene evaluation |
| Base-pose selection | Generic MobiPi scorer/BO/renderer retained unchanged; task registry and pipeline imports tested | Real matching 3DGS/point-cloud/DINO render and optimization |
| Navigation/collision/posture | Seeded differential-drive path equivalence; navigation hands through the same collision checker, approach mode and gripper holds | Actual virtual-base motion/collision/posture/camera test after mobile-profile setup; full scene navigation |
| Manipulation/arm return | Native policy tests; synthetic dual-arm servo return, other-arm holds, task-undone stop, solver restoration after error | Physical simulation arm return after actual policy completion |
| Result logging | Atomic JSON/CSV, timeout/error denominator and interrupted/pending-trial handling | Complete multi-stage run result/video review |

Native BC/other-algorithm suites, hardware-dependent Diffusion Policy tests and
full external-framework suites were not run; their implementations were not
changed. The experimental `test_mobilization.py` entry point is retained but
its eager renderer/model imports and complete execution were not validated.

## Deterministic real-data comparison

The baseline integration checkout at `77b6d88` has the **same Git tree** as
merged release `cf9868d`. It was kept unchanged for the longer baseline
MimicGen and bounded rollout commands while the clean refactoring worktree
was edited. The merged release itself also ran the initial 90 tests and
scripted collection before any moves.

Commands used (baseline uses the old module names; refactor uses these names):

```bash
python -m rby1.demos.collect_faucet_demos \
  --output /tmp/rby1-cleanup-final-demo --count 1 --max-attempts 1
python -m rby1.demos.faucet_pipeline prepare \
  --source /tmp/rby1-cleanup-final-demo/seeds.hdf5 \
  --output /tmp/rby1-cleanup-final-source.hdf5
python -m rby1.demos.faucet_pipeline generate \
  --source /tmp/rby1-cleanup-baseline-source.hdf5 \
  --output /tmp/rby1-cleanup-final-mg --target-successes 1 --max-attempts 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rby1.policies.evaluate_faucet_dp \
  --check-only --epochs 9 --device cpu --output /tmp/rby1-cleanup-post-inference
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m rby1.policies.evaluate_faucet_dp \
  --epochs 9 --seeds 0 --max-steps 3 --device cpu --output /tmp/rby1-cleanup-post-rollout
```

The generator deliberately uses the identical baseline prepared source/config
and seed. Source preparation is compared separately using identical seed
recordings. Commands that rely on the historical experiment root can instead
take explicit `--checkpoint`/`--run-dir` and `--dataset` paths.

```bash
python -m rby1.tests.compare_recordings \
  --before /tmp/rby1-cleanup-baseline-demo/seeds.hdf5 \
  --after /tmp/rby1-cleanup-final-demo/seeds.hdf5
python -m rby1.tests.compare_recordings \
  --before /tmp/rby1-cleanup-baseline-source.hdf5 \
  --after /tmp/rby1-cleanup-final-source.hdf5
python -m rby1.tests.compare_recordings \
  --before /tmp/rby1-cleanup-baseline-mg/generated.hdf5 \
  --after /tmp/rby1-cleanup-final-mg/generated.hdf5
```

Observed pre/post comparisons: successful scripted collection 1/1, 658 steps;
38 stored arrays bitwise equal. Native preparation: 47 arrays bitwise equal.
Native generation: successful 1/1, 1,444 steps; 48 arrays bitwise equal, including
RGB observations. Floating maximum difference: zero. File-path metadata
attributes are excluded, since implementations/output locations moved.
Epoch-9 checkpoint action and queue output match exactly. Three-step rollout
control and policy-action logs match exactly; result fields and physical
tracking match after excluding wall-clock timing/video paths. Three steps
correctly terminate at the horizon without claiming faucet activation/success.
Compact evidence is in [equivalence.json](equivalence.json).

## Manual full-pipeline procedure

1. Create a separate mobile-model environment; apply the tracked RoboSuite
   installer and download RoboCasa assets. Confirm `RBY1MobileBase` is registered
   and run `python -m unittest rby1.tests.test_pipeline_contracts.VirtualBaseContracts -v`.
   Do not replace the validated fixed-mount/floating-root faucet environment.
2. Set `CHECKPOINT`, `DATASET` and `SCENE_DIR` to actual matching task artifacts.
   The scene needs `pc.ply`, camera capture metadata and a compatible trained
   3DGS checkpoint; check [pipeline setup](../rby1_pipeline.md).
3. Run one seed, with a new output directory:

   ```bash
   python -m rby1.pipeline.run --task close_drawer \
     --checkpoint "$CHECKPOINT" --dataset "$DATASET" --scene-dir "$SCENE_DIR" \
     --seed 0 --skip-detection --output-dir /tmp/rby1-full-pipeline-seed0
   ```

4. Inspect selected base pose and rendered camera match, collision checks and
   arrival error, task torso/head posture, policy observation/action keys,
   stable task completion, arm return and `result.json`/videos. Repeat for
   `close_microwave` with its own matching artifacts. Faucet mobilization is
   not registered in the baseline task registry and was not added here.
5. Run short native door/drawer generation/replay using working copies before
   scaling datasets. Keep statistical grids separate from these smoke checks.

Historical references remain door absolute 63/125 (50.4%), relative 92/125
(73.6%); faucet contact 33/50 (66%), activation 9/50 (18%), full success 3/50
(6%). Those experiments were not rerun. The lightweight epoch-9 checks are not
a reproduction of the historical evaluation grid or physical deployment proof.
