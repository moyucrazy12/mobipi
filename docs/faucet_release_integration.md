# Faucet integration into release

Prepared on 2026-11-10. Integration target: `release`. No RB-Y1 module
refactoring or SDK work is included. The original development checkout remains
on `rby1_turn_on_sink_faucet_demos` with its original pending files intact.

## Branches and dependencies

- Superproject preparation: `integrate/faucet-release`, in the sibling
  `../mobipi-faucet-integration` worktree.
- Integration baseline: `origin/release` at
  `f41c808b71f282cd2862b832566bf50d1608a375`.
- RoboMimic: `integrate/faucet-gripper` at
  `c595ab8927f870cb47dc04618eb31ab943becc69`, published to
  `moyucrazy12/mobipi-robomimic`.
- Companion dependency PR: https://github.com/moyucrazy12/mobipi-robomimic/pull/6
- MimicGen: `edbaff93d5e988213a95f53fbbbbc99cecc01921`, already published in
  `moyucrazy12/mobipi-mimicgen` and already pinned by `release`; no new commit
  or upstream push was necessary.
- RoboCasa remains `3683fb01d6fc87d7849e7b5a886d01f4b1d7a55d`. Diffusion Policy,
  LeLaN and SPHINX retain the `release` gitlinks.

The local MimicGen development checkout was older than the team fork and would
have removed its door/drawer interfaces if pinned blindly. The integrated
checkout uses the newer team commit. The original checkout's upstream remote
configuration was preserved; team refs were fetched under `refs/remotes/team/`.

| Workflow | Already present in release dependencies | Added from pending work |
|---|---|---|
| Faucet collection | Native RoboCasa/RoboMimic environment support | Task-local runtime, right-arm bounded IK, recording and progress checks in MobiPi |
| MimicGen | Native segmentation/transforms plus team door/drawer interfaces | Faucet interface, source preparation and scoped waypoint executor in MobiPi |
| Relative policy training | Relative Cartesian geometry, factory/dataset hooks, native diffusion networks | Optional scalar gripper in `relative_diffusion_policy.py` and `rby1_relative_dataset.py` |
| Faucet evaluation | Native checkpoint/observation utilities and common evaluation exporters | Committed evaluator, seed helper, faucet metrics/reporting and experiment paths |
| Door and drawer | Task adapters, saved-base compatibility, shared pipeline/navigation, inference noise generator and min-max clipping | Existing local door replay/diagnostic/path tools preserved; no production change to release normalization or networks |

RoboMimic commits preserve the existing histories: `87090c2` records local
path/evaluation tools, `ea5d80b` adds gripper support, and `e226662` merges the
existing `robocasa` branch. Follow-up test commits make checks portable and
verify model-state reload and the clipping behavior already on `release`.
Historical door JSON edits change dataset paths only. Experimental diagnostic
tools are retained as explicit opt-in commands rather than classified as dead
code. Generated datasets, videos, checkpoints and logs were not committed.

## Preservation and merge resolution

Before changing refs or preparing merges, a recovery snapshot was created at:

```text
../faucet-integration-snapshot-20261110-160942/
```

It contains verified Git bundles, binary HEAD/index patches, changed and
untracked-file archives, file lists, original refs/status, a manifest and SHA-256
checksums for the superproject, RoboMimic, MimicGen, RoboCasa and SPHINX.
After integration preparation, snapshot hashes and archived file contents were
compared against the original checkout: all matched. Ignored downloaded assets
and existing experiment directories remain in place and were not overwritten.
Recovery should use a new directory: clone the relevant `history.bundle` and
apply its saved patches or overlay its `worktree.tar.gz`; do not restore over
the active development tree.

The superproject merge resolved one content conflict in
`mobipi/collect_demos/README.md`. Both complete faucet and drawer sections were
retained. The door collection instructions were also retained. RoboMimic's
replay/evaluation scripts merged automatically; review verified preservation
of the saved-base, task-progress and controller compatibility changes.
The final diff against `release` leaves its door/drawer implementation,
navigation, scoring and shared manipulation pipeline files unchanged.

## Regression commands

From a checkout of the integration branch, with the existing `mobipi` Conda
environment and downloaded RoboCasa assets available:

```bash
conda activate mobipi
export MUJOCO_GL=egl
export NUMBA_CACHE_DIR=/tmp/mobipi-cleanup-numba
export MPLCONFIGDIR=/tmp/mobipi-cleanup-mpl
export PYTHONPATH="$PWD:$PWD/external/robomimic:$PWD/external/mimicgen:$PWD/external/robocasa${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s mobipi/collect_demos/tests -v
python -m unittest discover -s external/robomimic/tests -p 'test_rby1*.py' -v
python -m unittest discover -s external/robomimic/tests -p 'test_diffusion_policy_inference.py' -v
python -m unittest discover -s external/robomimic/tests -p 'test_training_data.py' -v
git diff --check
```

| Validation | Result | What it establishes |
|---|---|---|
| Faucet, pipeline and release-integration suite | 51 passed, no skips | Bounded IK, fixture/reset behavior, recording validation, absolute targets, callback/config loading, recovery/merge behavior, door and drawer reset/controller steps |
| RoboMimic RB-Y1 suite | 33 passed | Cartesian geometry, relative window/queue anchors, gripper handling, path/CLI defaults, CPU training updates and model-state save/reload/continued training |
| Diffusion inference suite | 2 passed | EMA/online component selection with the release inference-generator interface |
| Training data suite | 4 passed | Dataset masks/language handling and shared normalization statistics with release clipping |
| Existing CLI entry points | 21 passed `--help` | Door/drawer/faucet collectors, preparation/evaluation/reporting, policy diagnostics and complete-pipeline entry points import successfully |
| Scripted faucet collection | 1/1 successful | 658 recorded control steps, final water on, final handle angle approximately 0.472 rad |
| Native MimicGen source preparation | Passed | Source accepted by native preparation; ordered boundaries at 263 and 513 |
| Native MimicGen generation | 1/1 successful | Native DataGenerator with transforms enabled; all three phases retained; final water on at approximately 0.492 rad |
| Real checkpoint offline check | Passed | Epoch 9 checkpoint, image preprocessing, 10-D action, relative queue and overview/right-wrist camera contract |

Pipeline recovery and evaluation-grid fixtures are synthetic and self-contained. They exercise
interruption, resume, source-ID remapping and locking without private data.
Saved-start evaluation fixtures retain the physical/geometry rejection,
image-difference diagnostics and resume safeguards without requiring private
rollout artifacts. Path tests no longer require local checkpoint files or a checkout named
`mobipi`. The training smoke uses synthetic vector observations; it verifies
model-state reload and continued updates, not full image training, native
checkpoint-file provenance or optimizer-state resume.

The first clean RoboCasa source checkout lacked downloaded object assets,
causing environment initialization to fail. Linking the existing downloaded
meshes/textures into the isolated checkout resolved that setup failure without
adding source files or changing RoboCasa. Initial older test expectations for
unbounded validation labels and a missing inference-generator fixture were
updated to match existing release behavior; production normalization was not
changed to make tests pass.

### Short real-data checks

These are the exact commands used; use different output names when repeating
because outputs are created exclusively:

```bash
python -m mobipi.collect_demos.collect_faucet_demos \
  --output /tmp/faucet-integration-one-20261110 --count 1 --max-attempts 1
python -m mobipi.collect_demos.faucet_pipeline prepare \
  --source /tmp/faucet-integration-one-20261110/seeds.hdf5 \
  --output /tmp/faucet-integration-source-20261110.hdf5
python -m mobipi.collect_demos.faucet_pipeline generate \
  --source /tmp/faucet-integration-source-20261110.hdf5 \
  --output /tmp/faucet-integration-mg-one-20261110 \
  --target-successes 1 --max-attempts 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m mobipi.collect_demos.evaluate_faucet_dp \
  --check-only --epochs 9 --device cpu \
  --output /tmp/faucet-integration-offline-20261110
```

The offline checkpoint check uses the existing historical experiment root under
`../experiment_results/TurnOnFaucet/`; it is an optional local artifact check,
not a requirement for the portable unit tests. Use `--run-dir`, `--checkpoint`
and `--dataset` to select explicit artifacts elsewhere.

## Reproducibility limits and review boundary

All integration source modules are committed. A separate checkout at
`/tmp/mobipi-faucet-committed-20261110` was constructed from exact superproject
and all six dependency gitlinks, without copying untracked source modules.
Downloaded RoboCasa meshes/textures were supplied as external assets. The
fresh committed checkout passed all 90 tests with no failures or skips. Its
working tree and dependency source trees were clean after testing. The
existing installed RoboSuite 1.5.0 RB-Y1 registration/model patches are still an
environment prerequisite; see `assets/robosuite/install_rby1.sh` and the root
installation instructions. These prerequisites are not claimed to be supplied
by Git alone.

No large generation batch, long GPU training, statistical evaluation or full
3DGS/DINO/navigation/manipulation execution was launched. Door/drawer checks
cover registration, configuration and finite controller steps, not complete
scripted task success. Scene models, trained checkpoints and datasets remain
external inputs. No SDK backend or SDK probes were changed.

Preserved experimental references, not remeasured here: door absolute
63/125 (50.4%), door relative 92/125 (73.6%); faucet contact 33/50 (66%),
activation 9/50 (18%), full success 3/50 (6%). The overview and right-wrist
cameras remain required. These results do not establish physical deployment
readiness.

GitHub authentication is available. `release` reports `protected=false` and no
active repository rulesets were returned; absence of protection is not workflow
authorization to merge. The dependency PR initially reported mergeable, no
checks and no review decision. Both PRs must be reviewed through the normal
team workflow. Prefer a normal merge for the companion dependency PR so the
tested gitlink remains reachable in its history. No merge into `release` or
`robocasa` is performed by this preparation.

After approved integration, update a clean `release` checkout from the remote
and record its exact SHA. Only then begin the separately requested refactoring
branch. Do not pull or switch the original dirty development checkout to
complete integration; use the dedicated integration worktree or a fresh clone.

After the team has reviewed and authorized the normal merges, these commands
complete the prepared integration (do not use force, squash or bypass flags):

```bash
gh pr view 6 --repo moyucrazy12/mobipi-robomimic
gh pr checks 6 --repo moyucrazy12/mobipi-robomimic
gh pr merge 6 --repo moyucrazy12/mobipi-robomimic --merge
gh pr view integrate/faucet-release --repo moyucrazy12/mobipi
gh pr checks integrate/faucet-release --repo moyucrazy12/mobipi
gh pr merge integrate/faucet-release --repo moyucrazy12/mobipi --merge
```

If no CI checks are configured, `gh pr checks` may report that none exist;
that does not substitute for review or authorize skipping any newly required
checks. After merging, use a fresh checkout to record the new baseline:

```bash
git clone --branch release --recurse-submodules \
  git@github.com:moyucrazy12/mobipi.git ../mobipi-release-verified
git -C ../mobipi-release-verified rev-parse HEAD
git -C ../mobipi-release-verified submodule status
```
