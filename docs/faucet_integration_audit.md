# Faucet integration gate and proposed RB-Y1 boundary

Audit date: 2026-11-10. This is a preliminary Phase 1 report, not the completed
repository history audit or a claim that refactoring is complete.

For the subsequent integration preparation, dependency commits and regression
results, see [faucet_release_integration.md](faucet_release_integration.md).
The observations below describe the original development checkout.

## Integration status

- Current branch: `rby1_turn_on_sink_faucet_demos`.
- Current HEAD: `8692c27cce5038d9ee9db63880f4b452c57ff575`.
- Remote default: `release`. The user confirmed that `release` replaces `main`
  as the integration target and later refactoring baseline.
- Fetched `origin/release`: `f41c808b71f282cd2862b832566bf50d1608a375`.
- HEAD is one commit ahead and 30 commits behind `origin/release`; its tracking
  branch is incorrectly set to `origin/release`, not a faucet feature branch.
- Common ancestor: `e0ae2b205e0bacabcc508a5af935ca7b845ad6af`.
- SSH fetch succeeded with network access. GitHub CLI PR inspection failed with
  `HTTP 401: Requires authentication` from `https://api.github.com/graphql`.
  Existing PRs, checks, reviews, protections and merge authorization are unknown.
- A read-only `git merge-tree` preview of the committed tips reports a content
  conflict in `mobipi/collect_demos/README.md`. This does not include pending
  working-tree changes and is not a complete integration rehearsal.

Per the requested phase boundary, do not create `refactor/codebase-cleanup` or
`rby1/` until faucet integration is completed and the resulting `release` commit
is recorded. No code was moved, removed, copied or consolidated in this audit.
No merge, push, dependency reset or SDK change was performed.

## Working-tree and dependency evidence

The existing working tree contains modified faucet collector, runtime, dataset,
MimicGen executor, configuration, documentation and tests. Untracked additions
include bounded IK, progress tracking, policy preparation/evaluation/reporting,
relative policy configurations, experiment paths, a resumable collection
pipeline and its tests. These must not be discarded as debugging artifacts.

`external/robomimic` contains both modified tracked files and untracked scripts;
the passing local tests exercise this working tree, not just its committed SHA.
In particular, faucet evaluation imports the currently untracked
`robomimic.scripts.eval_rby1_cartesian_dp`. A superproject-only source commit
would not make this checkout reproducible. Relative policy and dataset changes
add an optional scalar gripper without applying Cartesian transforms to it.
Other pending changes also affect door experiments; do not stage them wholesale
as faucet-only work without reviewing their dependencies.

| Dependency | HEAD gitlink | Checked-out commit | `origin/release` gitlink |
|---|---|---|---|
| MimicGen | `e0d55fc` | `05c723b` | `edbaff9` |
| RoboMimic | `a6706a5` | `c7e9c77` + local changes | `37ad81c` |
| RoboCasa | `3683fb0` | `3683fb0` | `3683fb0` |

RoboCasa has an untracked `robocasa/models/assets/textures.zip`. The superproject
also has an untracked `external/sphinx/` checkout and `repository_files.txt`;
neither is classified as removable. `origin/release` already tracks SPHINX as a
submodule. Diffusion Policy and LeLaN match their current HEAD gitlinks.
MimicGen's configured superproject URL points at the team fork while the local
submodule remote still points at `yjy0625/mobipi-mimicgen`; reconcile deliberately
before publishing dependency changes.

## Preliminary history and usage map

Authors below are Git records, not proof of original implementation ownership.
The complete contributor/file/rename audit remains pending after integration.

| File/module | Relevant commits | Recorded contributors | Current purpose and usage | Recommendation |
|---|---|---|---|---|
| `mobipi/collect_demos/faucet_*`, collector, tests | `8692c27`, pending local changes | David Iosifescu; pending changes unattributed | Right-arm scripted demonstrations, task adapter, validation, MimicGen | RB-Y1 extension candidates; preserve all phases and safety checks |
| `mobipi/collect_demos/auto_close_microwave_door.py` | `4f7e15d`, `9b16628` | David Iosifescu, moyucrazy12 | Door collector plus controller, pose and MuJoCo helpers imported by faucet | Extract shared RB-Y1 helpers only after numerical tests |
| `assets/robosuite/` | `166be92`, `b67a353`, `aeebb9a`, `659858a`, `f2731c6` | kahoujo1, David Iosifescu, moyucrazy12, Josef Kahoun | Robot/base/gripper assets and installer; faucet loads assets dynamically | Keep paths stable until installer and model references are audited |
| `mobipi/utils/rby1_utils.py`, mobilization entry point | `37446e7`, `6ac3884`, `aa25307`, `3661031` | Josef Kahoun | RB-Y1 base movement and mobilization; newer integration changes exist on release | Audit integrated version before moving |
| Drawer collection, task modules and pipeline on release | `a2b63d9`, `00bb9ef`, `c94e1cc`, `b763e02`, `3bf9d93` | Josef Kahoun | Drawer demonstrations, task dispatch, navigation/manipulation and arm return | Include in integrated dependency map; absent from current faucet checkout |
| RoboMimic Cartesian/relative policy modules | `b95529f`, `0e81bbf`, `4bac77a`, `03b728c`, `c7e9c77`, pending changes | Full per-file author audit pending | Shared networks, dataset windows, action geometry, checkpoint evaluation | Reuse native framework; assess extension extraction separately from hooks |

Faucet call graph:

```text
collect_faucet_demos -> faucet_runtime -> existing door controller/pose helpers
                    -> faucet_bounded_ik -> MuJoCo/SciPy
                    -> faucet_progress, faucet_dataset
faucet_pipeline -> collector/runtime/dataset -> native MimicGen preparation/generation
pipeline -> pipeline_tasks callback strings -> collect/prepare/generate/validate/convert
prepare_faucet_dp -> RoboMimic relative Cartesian geometry + HDF5
evaluate_faucet_dp -> runtime + RoboMimic checkpoint/observation/evaluation helpers
report_faucet_evaluation -> common RoboMimic report/export helpers
```

Native RoboMimic training remains the training implementation. The faucet
configs retain `faucet_overview_image` and `robot0_right_eye_in_hand_image`,
right-arm base-relative TCP observations, relative pose9 and scalar gripper.
No architecture, normalization, frames, mathematical definitions or training
hyperparameters were changed during this audit.

## Proposed migration map — no moves executed

Destinations are provisional until the integrated history and dependency map
are complete. Every existing script/module CLI must remain callable. Wrappers
must preserve imported public functions, not just invoke `main()`; existing
mock patch targets and dynamic callback strings also require regression checks.

| Existing path (repository-relative) | Candidate destination | Compatibility / upstream handling | Required move validation |
|---|---|---|---|
| `mobipi/collect_demos/collect_faucet_demos.py`, `faucet_pipeline.py`, `faucet_dataset.py`, `faucet_progress.py` | `rby1/demos/` | Old module/script wrappers; import native MimicGen and dataset utilities | Collector, runtime, dataset and progress suites; one real demo and source replay |
| `mobipi/collect_demos/faucet_runtime.py`, `faucet_bounded_ik.py` | `rby1/robot/` | Preserve old imports; retain RoboCasa subclass and scoped adapters | Runtime, reset, bounded IK, joint ordering, TCP/ref frames |
| `mobipi/collect_demos/prepare_faucet_dp.py`, `evaluate_faucet_dp.py`, `report_faucet_evaluation.py` | `rby1/policies/` | Preserve CLI/imports; reuse RoboMimic networks, transforms and exporters | Evaluation grid, relative geometry/queue/gripper, checkpoint and preprocessing checks |
| Faucet JSON configurations | `rby1/configs/` | Preserve old explicit config paths; avoid copying configs | Default and explicit config loading; camera/action equality; train smoke tests |
| `mobipi/collect_demos/pipeline.py`, `pipeline_tasks.py` | `rby1/pipeline/` if integrated usage supports it | Preserve dynamic callbacks and resumable manifests; distinguish demo pipeline from mobilization pipeline | Resume/recovery suite and old CLI behavior |
| `mobipi/collect_demos/tests/test_faucet*.py`, relevant pipeline tests | `rby1/tests/` | Preserve or redirect test discovery; resolve fixtures explicitly | Run before and after each move with identical fixtures |
| `mobipi/utils/rby1_utils.py`, release RB-Y1 pipeline/task entry points | `rby1/robot/`, `rby1/pipeline/` after audit | Existing entry-point wrappers; retain shared scoring/navigation in MobiPi | Door/drawer task initialization, collision/base motion, full pipeline smoke |
| `assets/robosuite/` | Retain initially | Installer modifies RoboSuite registration and base controller files; document exact patches | Installation/model-loading checks before any asset relocation |
| RoboMimic framework hooks and MimicGen integration | Retain initially | Document necessary native registration/dataset/algo hooks; no upstream file copies | Absolute/relative training, inference and MimicGen integration tests |

`experiment_paths.py` uses file-depth-dependent project-root discovery; moving
it without an explicit root strategy changes dataset/checkpoint defaults.
`faucet_runtime.py` similarly derives assets from `__file__`. JSON defaults use
`with_name()`, and task callbacks contain old module names. These are concrete
path/import hazards, not grounds for dropping code.

No duplicate has yet been proven safe to delete. Task-specific reporting and
action handling have different success/arm/gripper semantics; retain them until
equivalence or deliberate separation is demonstrated.

## Baseline test results and commands

From the repository root, using the existing environment:

```bash
export MUJOCO_GL=egl
export NUMBA_CACHE_DIR=/tmp/mobipi-cleanup-numba
export MPLCONFIGDIR=/tmp/mobipi-cleanup-mpl
PYTHON=/home/david-isfc/miniconda3/envs/mobipi/bin/python
"$PYTHON" -m unittest discover -s mobipi/collect_demos/tests -v
"$PYTHON" -m unittest discover -s external/robomimic/tests -p 'test_rby1*.py' -v
"$PYTHON" -m unittest discover -s external/robomimic/tests -p 'test_diffusion_policy_inference.py' -v
git diff --check
```

| Check | Observed result | Limits |
|---|---|---|
| Faucet suite | 44 run, OK, one class-level skip | `PipelineTests` requires `data/faucet_mg_diverse_100/generated.hdf5`, now absent there; real recordings exist outside the repo under `../experiment_results/TurnOnFaucet/demos/` |
| RoboMimic `test_rby1*.py` | 32 passed | Local modified dependency tree, not a clean checkout |
| Diffusion inference | 2 passed | EMA/online component checks, no trained-policy performance evaluation |
| Whitespace | Passed | Tracked pre-existing diff only |

An initial faucet run without a writable Numba cache failed during imports.
Setting the temporary cache directories above resolved those environment
errors without modifying installed packages. No long training or policy
evaluation was launched. Real demonstration success, 100-demo MimicGen
generation, train/save/reload/resume, all three task workflows and complete
mobilization have not been rerun in this audit. Local test logs are
`/tmp/mobipi-faucet-tests.log`, `/tmp/mobipi-rby1-tests.log` and
`/tmp/mobipi-inference-tests.log` (temporary, not durable artifacts).

## Experimental references and artifact policy

Preserve the user-supplied statistical references independently of smoke tests:

- CloseSingleDoor absolute Cartesian: 63/125 (50.4%).
- CloseSingleDoor relative Cartesian: 92/125 (73.6%).
- TurnOnSinkFaucet, two initial configurations with 25 diffusion seeds each:
  contact 33/50 (66%), activation 9/50 (18%), full completion 3/50 (6%).

These results were not reproduced by the checks above. A separately inspected
local halftime presentation report summarizes 23 rollouts (3 full successes,
5 activations, 19 contacts); it is a different experiment and must not replace
the supplied 50-rollout reference. The checkpoint's overview and right wrist
camera contract is preserved; physical deployment readiness is not established.

Datasets, checkpoints, videos and verbose logs should remain outside source
Git. Existing `data/`, `ckpts/`, `logs/` and Python cache ignores cover some,
but not all, possible output locations. No tracked HDF5/PT/PTH/MP4/log files
were found by the targeted extension check. Do not broadly delete or ignore
all untracked files: several are necessary source modules. No generated
artifact or SDK probe was modified.

## Remaining gate

Authenticate GitHub PR access, review and publish the required dependency
changes on their appropriate feature branches, resolve the faucet/release
integration conflict without dropping door/drawer documentation, and verify
the required checks/reviews. Prepare or update the faucet PR targeting
`release`; merge only if the requested workflow conditions are established.
Until integration is complete, the refactor baseline does not exist. This
report does not declare the current faucet branch ready for integration.
