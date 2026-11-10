# RB-Y1 boundary audit (before migration)

Baseline: `cf9868d5cc4471f573a2691b8849ff13ab047344` on `origin/release`.
PRs MobiPi #22 and RoboMimic #6 are confirmed merged. The baseline source tree
is identical to integration commit `77b6d88`; the merged release was nevertheless
tested independently. Exact dependency revisions are in [baseline.json](baseline.json).

## History evidence

[history.json](history.json) records all reachable commits attributed to names
present in the superproject team history, with full SHAs, parents, authors,
dates and changed paths (rename detection enabled). This covers 72 superproject,
30 RoboMimic and 5 MimicGen commits; no matching team-author commit exists in
the pinned RoboCasa history. These are attribution records, not proof of original
authorship. Merge commits record their parents; non-merge path lists show which
files were introduced or repeatedly changed. Nonintegrated remote experiments,
including SDK work, are outside this cleanup. No branch history is discarded.

| Area | Historical evidence | Usage and boundary decision |
|---|---|---|
| Original MobiPi | `19b130b`, Jingyun Yang, initial release | Keep generic scoring, DINO encoders, scene renderer, optimization, navigation utilities, policy loading, baseline evaluations and visualization in `mobipi` |
| RB-Y1 model/assets | `166be92`, `b67a353`, `659858a`, `f2731c6`; kahoujo1, David Iosifescu, moyucrazy12, Josef Kahoun | Team robot/base/gripper models and installer; retain installed asset paths to avoid changing MJCF resolution/registration |
| Door demonstrations | `4f7e15d`, `9b16628`; David Iosifescu, moyucrazy12 | Active microwave surface collector supplies common controller/pose helpers to faucet and drawer; general door alternative remains experimental, not deleted |
| Drawer demonstrations/environment | `a2b63d9`, `021ffb5`, `48de77c`, `00bb9ef`, `c94e1cc`; Josef Kahoun | Left-arm collector, randomized batching, raw-HDF5 gathering and CloseDrawerLeft reset/placement variant belong to `rby1` |
| Faucet | `8692c27`, `53aaa86`, integration/test commits through `8ca5c75`; David Iosifescu | Right-arm task adapter, bounded IK, recording, validation, scoped MimicGen executor, policies and reports belong to `rby1` |
| Mobilization/pipeline | `37446e7`, `aa25307`, `8e88c83`, `b763e02`, `3bf9d93`; Josef Kahoun, moyucrazy12 | RB-Y1 base/controller adaptations, task registry, policy execution/arm return and result orchestration move; import generic MobiPi scoring/planning |
| Shared scoring | `c94e1cc` changes `mobipi/utils/score_utils.py` | Retain the shared framework implementation; a team edit does not make the entire scorer robot-specific |
| RoboMimic Cartesian/relative policy | `b95529f`, `0e81bbf`, `4bac77a`, `03b728c`, `ea5d80b`, `e226662` | Retain native algorithm/dataset hooks and geometry used by checkpoint loading and standalone RoboMimic commands; import them from RB-Y1 policy entry points |
| MimicGen team interfaces | `e4283cf`, `c5d76e9`, `80f4a9c`, `edbaff9` | Retain native interface registration and door/drawer adapters in the dependency; faucet's scoped adapter remains in the superproject |

RoboSuite is an installed dependency, not a superproject submodule. Its team
patch provenance is tracked in `assets/robosuite`: installer edits registration
in `models/robots/manipulators/__init__.py`, `models/grippers/__init__.py`,
`robots/__init__.py` and `controllers/parts/mobile_base/joint_vel.py`, and installs
the tracked RB-Y1 model, gripper, bases and controller JSON. Keep those changes
minimal and do not reinstall or alter the host environment in this cleanup.
RoboCasa faucet hinge-address corrections are scoped to the team subclass;
no RoboCasa source change or pin update is required.

## Dependency map

| Workflow / entry point before migration | Dependencies and config | Inputs / outputs | Baseline coverage |
|---|---|---|---|
| `collect_demos/auto_close_microwave_door.py`, `collect_close_door_demos.py` | RoboCasa CloseSingleDoor, installed RBY1 WholeBody controller, left arm | Raw DataCollectionWrapper episodes, batch manifest/logs | Environment/controller step and CLI; full task demonstration requires separate run |
| `auto_close_drawer.py`, `collect_close_drawer_demos.py`, `gather_demos_hdf5.py` | `mobipi.envs.kitchen_drawer`, door controller helpers, `rby1_utils`, native RoboMimic schema | Raw recordings → HDF5; drawer config under `configs/mimicgen` | Environment/controller step and CLI; full task success not established |
| `collect_faucet_demos.py` | `faucet_runtime`, bounded IK, progress, dataset writer; faucet JSON; right arm and overview/right-wrist cameras | `seeds.hdf5`, video, diagnostics, summary | Scripted tests and one real fixed-seed collection |
| `faucet_pipeline.py` | Native MimicGen task spec, DataGenerator and preparation; scoped waypoint execution | Seeds → prepared source → generated HDF5 | Segmentation/schema/executor tests and one real generation |
| Demo `pipeline.py` / `pipeline_tasks.py` | Dynamic callback strings; existing collection/preparation/generation/conversion | Manifest, recoverable attempts, merged dataset, optional policy dataset | Synthetic interruption/resume/locking/source-index tests |
| `prepare_faucet_dp.py` | RoboMimic relative geometry and dataset; task policy JSON | Exactly 100 generated episodes → source-disjoint train/valid HDF5 | Structural tests; existing dataset/checkpoint check; conversion is not a generic small-fixture converter |
| Native RoboMimic `train_rby1` / `train` | Relative/absolute factories, action normalization, EMA, dataset window anchors | Policy HDF5 → checkpoint/logs | CPU absolute/relative+gripper update, model-state reload/continued update; optimizer resume not established |
| `evaluate_faucet_dp.py`, reporter | Native checkpoint loader/preprocessing, relative pose9+gripper, bounded controller and task metrics | Explicit checkpoint/dataset → saved states, rollout logs/video, metrics/reports | Restore guards, exports, seed grids and real offline inference; statistical rates not rerun |
| `run_rby1_pipeline.py`, `evaluate_rby1_pipeline.py` | Task registry → generic MobiPi scene/DINO/BO scoring → RB-Y1 base planner → RoboMimic manipulation → arm return | Checkpoint, dataset, point cloud, 3DGS scene → navigation and task results | Task/controller/CLI checks; full GPU scene/navigation execution requires external assets/manual run |
| `eval/test_mobilization.py` | RB-Y1 utilities plus shared score/scene/nav modules | Scene model, dataset or policy → mobilization logs | Experimental entry point retained; full run unverified |

Path hazards: file-depth-derived roots, sibling JSON lookup, collector subprocess
paths, bare sibling imports, task callback strings, module identity used by mock
patches, source-inspecting tests, and evaluation code hashes in resumable manifests.
Do not resume a pre-cleanup evaluation manifest as though its code fingerprint
were unchanged. Dataset/checkpoint paths and numerical configuration contents
must remain stable. Native submodule commands must not acquire a dependency on
the new superproject package.

## Duplication decisions

[duplicates.json](duplicates.json) is the pre-move AST comparison. No exact
duplicate top-level Python files were found. Eleven top-level functions have
identical ASTs in the general and microwave door scripts. Consolidate only the
small simulator-access and geometric helpers whose globals/dependencies can be
kept identical; retain script-specific discovery/controller/contact functions
until their different assumptions are covered by tests. Preserve imported names
through re-exports. Validate extracted functions against the baseline source
and deterministic numerical outputs before accepting consolidation.

The two door planners, task-specific contact/success logic, gripper conventions,
faucet/door reports, and absolute/relative action encoders are intentionally
different. RoboMimic and faucet path helpers have different standalone-package
requirements; do not merge them by introducing a reverse framework dependency.
Do not delete diagnostic/trace tools or general door experiments. Moving code
and replacing its old implementation with a module-identity compatibility
wrapper removes duplicate maintenance sites without copying framework code.

## Baseline and scope

The independently executed merged-release suite passed 90 tests. Repeat the
21 CLI checks, one collection/generation and real checkpoint checks with fresh
output paths. Preserve artifacts outside Git and compare identical-seed results
after migration. Record later results and manual procedures in the regression
matrix. Large training/evaluation and hardware SDK integration are excluded.
