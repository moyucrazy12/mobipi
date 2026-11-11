# Migrating an existing RB-Y1 feature branch

Read this guide and [README.md](README.md) before adapting an older branch.
Merge only after PR #23 and its companion RoboMimic/MimicGen PRs have merged.

## Architecture and imports

```text
rby1/
  demos/       Scripted planners, recording, validation and faucet MimicGen
  robot/       Shared controllers, simulation helpers, task adapters, navigation
  policies/    Faucet dataset preparation, checkpoint evaluation and reports
  pipeline/    Collection orchestration and complete MobiPi/task entry points
mobipi/        Original framework: scene scoring, base-pose selection, planning
external/      Pinned dependencies, including RoboMimic training and MimicGen
```

Legacy compatibility wrappers were intentionally removed. Update imports,
dynamic callback strings, subprocess commands and scripts to canonical names:

| Old module or imported helpers | Canonical location |
|---|---|
| `mobipi.collect_demos.auto_close_microwave_door` | `rby1.demos.auto_close_microwave_door` |
| `mobipi.collect_demos.auto_close_drawer` | `rby1.demos.auto_close_drawer` |
| `mobipi.collect_demos.collect_close_door_demos` | `rby1.demos.collect_close_door_demos` |
| `mobipi.collect_demos.collect_close_drawer_demos` | `rby1.demos.collect_close_drawer_demos` |
| `mobipi.collect_demos.collect_faucet_demos` | `rby1.demos.collect_faucet_demos` |
| `mobipi.collect_demos.faucet_pipeline` | `rby1.demos.faucet_pipeline` |
| `mobipi.collect_demos.faucet_dataset` | `rby1.demos.faucet_dataset` |
| `mobipi.collect_demos.faucet_runtime` | `rby1.robot.faucet_runtime` |
| `mobipi.collect_demos.faucet_bounded_ik` | `rby1.robot.faucet_bounded_ik` |
| `mobipi.collect_demos.auto_close_microwave_door` controller helpers | `rby1.robot.controllers` |
| `mobipi.utils.rby1_utils` | `rby1.robot.navigation` |
| `mobipi.envs.kitchen_drawer` | `rby1.robot.kitchen_drawer` |
| `mobipi.collect_demos.prepare_faucet_dp` | `rby1.policies.prepare_faucet_dp` |
| `mobipi.collect_demos.evaluate_faucet_dp` | `rby1.policies.evaluate_faucet_dp` |
| `mobipi.collect_demos.pipeline` | `rby1.pipeline.collect` |
| `mobipi.collect_demos.pipeline_tasks` | `rby1.pipeline.collection_tasks` |
| `mobipi.pipeline.tasks` | `rby1.pipeline.tasks` |
| `mobipi.scripts.run_rby1_pipeline` | `rby1.pipeline.run` |
| `mobipi.scripts.evaluate_rby1_pipeline` | `rby1.pipeline.evaluate` |

For example, import `build_controller_interface` from `rby1.robot.controllers`.
Use `python -m rby1...` for subprocesses. Native RoboMimic/MimicGen entry points
remain in their dependency packages; do not copy them into `rby1/`.

## Configurations and commands

Experiment-specific JSON files come from the team's private OneDrive folder;
ask a teammate for access. Store them outside Git and supply explicit paths.
`CONFIG_DIR` below is a local downloaded/synced folder, not a public URL.
Runtime robot/controller assets remain under `assets/robosuite/`.

Run from the repository root in the matching simulation environment, with the
installation and `PYTHONPATH` from [README.md](README.md). Choose new outputs;
set artifact variables to real, matching files. Main command examples:

```bash
export CONFIG_DIR="/path/to/private/team/configs"
python -m rby1.demos.collect_faucet_demos --config "$CONFIG_DIR/faucet_config.json" --count 1 --max-attempts 1 --output data/faucet_seed
python -m rby1.demos.faucet_pipeline prepare --source data/faucet_seed/seeds.hdf5 --output data/faucet_source.hdf5
python -m rby1.demos.faucet_pipeline generate --config "$CONFIG_DIR/faucet_config.json" --source data/faucet_source.hdf5 --output data/faucet_mg --target-successes 1 --max-attempts 1
python -m robomimic.scripts.train_rby1 --config "$CONFIG_DIR/faucet_relative_cartesian_dp.json" --dataset "$DATASET"
python -m rby1.policies.evaluate_faucet_dp --checkpoint "$CHECKPOINT" --dataset "$DATASET" --check-only --device cpu --output /tmp/faucet_check
python -m rby1.pipeline.run --task close_drawer --checkpoint "$DRAWER_CHECKPOINT" --dataset "$DRAWER_DATASET" --scene-dir "$SCENE_DIR" --skip-detection
```

The README covers door/drawer collectors, native MimicGen, training-data
preparation, absolute/relative policies, rollout evaluation and batch resume.
Faucet rollout evaluation requires `--task-config`; offline `--check-only` does not.
Full mobilization supports `close_drawer` and `close_microwave`, requires mobile
model assets and CUDA scene scoring. The door collector also needs a display.

## Preserve and merge a feature branch

1. Stay on your feature branch. Inspect changes and make an external backup.
   These commands preserve commits and tracked diffs; the untracked list is only
   an inventory. Copy those source files to the backup with their relative paths,
   and separately preserve any important ignored files, before continuing.

   ```bash
   git status --short
   stamp=$(date +%Y%m%d-%H%M%S)
   backup_dir="$PWD/../development_archive/branch-migration/$stamp"
   mkdir -p "$backup_dir"
   git branch "backup/pre-migration-$stamp" HEAD
   git bundle create "$backup_dir/superproject.bundle" HEAD
   git diff --binary > "$backup_dir/worktree.patch"
   git diff --cached --binary > "$backup_dir/index.patch"
   git ls-files --others --exclude-standard > "$backup_dir/untracked.txt"
   ```

2. Inspect every initialized submodule, including nested ones. Repeat this backup
   example for each modified submodule; copy its listed untracked source too.
   Safety branches/bundles preserve local commits, including detached HEAD work.

   ```bash
   git submodule foreach --recursive 'git status --short; git log -1 --oneline'
   git -C external/robomimic branch "backup/pre-migration-$stamp" HEAD
   git -C external/robomimic bundle create "$backup_dir/robomimic.bundle" HEAD
   git -C external/robomimic diff --binary > "$backup_dir/robomimic-worktree.patch"
   git -C external/robomimic diff --cached --binary > "$backup_dir/robomimic-index.patch"
   git -C external/robomimic ls-files --others --exclude-standard > "$backup_dir/robomimic-untracked.txt"
   ```

3. After backups, commit selected source changes locally or stash uncommitted work
   in each affected repository. `git stash push --include-untracked` is an option;
   a superproject stash does not save submodule work. Record each stash commit ID with `git rev-parse stash`.
   Reconcile local submodule commits/gitlink changes deliberately; do not check
   them out away merely to make the status clean. Proceed with a clean merge input.

4. Once the PRs are merged, fetch and merge the updated release on your feature
   branch. Resolve conflicts by combining both branches' intended behavior.

   ```bash
   git fetch origin
   git merge origin/release
   git status
   ```

   Update old paths using the table. Stage only reviewed resolutions and use
   `git merge --continue` if a conflict resolution is pending. Do not rebase or
   force-push shared history. Preserve teammates' new functionality.

5. Inspect `git diff --submodule=log` and `git submodule status` before syncing.
   Check each submodule's working tree and backed-up local commits against the
   resolved gitlink. Only after changes are safe and references are resolved:

   ```bash
   git submodule sync --recursive
   git submodule update --init --recursive
   ```

   Reapply stashes with `git stash apply <stash-commit-id>` in their own
   repositories; retain them until verified. Adapt old imports in restored
   work too. Reconcile dependency commits in their fork before changing gitlinks.

6. Run affected CLI `--help`, canonical imports, and
   `python -m unittest discover -s tests`. Set `RBY1_TEST_CONFIG` to the external
   faucet task JSON to include simulator guards. Use a one-demo run,
   prepared-dataset validation, or offline checkpoint check; respect model/display requirements.

## Coding agents

Read this guide and `rby1/README.md` first. Use canonical `rby1.*` imports;
do not recreate compatibility wrappers. Preserve feature-branch modifications
and inspect submodule references before changing dependencies. Preserve policy
action representations, coordinate frames and observation contracts. Keep
experiment configs and generated artifacts outside Git. Change generic MobiPi
functionality only when necessary for the task.
