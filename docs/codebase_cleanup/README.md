# Codebase cleanup report

Branch: **`codebase-cleanup`**, based on merged `origin/release`
**`cf9868d5cc4471f573a2691b8849ff13ab047344`**. Work was performed in the clean
`../mobipi-codebase-cleanup` worktree. The original dirty faucet checkout and
SDK/probe files were not changed. Do not merge this branch automatically.

## Evidence and migration

- [History/contribution audit and dependency map](audit.md), with full
  [team commit records](history.json) and pre-move [dependency pins](baseline.json).
- [File/function migration map](migration.md): 41 relocations, with 26 legacy
  module redirects, 8 test-discovery shims, 1 task-package re-export and 6
  configuration symlinks. Old public imports, private helpers used by callers,
  dynamic entry points and CLI paths remain available.
- [Regression matrix and manual procedures](regression.md), plus compact
  [numerical comparison evidence](equivalence.json).
- [Practical entry points and configuration guide](../../rby1/README.md).

The resulting `rby1/` package contains `configs`, `demos`, `policies`, `pipeline`,
`robot` and `tests`, plus a lightweight path module and shared compatibility
loader. Generic MobiPi functionality remains in its existing framework modules.
Asset/installer paths and native standalone dependency tools remain in place.

Eleven AST-identical function duplicates were consolidated: ten simulator/pose
helpers and the shared IK-reset helper. Other existing controller adapters were
moved unchanged from the microwave planner into `rby1/robot/controllers.py`;
both planners retain public re-exports. No diagnostic tools or planners were
deleted. General-door, model-import and mobilization experiments are retained
with their limitations documented. Cross-repository `experiment_root` functions
remain separate because their module-global roots and installation assumptions
differ. [Full duplicate scan](duplicates_extended.json) covers 108 relevant
baseline Python/JSON files and found no exact nonempty file duplicates.

## Dependency revisions

| Dependency | Baseline | After cleanup |
|---|---|---|
| RoboMimic | `c595ab8927f870cb47dc04618eb31ab943becc69` | `657f38cffc555d4de478a474f2412b1863ddac9f` — test-only source-inspection compatibility |
| MimicGen | `edbaff93d5e988213a95f53fbbbbc99cecc01921` | unchanged |
| RoboCasa | `3683fb01d6fc87d7849e7b5a886d01f4b1d7a55d` | unchanged |
| Diffusion Policy | `5ba07ac6661db573af695b419a7947ecb704690f` | unchanged |
| LeLaN | `8a84208be913b3838f2e550929d39cd0d674b252` | unchanged |
| SPHINX | `a8e6142cb7cf4670305c9ac4834a68b5273412b7` | unchanged |

Companion [RoboMimic PR #7](https://github.com/moyucrazy12/mobipi-robomimic/pull/7)
changes only `tests/test_rby1_paths.py`. Production dependency code is unchanged.
Use a normal reviewed merge so the pinned commit stays reachable. Neither PR
is merged by this task.

## Verification and limits

- Before moves: 90 tests passed, 21 historical CLI checks passed, scripted
  faucet collection and native MimicGen generation each succeeded 1/1, real
  checkpoint load/inference passed.
- After cleanup: 104 tests run, **103 passed and 1 skipped**; 51 legacy discovery
  tests also passed. All 21 CLI checks pass; help is equivalent after replacing
  moved module names in examples. Direct legacy scripts work from another cwd.
- Scripted collection, native source preparation and native generation reproduce
  all 38, 47 and 48 stored arrays respectively **bit-for-bit**, including images.
  The same seeded checkpoint returns the same action/queue. A bounded three-step
  rollout reproduces control/actions and physical metrics, excluding timing and
  output paths. No statistical-performance claim follows from these checks.
- Six task/policy configuration files retain their exact baseline bytes.
  Wheel building/inspection confirms canonical packages, configuration data,
  test fixtures and legacy namespace modules are included.

The installed faucet-model environment lacks `RBY1MobileBase`; its virtual-base
test fails identically on baseline/refactor without a prerequisite guard and
is now explicitly skipped. Prepare the documented separate mobile environment
before running that test or the full navigation pipeline. No installed packages
or model profiles were silently replaced. DINO/3DGS execution, full door/drawer
script success, optimizer-state resume, long training and statistical policy
grids remain unverified. Historical/default config artifact paths sometimes
require explicit overrides. Existing evaluation manifests fingerprint source
code; start new evaluation output directories after moving implementations.

## Incremental history

Commits record the audit before any moves, then demo/policy/config migration,
robot/pipeline migration, exact-helper consolidation, packaging, test relocation,
controller extraction and final documentation. Get the authoritative complete
branch history with:

```bash
git log --reverse --oneline cf9868d5cc4471f573a2691b8849ff13ab047344..codebase-cleanup
git diff --stat cf9868d5cc4471f573a2691b8849ff13ab047344..codebase-cleanup
```

SDK integration remains a separate task. Physical RB-Y1 hardware revision is
not inferred from the simulation model filename.
