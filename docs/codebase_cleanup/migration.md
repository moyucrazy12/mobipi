# File migration map

Old paths remain usable. Python modules redirect to the canonical module object; configuration paths are relative symlinks to one source. Test discovery and task-package initializers re-export symbols without redirecting their package search paths.

| Old path | Canonical path | Compatibility | Validation |
|---|---|---|---|
| `mobipi/collect_demos/auto_close_drawer.py` | `rby1/demos/auto_close_drawer.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/auto_close_door_general.py` | `rby1/demos/auto_close_door_general.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/collect_close_drawer_demos.py` | `rby1/demos/collect_close_drawer_demos.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/faucet_runtime.py` | `rby1/robot/faucet_runtime.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/faucet_progress.py` | `rby1/demos/faucet_progress.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/experiment_paths.py` | `rby1/paths.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/faucet_dataset.py` | `rby1/demos/faucet_dataset.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/gather_demos_hdf5.py` | `rby1/demos/gather_demos_hdf5.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/evaluate_faucet_dp.py` | `rby1/policies/evaluate_faucet_dp.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/prepare_faucet_dp.py` | `rby1/policies/prepare_faucet_dp.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/faucet_bounded_ik.py` | `rby1/robot/faucet_bounded_ik.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/pipeline_tasks.py` | `rby1/pipeline/collection_tasks.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/collect_close_door_demos.py` | `rby1/demos/collect_close_door_demos.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/collect_faucet_demos.py` | `rby1/demos/collect_faucet_demos.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/pipeline.py` | `rby1/pipeline/collect.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/report_faucet_evaluation.py` | `rby1/policies/report_faucet_evaluation.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/faucet_pipeline.py` | `rby1/demos/faucet_pipeline.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/auto_close_microwave_door.py` | `rby1/demos/auto_close_microwave_door.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/collect_demos/faucet_relative_cartesian_dp.json` | `rby1/configs/faucet_relative_cartesian_dp.json` | relative symlink | byte hash equals release; legacy config loading |
| `mobipi/collect_demos/faucet_relative_cartesian_dp_extended.json` | `rby1/configs/faucet_relative_cartesian_dp_extended.json` | relative symlink | byte hash equals release; legacy config loading |
| `mobipi/collect_demos/faucet_config.json` | `rby1/configs/faucet_config.json` | relative symlink | byte hash equals release; legacy config loading |
| `mobipi/collect_demos/faucet_mimicgen_diverse_config.json` | `rby1/configs/faucet_mimicgen_diverse_config.json` | relative symlink | byte hash equals release; legacy config loading |
| `mobipi/utils/rby1_utils.py` | `rby1/robot/navigation.py` | module redirect | seeded navigation references; collision handoff; arm-return contracts |
| `mobipi/envs/kitchen_drawer.py` | `rby1/robot/kitchen_drawer.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/scripts/run_rby1_pipeline.py` | `rby1/pipeline/run.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/scripts/evaluate_rby1_pipeline.py` | `rby1/pipeline/evaluate.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/eval/test_mobilization.py` | `rby1/pipeline/test_mobilization.py` | module redirect | source retained; full experimental execution remains manual |
| `mobipi/pipeline/tasks/close_microwave.py` | `rby1/pipeline/tasks/close_microwave.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/pipeline/tasks/close_drawer.py` | `rby1/pipeline/tasks/close_drawer.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/pipeline/tasks/base.py` | `rby1/pipeline/tasks/base.py` | module redirect | module identity; CLI checks; task/dataset regression suite |
| `mobipi/pipeline/tasks/__init__.py` | `rby1/pipeline/tasks/__init__.py` | package symbol re-export | task registry and child-module identity |
| `configs/mimicgen/CloseDrawerLeft_RBY1_300mg.json` | `rby1/configs/mimicgen/CloseDrawerLeft_RBY1_300mg.json` | relative symlink | byte hash equals release; legacy config loading |
| `configs/mimicgen/CloseSingleDoor_RBY1_surface_layout1_115mg.json` | `rby1/configs/mimicgen/CloseSingleDoor_RBY1_surface_layout1_115mg.json` | relative symlink | byte hash equals release; legacy config loading |
| `mobipi/collect_demos/tests/test_faucet_runtime.py` | `rby1/tests/test_faucet_runtime.py` | discovery re-export | 51 legacy tests; canonical discovery |
| `mobipi/collect_demos/tests/test_pipeline.py` | `rby1/tests/test_pipeline.py` | discovery re-export | 51 legacy tests; canonical discovery |
| `mobipi/collect_demos/tests/test_faucet_progress.py` | `rby1/tests/test_faucet_progress.py` | discovery re-export | 51 legacy tests; canonical discovery |
| `mobipi/collect_demos/tests/test_release_integration.py` | `rby1/tests/test_release_integration.py` | discovery re-export | 51 legacy tests; canonical discovery |
| `mobipi/collect_demos/tests/test_faucet_dataset.py` | `rby1/tests/test_faucet_dataset.py` | discovery re-export | 51 legacy tests; canonical discovery |
| `mobipi/collect_demos/tests/test_faucet_evaluation_grid.py` | `rby1/tests/test_faucet_evaluation_grid.py` | discovery re-export | 51 legacy tests; canonical discovery |
| `mobipi/collect_demos/tests/test_faucet_reset.py` | `rby1/tests/test_faucet_reset.py` | discovery re-export | 51 legacy tests; canonical discovery |
| `mobipi/collect_demos/tests/test_faucet_bounded_ik.py` | `rby1/tests/test_faucet_bounded_ik.py` | discovery re-export | 51 legacy tests; canonical discovery |

## Function consolidation

| Original functions | Canonical implementation | Why safe / validation |
|---|---|---|
| Both door scripts: `raw_model`, `raw_data`, `forward`, `normalize`, `smoothstep`, `site_pose`, `make_T`, `interpolate_rotation`, `arm_joint_names`, `force_tangent_direction` | `rby1/robot/sim_utils.py` | Identical pre-move ASTs; identical imported globals; re-export identity, fixed seeded numerical references and real recordings |
| Both door scripts: `set_ik_q0_from_current_pose` | `rby1/robot/controllers.py` | Identical ASTs and native WholeBody/NumPy dependencies; both names re-export one implementation |
| Microwave script: `pose6`, `pose_target_for_ref`, `arm_tcp_site`, `arm_default_ref_site`, `whole_body_arm_refs`, `choose_active_arm`, `joint_hold_command`, `build_controller_interface`, `create_robot_action`, `execute_target` | `rby1/robot/controllers.py` | Existing bodies moved unchanged; public names re-exported; left/right controller and TCP round-trip tests; removes robot-adapter dependency on a demo planner |

No scripts or diagnostic tools were deleted. The general-door planner, asset import tests and mobilization experiment remain available. Different `pose6`/controller/contact/planner variants were not merged merely because they look alike.

## Framework files intentionally retained

- `mobipi/scene_model/scene_model.py` and `mobipi/utils/score_utils.py`: shared portability/scoring changes retained.
- Other generic `mobipi` rendering, BO, navigation, policy loading, baselines and visualization: unchanged.
- `assets/robosuite/`: fixed installation/model paths retained, including model import/legacy diagnostic scripts.
- RoboMimic policy networks, absolute/relative Cartesian adapters, normalization, datasets, checkpoint and standalone CLI utilities: retained in the dependency, imported where needed.
- MimicGen native interfaces and task registration: retained in the dependency.
- RoboMimic `tests/test_rby1_paths.py`: the only dependency file changed by this refactor; source inspection follows redirects and uses the shared path module. No production dependency file changed.

Total file/config/test relocations with compatibility paths: **41**.
