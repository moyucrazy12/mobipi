"""Interface between the task-agnostic RB-Y1 pipeline and one manipulation task."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class PipelineTask:
    """
    Everything the pipeline needs to know about one task. Subclass it and add
    the subclass to mobipi.pipeline.tasks.TASKS to run the task with
    run_rby1_pipeline.py / evaluate_rby1_pipeline.py (--task <name>).

    Task completion is a scalar progress (higher is closer to done) that
    succeeds at success_threshold; the pipeline stops the policy once it has
    stayed successful for 0.5 seconds and requires it to stay successful while
    the arm returns.
    """

    name = None  # --task value
    env_name = None  # robosuite environment of the policy's training data
    default_checkpoint = None
    default_dataset = None
    progress_label = "progress"  # for log lines
    success_definition = None  # for evaluation summaries
    success_threshold = 0.0
    # Scoring collision radius as a fraction of the robot radius (0.5 in Mobi-pi)
    collision_threshold_multiplier = 0.5

    def env_overrides(self, kwargs, in_training_layout):
        """Edits the robosuite kwargs from the dataset metadata before the env is made."""

    def scene_overrides(self, env, scene_dir):
        """
        Extra env kwargs that make the simulated scene match the 3DGS capture in
        scene_dir; the pipeline rebuilds the env with them if any are returned.
        """
        return {}

    def progress_fn(self, env):
        """Checks that env is this task and returns a zero-argument callable giving its progress."""
        raise NotImplementedError
