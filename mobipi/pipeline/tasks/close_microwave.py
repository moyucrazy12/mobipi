"""RoboCasa CloseSingleDoor with the microwave door; progress is the hinge angle."""

from mobipi.pipeline.tasks.base import ROOT, PipelineTask


class CloseMicrowave(PipelineTask):
    name = "close_microwave"
    env_name = "CloseSingleDoor"
    default_checkpoint = ROOT / "ckpts/model_115_cartesian.pth"
    default_dataset = ROOT / "data/mimicgen/CloseDoorLeft_RBY1_115mg/demo_im128_head_wrist_clean37.hdf5"
    progress_label = "hinge rad"
    success_definition = "Navigation, stable door closure, and arm return all succeed"

    @property
    def success_threshold(self):
        from robomimic.utils.rby1_evaluation import SUCCESS_THRESHOLD_RAD
        return SUCCESS_THRESHOLD_RAD

    def env_overrides(self, kwargs, in_training_layout):
        if not in_training_layout:
            from robocasa.models.fixtures import FixtureType
            kwargs["door_id"] = FixtureType.MICROWAVE

    def progress_fn(self, env):
        from robomimic.scripts.replay_rby1_cartesian import door_geometry

        hinge_name, hinge_address, _ = door_geometry(env, env.get_ep_meta())
        if "microjoint" not in hinge_name or env.behavior != "close":
            raise ValueError("This checkpoint expects the microwave-closing task")
        return lambda: float(env.sim.data.qpos[hinge_address])
