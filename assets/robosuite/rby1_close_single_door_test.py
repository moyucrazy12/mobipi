"""Deterministic scripted CloseSingleDoor test for the RoboCasa RB-Y1.

The left arm follows fixed joint-space waypoints while the right arm, torso,
head, and mobile base remain parked. No learned policy or runtime IK is used.

How the waypoints were chosen:
I first parked the unused arm and moved the left shoulder into open space. I
then stepped the arm through collision-checked poses until the open gripper
lined up with the physical handle bar rather than the handle object's origin.
The closing poses came from sampling the handle along the hinge arc, solving
for nearby joint configurations, and adjusting them in simulation to keep the
fingers off the door trim. These are intentionally simple, layout-specific
test values, not a general motion planner.

Run with the interactive MuJoCo viewer (the default):

    python assets/robosuite/rby1_close_single_door_test.py

Run headless:

    python assets/robosuite/rby1_close_single_door_test.py --no-render
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from types import MethodType

import numpy as np
import robocasa

# Ensure RBY1 is registered when this script is run directly.
import robosuite.models.robots.manipulators.rby1_robot  # noqa: F401
from robosuite.controllers import load_composite_controller_config


# 26-D RB-Y1 action partition:
#   0:7   right arm (parked by this script)
#   7:14  left arm (7 absolute joint-position targets, radians)
#   14:20 torso (parked)
#   20:22 head (parked)
#   22:24 base (parked two-wheel velocity command)
#   24     right gripper (parked closed)
#   25     left gripper (-1 open, +1 close)
EXPECTED_ACTION_PARTITION = OrderedDict(
    [
        ("right", (0, 7)),
        ("left", (7, 14)),
        ("torso", (14, 20)),
        ("head", (20, 22)),
        ("base", (22, 24)),
        ("right_gripper", (24, 25)),
        ("left_gripper", (25, 26)),
    ]
)

SEED = 11
LAYOUT_ID = 0
STYLE_ID = 0
DOOR_FIXTURE = "cab_1_main_group"

# Keep the robot stationary, but start 5 cm farther from the counter than the
# previous trajectory. The x offset gives the left elbow clearance from layout
# 0's left wall.
ROBOT_BASE_POSITION = np.array([0.70, -1.10, 0.0])
ROBOT_BASE_ORIENTATION = np.array([0.0, 0.0, np.pi / 2])
STATIONARY_BASE_INERTIA_SCALE = 100.0


# Main tuning surface. These collision-checked targets are for layout 0,
# style 0, cab_1_main_group, and the fixed base pose above. The first targets
# lift the arm into free space. The final approach uses the staggered phase
# arrays below so several joints move together without sweeping the elbow
# through the left wall or upper cabinet doors.
LEFT_ARM_APPROACH_WAYPOINTS = OrderedDict(
    [
        ("park left arm", np.array([1.0, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0])),
        ("raise behind counter", np.array([2.0, 0.8, 0.0, -1.2, 0.0, 0.1, 0.0])),
        (
            "clear left wall",
            np.array([2.35618000, 0.8, 0.0, -1.2, 0.0, 0.1, 0.0]),
        ),
        (
            "near handle",
            np.array(
                [
                    2.25004234,
                    0.56720719,
                    1.54993574,
                    -1.75093856,
                    -2.40772740,
                    1.23985670,
                    0.10873634,
                ]
            ),
        ),
        (
            "contact handle",
            np.array(
                [
                    2.35618000,
                    0.77341182,
                    1.82397071,
                    -1.90104706,
                    -2.19233291,
                    0.83652691,
                    -0.04235979,
                ]
            ),
        ),
    ]
)

# This intermediate keeps the elbow clear of the room's left wall before the
# wrist turns toward the handle.
LEFT_ARM_CLEARANCE_WAYPOINT = np.array(
    [
        2.35618000,
        0.29287770,
        1.66313013,
        -1.97371976,
        -2.23660403,
        0.10000000,
        0.00000000,
    ]
)
CLEARANCE_PHASE_START = np.array([0.00, 0.00, 0.20, 0.45, 0.70, 0.00, 0.00])
CLEARANCE_PHASE_END = np.array([1.00, 0.25, 0.50, 0.75, 1.00, 1.00, 1.00])

# A straight Cartesian pregrasp-to-handle line was sampled offline, then
# converted to these joint-space points. Small segments prevent the arm from
# bowing into the door trim between the collision-free endpoints.
LEFT_ARM_INSERT_WAYPOINTS = OrderedDict(
    [
        (
            "insert 1/8",
            np.array(
                [
                    2.26376607,
                    0.58327589,
                    1.57877126,
                    -1.78466819,
                    -2.38307495,
                    1.18800066,
                    0.09045439,
                ]
            ),
        ),
        (
            "insert 2/8",
            np.array(
                [
                    2.27707386,
                    0.60220708,
                    1.60931100,
                    -1.81410066,
                    -2.35742906,
                    1.13652830,
                    0.07208492,
                ]
            ),
        ),
        (
            "insert 3/8",
            np.array(
                [
                    2.29006324,
                    0.62396964,
                    1.64149719,
                    -1.83926342,
                    -2.33095565,
                    1.08541077,
                    0.05357984,
                ]
            ),
        ),
        (
            "insert 4/8",
            np.array(
                [
                    2.30284956,
                    0.64852983,
                    1.67525638,
                    -1.86016553,
                    -2.30384246,
                    1.03463930,
                    0.03490553,
                ]
            ),
        ),
        (
            "insert 5/8",
            np.array(
                [
                    2.31556993,
                    0.67584551,
                    1.71050383,
                    -1.87680423,
                    -2.27630159,
                    0.98422843,
                    0.01604520,
                ]
            ),
        ),
        (
            "insert 6/8",
            np.array(
                [
                    2.32838664,
                    0.70586059,
                    1.74715056,
                    -1.88917065,
                    -2.24856987,
                    0.93421591,
                    -0.00299878,
                ]
            ),
        ),
        (
            "insert 7/8",
            np.array(
                [
                    2.34149061,
                    0.73850007,
                    1.78511258,
                    -1.89725433,
                    -2.22090874,
                    0.88466250,
                    -0.02220122,
                ]
            ),
        ),
        (
            "insert 8/8",
            np.array(
                [
                    2.35510547,
                    0.77366548,
                    1.82432205,
                    -1.90104663,
                    -2.19360420,
                    0.83565160,
                    -0.04150891,
                ]
            ),
        ),
    ]
)

# These targets follow the handle's closing arc. The suffix is the intended
# remaining open fraction; execution stops as soon as RoboCasa reports success.
LEFT_ARM_CLOSE_WAYPOINTS = OrderedDict(
    [
        (
            "close_1.00",
            np.array(
                [
                    2.35618000,
                    0.83429631,
                    1.80340652,
                    -1.81935931,
                    -2.22011645,
                    0.86242375,
                    -0.03042316,
                ]
            ),
        ),
        (
            "close_0.98",
            np.array(
                [
                    2.34475568,
                    0.83124120,
                    1.82427798,
                    -1.85163905,
                    -2.21816207,
                    0.86919161,
                    -0.04234648,
                ]
            ),
        ),
        (
            "close_0.96",
            np.array(
                [
                    2.33350994,
                    0.83016297,
                    1.84387650,
                    -1.88098755,
                    -2.21714751,
                    0.87760365,
                    -0.05350920,
                ]
            ),
        ),
        (
            "close_0.95",
            np.array(
                [
                    2.33609245,
                    0.82817780,
                    1.85100214,
                    -1.89466009,
                    -2.20787196,
                    0.88885478,
                    -0.06500433,
                ]
            ),
        ),
        (
            "close_0.90",
            np.array(
                [
                    2.30938560,
                    0.83302652,
                    1.89239054,
                    -1.95444746,
                    -2.21155956,
                    0.91978301,
                    -0.08896314,
                ]
            ),
        ),
        (
            "close_0.85",
            np.array(
                [
                    2.28193057,
                    0.84290500,
                    1.92471515,
                    -2.00297119,
                    -2.22433050,
                    0.96038023,
                    -0.10710656,
                ]
            ),
        ),
        (
            "close_0.833",
            np.array(
                [
                    2.27367231,
                    0.88220032,
                    1.94022319,
                    -2.02434026,
                    -2.20172781,
                    0.99193829,
                    -0.07602272,
                ]
            ),
        ),
        (
            "close_0.817",
            np.array(
                [
                    2.26380076,
                    0.92539878,
                    1.95273643,
                    -2.03981189,
                    -2.17940713,
                    1.02534460,
                    -0.04613093,
                ]
            ),
        ),
        (
            "close_0.80",
            np.array(
                [
                    2.25158774,
                    0.97215745,
                    1.96176251,
                    -2.04996178,
                    -2.15793500,
                    1.06127728,
                    -0.01622436,
                ]
            ),
        ),
        (
            "close_0.75",
            np.array(
                [
                    2.20909326,
                    1.11946925,
                    1.99996190,
                    -2.09544983,
                    -2.06737675,
                    1.14527522,
                    0.04854056,
                ]
            ),
        ),
        (
            "close_0.70",
            np.array(
                [
                    2.16219978,
                    1.30108554,
                    2.03742426,
                    -2.12585767,
                    -1.96384174,
                    1.22000183,
                    0.08028330,
                ]
            ),
        ),
        (
            "close_0.55",
            np.array(
                [
                    1.83926490,
                    1.77274467,
                    2.07017248,
                    -2.18720323,
                    -1.96192214,
                    0.84641332,
                    0.25548672,
                ]
            ),
        ),
        (
            "close_0.40",
            np.array(
                [
                    1.70089335,
                    2.22270706,
                    2.09073309,
                    -2.08053946,
                    -1.80522124,
                    0.51642346,
                    0.19891226,
                ]
            ),
        ),
        (
            "close_0.25",
            np.array(
                [
                    1.61069587,
                    2.62037820,
                    2.08767672,
                    -1.81086591,
                    -1.49775985,
                    0.30307993,
                    -0.06746956,
                ]
            ),
        ),
        (
            "close_0.12",
            np.array(
                [
                    1.44383983,
                    2.89524140,
                    1.93319849,
                    -1.45126320,
                    -1.18011741,
                    0.29240238,
                    -0.26408467,
                ]
            ),
        ),
        (
            "close_0.05",
            np.array(
                [
                    1.35697493,
                    3.04169403,
                    1.82812581,
                    -1.26416553,
                    -1.01376510,
                    0.29805136,
                    -0.32173078,
                ]
            ),
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable mjviewer for headless validation.",
    )
    parser.add_argument(
        "--stage-steps",
        type=int,
        default=80,
        help="Control steps used to blend between arm waypoints (default: 80).",
    )
    return parser.parse_args()


def make_env(render: bool):
    return robocasa.make(
        env_name="CloseSingleDoor",
        robots="RBY1",
        door_id=DOOR_FIXTURE,
        controller_configs=load_composite_controller_config(robot="RBY1"),
        seed=SEED,
        layout_ids=[LAYOUT_ID],
        style_ids=[STYLE_ID],
        force_robot_placement=(ROBOT_BASE_POSITION, ROBOT_BASE_ORIENTATION),
        initialization_noise=None,
        randomize_base_init_pose=None,
        randomize_cameras=False,
        use_distractors=False,
        use_camera_obs=False,
        has_renderer=render,
        has_offscreen_renderer=False,
        renderer="mjviewer" if render else "mujoco",
        render_camera=None,
        control_freq=20,
        ignore_done=True,
        hard_reset=False,
    )


def assert_action_partition(env) -> None:
    robot = env.robots[0]
    actual = OrderedDict(robot._action_split_indexes)
    if env.action_dim != 26 or actual != EXPECTED_ACTION_PARTITION:
        raise RuntimeError(
            "Unexpected RB-Y1 action space. Refusing hard-coded commands.\n"
            f"Expected: dim=26, partition={dict(EXPECTED_ACTION_PARTITION)}\n"
            f"Actual:   dim={env.action_dim}, partition={dict(actual)}"
        )
    if robot.part_controllers["left"].input_type != "absolute":
        raise RuntimeError(
            "This script requires the left JOINT_POSITION controller in absolute mode"
        )
    print(f"RB-Y1 action partition: {dict(actual)}")


def set_exact_open_start(env) -> None:
    """Remove the task's 90-100% reset randomization."""
    door = env.door_fxtr
    door.set_door_state(min=1.0, max=1.0, env=env, rng=env.rng)
    hinge_id = env.sim.model.joint_name2id(f"{door.name}_doorhinge")
    env.sim.data.qvel[env.sim.model.jnt_dofadr[hinge_id]] = 0.0
    env.sim.forward()


def ensure_named_joint_door_state(env) -> None:
    """Keep RoboCasa's success check correct when RB-Y1 has a free root joint."""
    door = env.door_fxtr
    joint_name = f"{door.name}_doorhinge"
    sign = -1 if door.orientation == "left" else 1

    def get_door_state_by_name(self, env):
        hinge_qpos = float(env.sim.data.get_joint_qpos(joint_name))
        return {"door": hinge_qpos * sign / (np.pi / 2)}

    named_state = get_door_state_by_name(door, env)["door"]
    reported_state = door.get_door_state(env)["door"]
    if not np.isclose(named_state, reported_state):
        # Some RoboCasa versions index qpos by joint id. That stops working when
        # a free joint precedes the cabinet joints. Patch this fixture instance
        # only; env._check_success still supplies the task success condition.
        door.get_door_state = MethodType(get_door_state_by_name, door)
        print("Using named-joint cabinet state compatibility for mobile RB-Y1")


def get_left_gripper_qpos(env) -> float:
    joint_name = env.robots[0].gripper["left"].joints[0]
    return float(env.sim.data.get_joint_qpos(joint_name))


def get_handle_distance(env) -> float:
    """Distance from the left finger-base midpoint to the physical handle bar."""
    model = env.sim.model
    data = env.sim.data
    finger_ids = [
        model.body_name2id("robot0_ee_finger_l1"),
        model.body_name2id("robot0_ee_finger_l2"),
    ]
    finger_midpoint = np.mean([data.body_xpos[idx] for idx in finger_ids], axis=0)
    handle_geom = model.geom_name2id(f"{env.door_fxtr.name}_door_handle_handle")
    return float(np.linalg.norm(finger_midpoint - data.geom_xpos[handle_geom]))


class ScriptRunner:
    def __init__(self, env, render: bool):
        self.env = env
        self.robot = env.robots[0]
        self.render = render
        self.left_qpos_index = np.asarray(
            self.robot.part_controllers["left"].qpos_index
        )
        self.right_qpos_index = np.asarray(
            self.robot.part_controllers["right"].qpos_index
        )
        self.commanded_left_qpos = env.sim.data.qpos[self.left_qpos_index].copy()
        self.minimum_handle_distance = float("inf")
        self.unintended_contacts: set[str] = set()
        self.unintended_contact_pairs: set[str] = set()
        self.handle_contact_observed = False

        # This task isolates arm plumbing from unfinished mobile-base dynamics.
        # If the model has a free chassis joint, give the base high stationary
        # inertia so arm reactions do not roll or tip it. Wheel actions remain
        # zero and wheel code is not changed by this script.
        try:
            env.sim.model.joint_name2id("robot0_world_j")
        except ValueError:
            pass
        else:
            base_body_id = env.sim.model.body_name2id("robot0_base")
            env.sim.model.body_mass[base_body_id] *= STATIONARY_BASE_INERTIA_SCALE
            env.sim.model.body_inertia[
                base_body_id
            ] *= STATIONARY_BASE_INERTIA_SCALE
            env.sim.forward()
            print("Stationary test mode: RB-Y1 chassis stabilized")

        self.base_body_id = env.sim.model.body_name2id("robot0_base")
        self.initial_base_pos = env.sim.data.body_xpos[self.base_body_id].copy()
        self.initial_base_ori = env.sim.data.body_xmat[self.base_body_id].reshape(
            3, 3
        ).copy()

        # Hold the right arm at reset so it stays below and outside the cabinet.
        self.held_parts = {
            name: env.sim.data.qpos[controller.qpos_index].copy()
            for name, controller in self.robot.part_controllers.items()
            if name in {"right", "torso", "head"}
        }
        self.held_parts.update(
            {
                "base": np.zeros(2),
                "right_gripper": np.array([1.0]),
            }
        )

    def action(self, left_qpos: np.ndarray, gripper: float) -> np.ndarray:
        return self.robot.create_action_vector(
            {
                **self.held_parts,
                "left": left_qpos,
                "left_gripper": np.array([gripper]),
            }
        )

    def _record_unintended_contacts(self) -> None:
        model = self.env.sim.model
        for idx in range(self.env.sim.data.ncon):
            contact = self.env.sim.data.contact[idx]
            names = {
                model.geom_id2name(contact.geom1) or "",
                model.geom_id2name(contact.geom2) or "",
            }
            robot_names = {name for name in names if name.startswith("robot0_")}
            if not robot_names or len(robot_names) == len(names):
                continue
            other = next(name for name in names if name not in robot_names)
            if other.startswith("floor_"):
                continue
            if other == f"{self.env.door_fxtr.name}_door_handle_handle":
                self.handle_contact_observed = True
                continue
            self.unintended_contacts.add(other)
            for robot_name in robot_names:
                self.unintended_contact_pairs.add(f"{robot_name} <-> {other}")

    def step(self, left_qpos: np.ndarray, gripper: float) -> bool:
        self.env.step(self.action(left_qpos, gripper))
        if self.render:
            self.env.render()
        self.minimum_handle_distance = min(
            self.minimum_handle_distance, get_handle_distance(self.env)
        )
        self._record_unintended_contacts()
        return bool(self.env._check_success())

    def move_to(
        self,
        name: str,
        target: np.ndarray,
        gripper: float,
        steps: int,
        stop_on_success: bool = False,
    ) -> bool:
        start = self.commanded_left_qpos.copy()
        success = False
        contacts_before = self.unintended_contacts.copy()
        for step in range(steps):
            u = (step + 1) / steps
            blend = u * u * (3.0 - 2.0 * u)
            command = (1.0 - blend) * start + blend * target
            success = self.step(command, gripper)
            if stop_on_success and success:
                break
        self.commanded_left_qpos = target.copy()
        print(
            f"{name:>22}: door={self.env.door_fxtr.get_door_state(self.env)}, "
            f"handle_distance={get_handle_distance(self.env):.3f} m, "
            f"success={success}"
        )
        new_contacts = self.unintended_contacts - contacts_before
        if new_contacts:
            print(f"{'unexpected contact':>22}: {sorted(new_contacts)}")
        return success

    def move_staggered(
        self,
        name: str,
        target: np.ndarray,
        gripper: float,
        steps: int,
        phase_start: np.ndarray,
        phase_end: np.ndarray,
    ) -> bool:
        """Blend all arm joints continuously with collision-safe phase offsets."""
        start = self.commanded_left_qpos.copy()
        success = False
        contacts_before = self.unintended_contacts.copy()
        for step in range(steps):
            progress = (step + 1) / steps
            phase = np.clip(
                (progress - phase_start) / (phase_end - phase_start), 0.0, 1.0
            )
            blend = phase * phase * (3.0 - 2.0 * phase)
            success = self.step(start + (target - start) * blend, gripper)
        self.commanded_left_qpos = target.copy()
        print(
            f"{name:>22}: door={self.env.door_fxtr.get_door_state(self.env)}, "
            f"handle_distance={get_handle_distance(self.env):.3f} m, "
            f"success={success}"
        )
        new_contacts = self.unintended_contacts - contacts_before
        if new_contacts:
            print(f"{'unexpected contact':>22}: {sorted(new_contacts)}")
        return success

    def hold_gripper(self, name: str, value: float, steps: int) -> bool:
        success = False
        contacts_before = self.unintended_contacts.copy()
        for _ in range(steps):
            success = self.step(self.commanded_left_qpos, value)
        print(f"{name:>22}: gripper_qpos={get_left_gripper_qpos(self.env):.4f} m")
        new_contacts = self.unintended_contacts - contacts_before
        if new_contacts:
            print(f"{'unexpected contact':>22}: {sorted(new_contacts)}")
        return success


def main() -> int:
    args = parse_args()
    if args.stage_steps < 10:
        raise ValueError("--stage-steps must be at least 10")

    env = make_env(render=not args.no_render)
    try:
        env.reset()
        assert_action_partition(env)
        ensure_named_joint_door_state(env)
        set_exact_open_start(env)
        if env._check_success():
            raise RuntimeError("CloseSingleDoor started successful; open setup failed")
        print(f"Initial door state: {env.door_fxtr.get_door_state(env)}")

        runner = ScriptRunner(env, render=not args.no_render)
        initial_left_qpos = env.sim.data.qpos[runner.left_qpos_index].copy()
        initial_right_qpos = env.sim.data.qpos[runner.right_qpos_index].copy()

        runner.move_to(
            "starting posture",
            LEFT_ARM_APPROACH_WAYPOINTS["park left arm"],
            gripper=-1.0,
            steps=args.stage_steps,
        )
        left_joint_motion = np.abs(
            env.sim.data.qpos[runner.left_qpos_index] - initial_left_qpos
        )

        runner.hold_gripper("gripper close test", value=1.0, steps=20)
        closed_qpos = get_left_gripper_qpos(env)
        runner.hold_gripper("gripper open test", value=-1.0, steps=20)
        open_qpos = get_left_gripper_qpos(env)
        gripper_motion = abs(open_qpos - closed_qpos)

        for name in ("raise behind counter", "clear left wall"):
            runner.move_to(
                name,
                LEFT_ARM_APPROACH_WAYPOINTS[name],
                gripper=-1.0,
                steps=args.stage_steps,
            )
        runner.move_staggered(
            "clearance arc",
            LEFT_ARM_CLEARANCE_WAYPOINT,
            gripper=-1.0,
            steps=args.stage_steps * 3,
            phase_start=CLEARANCE_PHASE_START,
            phase_end=CLEARANCE_PHASE_END,
        )
        runner.move_to(
            "near handle",
            LEFT_ARM_APPROACH_WAYPOINTS["near handle"],
            gripper=-1.0,
            steps=args.stage_steps,
        )
        left_tracking_error = np.abs(
            env.sim.data.qpos[runner.left_qpos_index]
            - LEFT_ARM_APPROACH_WAYPOINTS["near handle"]
        )
        insertion_steps = max(10, args.stage_steps // 4)
        for name, target in LEFT_ARM_INSERT_WAYPOINTS.items():
            runner.move_to(name, target, gripper=-1.0, steps=insertion_steps)
        runner.move_to(
            "contact handle",
            LEFT_ARM_APPROACH_WAYPOINTS["contact handle"],
            gripper=-1.0,
            steps=insertion_steps,
        )
        runner.hold_gripper("grasp handle", value=1.0, steps=30)
        grasp_qpos = get_left_gripper_qpos(env)
        handle_captured = (
            runner.handle_contact_observed and grasp_qpos < closed_qpos - 0.005
        )
        for name in ("insert 7/8", "insert 6/8", "insert 5/8", "insert 4/8"):
            runner.move_to(
                f"retract via {name}",
                LEFT_ARM_INSERT_WAYPOINTS[name],
                gripper=1.0,
                steps=insertion_steps,
            )

        success = bool(env._check_success())
        for name, target in LEFT_ARM_CLOSE_WAYPOINTS.items():
            if success:
                break
            success = runner.move_to(
                name,
                target,
                gripper=1.0,
                steps=args.stage_steps,
                stop_on_success=True,
            )

        right_arm_drift = np.max(
            np.abs(env.sim.data.qpos[runner.right_qpos_index] - initial_right_qpos)
        )
        final_base_pos = env.sim.data.body_xpos[runner.base_body_id].copy()
        final_base_ori = (
            env.sim.data.body_xmat[runner.base_body_id].reshape(3, 3).copy()
        )
        base_translation = np.linalg.norm(final_base_pos - runner.initial_base_pos)
        relative_base_ori = runner.initial_base_ori.T @ final_base_ori
        base_tilt = np.arccos(
            np.clip((np.trace(relative_base_ori) - 1.0) / 2.0, -1.0, 1.0)
        )
        arm_ok = np.linalg.norm(left_joint_motion) > 0.05 and np.all(
            left_tracking_error < 0.10
        )
        gripper_ok = gripper_motion > 0.005
        reachable = runner.minimum_handle_distance < 0.08
        right_parked = right_arm_drift < 0.10
        base_stable = base_translation < 0.05 and base_tilt < 0.10
        collision_free = not runner.unintended_contacts

        print(f"Left arm moved correctly: {arm_ok}")
        print(
            "Left-arm near-handle tracking error: "
            f"{np.round(left_tracking_error, 3)} rad"
        )
        print(f"Left gripper opened/closed: {gripper_ok}")
        print(
            "Handle captured without fully closing: "
            f"{handle_captured} (gripper qpos {grasp_qpos:.4f} m)"
        )
        print(
            "Cabinet handle reachable: "
            f"{reachable} (minimum distance {runner.minimum_handle_distance:.3f} m)"
        )
        print(
            "Right arm remained parked: "
            f"{right_parked} (drift {right_arm_drift:.3f} rad)"
        )
        print(
            "Base remained stable: "
            f"{base_stable} (translation {base_translation:.3f} m, "
            f"rotation {base_tilt:.3f} rad)"
        )
        print(
            "Unintended environment contacts: "
            f"{sorted(runner.unintended_contact_pairs)}"
        )
        print(f"RoboCasa task success: {success}")
        passed = (
            arm_ok
            and gripper_ok
            and handle_captured
            and reachable
            and right_parked
            and base_stable
            and collision_free
        )
        return 0 if passed and success else 1
    finally:
        print("Done")


if __name__ == "__main__":
    raise SystemExit(main())
