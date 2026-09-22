import numpy as np
import robocasa
import robosuite.models.robots.manipulators.rby1_robot
from types import MethodType
from robosuite.controllers import load_composite_controller_config


# ---------------- Environment ----------------

env = robocasa.make(
    env_name="CloseSingleDoor",
    robots="RBY1",
    door_id="cab_1_main_group",
    controller_configs=load_composite_controller_config(robot="RBY1"),
    seed=11,
    layout_ids=[0],
    style_ids=[0],
    force_robot_placement=(
        np.array([0.70, -1.10, 0.0]),
        np.array([0.0, 0.0, np.pi / 2]),
    ),
    initialization_noise=None,
    randomize_base_init_pose=None,
    randomize_cameras=False,
    use_distractors=False,
    use_camera_obs=False,
    has_renderer=True,
    has_offscreen_renderer=False,
    renderer="mjviewer",
    render_camera=None,
    control_freq=20,
    ignore_done=True,
    hard_reset=False,
)

env.reset()
robot = env.robots[0]


# ---------------- Fix deterministic start ----------------

door = env.door_fxtr
door.set_door_state(min=1.0, max=1.0, env=env, rng=env.rng)
env.sim.forward()

# Keep the mobile base basically fixed during this manipulation test.
base_id = env.sim.model.body_name2id("robot0_base")
env.sim.model.body_mass[base_id] *= 100
env.sim.model.body_inertia[base_id] *= 100
env.sim.forward()


# RoboCasa door-state compatibility hack for the RB-Y1 free root joint.
joint_name = f"{door.name}_doorhinge"
sign = -1 if door.orientation == "left" else 1

def get_door_state(self, env):
    q = float(env.sim.data.get_joint_qpos(joint_name))
    return {"door": q * sign / (np.pi / 2)}

door.get_door_state = MethodType(get_door_state, door)


# ---------------- Waypoints ----------------

# left arm: 7 absolute joint positions
waypoints = [
    np.array([1.0, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0]),

    np.array([2.0, 0.8, 0.0, -1.2, 0.0, 0.1, 0.0]),

    np.array([
        2.25004234, 0.56720719, 1.54993574,
        -1.75093856, -2.40772740, 1.23985670, 0.10873634
    ]),

    np.array([
        2.35618000, 0.77341182, 1.82397071,
        -1.90104706, -2.19233291, 0.83652691, -0.04235979
    ]),
]

close_waypoints = [
    np.array([2.35618000, 0.83429631, 1.80340652, -1.81935931, -2.22011645, 0.86242375, -0.03042316]),
    np.array([2.25158774, 0.97215745, 1.96176251, -2.04996178, -2.15793500, 1.06127728, -0.01622436]),
    np.array([1.83926490, 1.77274467, 2.07017248, -2.18720323, -1.96192214, 0.84641332, 0.25548672]),
    np.array([1.70089335, 2.22270706, 2.09073309, -2.08053946, -1.80522124, 0.51642346, 0.19891226]),
    np.array([1.61069587, 2.62037820, 2.08767672, -1.81086591, -1.49775985, 0.30307993, -0.06746956]),
    np.array([1.35697493, 3.04169403, 1.82812581, -1.26416553, -1.01376510, 0.29805136, -0.32173078]),
]


# ---------------- Dumb controller ----------------

left_idx = np.asarray(robot.part_controllers["left"].qpos_index)
current_left = env.sim.data.qpos[left_idx].copy()

held = {
    "right": env.sim.data.qpos[robot.part_controllers["right"].qpos_index].copy(),
    "torso": env.sim.data.qpos[robot.part_controllers["torso"].qpos_index].copy(),
    "head": env.sim.data.qpos[robot.part_controllers["head"].qpos_index].copy(),
    "base": np.zeros(2),
    "right_gripper": np.array([1.0]),
}


def move_arm(target, gripper=-1.0, steps=80):
    global current_left

    start = current_left.copy()

    for i in range(steps):
        u = (i + 1) / steps
        q = (1 - u) * start + u * target

        action = robot.create_action_vector({
            **held,
            "left": q,
            "left_gripper": np.array([gripper]),
        })

        env.step(action)
        env.render()

    current_left = target.copy()


# ---------------- Execute task ----------------

# Approach handle with gripper open
for q in waypoints:
    move_arm(q, gripper=-1.0)

# Close gripper around handle
for _ in range(30):
    action = robot.create_action_vector({
        **held,
        "left": current_left,
        "left_gripper": np.array([1.0]),
    })
    env.step(action)
    env.render()

# Follow approximate door-closing arc
for q in close_waypoints:
    move_arm(q, gripper=1.0)

    if env._check_success():
        print("SUCCESS")
        break

print("Door state:", door.get_door_state(env))
print("RoboCasa success:", env._check_success())

env.close()