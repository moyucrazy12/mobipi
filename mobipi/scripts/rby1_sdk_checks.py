r"""Check the RB-Y1 SDK connection and frame conventions before running the pipeline with --backend sdk.

Steps 0 and 1 of the test sequence in docs/rby1_sdk_integration.md:

1. connect: power, servo and control manager, 24 joint state, odometry.
2. tcp: SDK FK (base -> ee_left) with the fixed offsets in rby1_sdk_utils
   against the RoboCasa model's base -> left grip_site, at random joint vectors
   (no motion). Pass: < 1 mm and < 0.5 deg.
3. odometry (moves the base; skipped with --no-motion): turns in place and
   checks that odometry tracks the base link, which circles the axle at
   0.228 m, like get_base_vec in RoboCasa.

    python -m mobipi.scripts.rby1_sdk_checks --checkpoint ~/Downloads/model_115_cartesian.pth
"""

import argparse
import copy
from pathlib import Path
import time


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--checkpoint", type=Path, required=True,
                        help="RB-Y1 policy checkpoint; its env metadata builds the RoboCasa model")
    result.add_argument("--robot-address", default="localhost:50051")
    result.add_argument("--num-samples", type=int, default=50, help="Random joint vectors for the TCP check")
    result.add_argument("--no-motion", action="store_true", help="Skip the check that moves the base")
    result.add_argument("--seed", type=int, default=0)
    return result


def make_model_env(checkpoint):
    import torch
    import robosuite
    import robocasa  # noqa: F401  (registers the kitchen environments)
    import mobipi.envs  # noqa: F401

    metadata = torch.load(str(checkpoint), map_location="cpu", weights_only=False)["env_metadata"]
    kwargs = copy.deepcopy(metadata["env_kwargs"])
    layouts, styles = kwargs.pop("layout_ids", None) or [1], kwargs.pop("style_ids", None) or [1]
    kwargs.update(base_types="RBY1MobileBase", layout_and_style_ids=[[layouts[0], styles[0]]], has_renderer=False,
                  has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True, render_camera=None)
    env = robosuite.make(metadata["env_name"], **kwargs)
    env.reset()
    return env


def check_tcp(env, robot, num_samples, rng):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from robomimic.utils.rby1_cartesian import make_transform
    from mobipi.utils import rby1_utils
    from mobipi.utils.rby1_sdk_utils import RS_PREFIX

    sim = env.sim
    tcp_site = env.robots[0].gripper["left"].important_sites["grip_site"]

    def site(name):
        return make_transform(sim.data.get_site_xpos(name), sim.data.get_site_xmat(name))

    position_errors, rotation_errors = [], []
    for _ in range(num_samples):
        q = np.zeros(len(robot.joint_names))
        for i, name in enumerate(robot.joint_names):
            if "wheel" in name:
                continue
            low, high = sim.model.jnt_range[sim.model.joint_name2id(RS_PREFIX + name)]
            q[i] = rng.uniform(low, high) * 0.8
            sim.data.set_joint_qpos(RS_PREFIX + name, q[i])
        sim.forward()
        expected = np.linalg.inv(site(rby1_utils.BASE_SITE)) @ site(tcp_site)
        actual = robot.tcp_in_base(q)
        position_errors.append(np.linalg.norm(actual[:3, 3] - expected[:3, 3]) * 1000)
        rotation_errors.append(np.rad2deg(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude()))
    passed = max(position_errors) < 1.0 and max(rotation_errors) < 0.5
    print(f"[tcp] {num_samples} joint vectors: max error {max(position_errors):.4f} mm, "
          f"{max(rotation_errors):.5f} deg -> {'PASS' if passed else 'FAIL'}", flush=True)
    return passed


def check_odometry(robot, axle_offset, speed=0.5, duration=3.0):
    import numpy as np
    from mobipi.utils.nav_utils import angle_wrap

    robot.reset_base_pose(np.zeros(3))
    end = time.perf_counter() + duration
    try:
        while time.perf_counter() < end:
            robot.command_base(0.0, speed)
            time.sleep(0.05)
        robot.command_base(0.0, 0.0)
        time.sleep(1.0)
    finally:
        robot.stop()
    robot.update()
    x, y, yaw = robot.base_pose()
    # the base link starts axle_offset behind the turning axle and circles it
    expected = np.array([axle_offset * (1 - np.cos(yaw)), -axle_offset * np.sin(yaw)])
    error = float(np.linalg.norm([x, y] - expected))
    passed = abs(angle_wrap(yaw)) > 0.2 and error < 0.02
    print(f"[odometry] turned {yaw:.3f} rad in place: base link at ({x:.3f}, {y:.3f}), expected "
          f"({expected[0]:.3f}, {expected[1]:.3f}) for the base link {axle_offset} m behind the axle "
          f"-> {'PASS' if passed else 'FAIL'}", flush=True)
    if abs(angle_wrap(yaw)) < 0.6 * speed * duration:
        print(f"[odometry] note: turned {yaw:.2f} rad for {speed * duration:.2f} rad commanded; the closed-loop "
              "base driver compensates, but expect slow base motion", flush=True)
    return passed


def main(argv=None):
    args = parser().parse_args(argv)
    import numpy as np
    from mobipi.utils import rby1_utils
    from mobipi.utils.rby1_sdk_utils import SDKRobot

    robot = SDKRobot(args.robot_address)
    results = {}
    try:
        state = robot.update()
        results["connect"] = len(state.position) == len(robot.joint_names) == 24
        print(f"[connect] {robot.address}: {len(state.position)} joints, base pose {robot.base_pose().round(3)}, "
              f"TCP in base {robot.tcp_in_base()[:3, 3].round(3)} -> {'PASS' if results['connect'] else 'FAIL'}",
              flush=True)
        env = make_model_env(args.checkpoint.expanduser())
        try:
            results["tcp"] = check_tcp(env, robot, args.num_samples, np.random.default_rng(args.seed))
            model = env.sim.model
            axle_offset = float(model.jnt_pos[model.joint_name2id(rby1_utils.YAW_JOINT)][0])
        finally:
            env.close()
        if not args.no_motion:
            results["odometry"] = check_odometry(robot, axle_offset)
    finally:
        robot.close()
    print("All checks passed" if all(results.values()) else f"Failed: {[k for k, v in results.items() if not v]}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
