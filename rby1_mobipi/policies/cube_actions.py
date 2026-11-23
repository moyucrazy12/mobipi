"""CubeInCup action contracts; no controller connections or motion execution."""
import numpy as np
from scipy.spatial.transform import Rotation
from robomimic.utils.rby1_cartesian import rotation6d_to_matrix
from robomimic.utils.rby1_relative_cartesian import (
    tcp_pose, transform_to_pose9, decode_relative_target,
)

EXPERIMENTS = ("baseline_7d", "binary_7d", "rot6d_10d", "relative_10d")
WIDTH_THRESHOLD = 0.04  # Metres; training-only distribution has modes at 0 and .08.


def binary_gripper(width):
    return np.where(np.asarray(width) >= WIDTH_THRESHOLD, -1., 1.).astype(np.float32)


class GripperHysteresis:
    """Threshold physical binary predictions; never send continuous DP outputs.

    Positive means closed. Retain previous state inside [-.2, .2].
    """
    def __init__(self, closed=False, open_width=0.08):
        self.closed = bool(closed)
        self.open_width = float(open_width)

    def decode(self, prediction):
        if not np.isfinite(prediction):
            raise ValueError("Non-finite gripper prediction")
        if prediction >= .2:
            self.closed = True
        elif prediction <= -.2:
            self.closed = False
        return 0. if self.closed else self.open_width


def measured_poses(obs):
    return tcp_pose(obs['right_ee_pos'], obs['right_ee_quat'], 'wxyz')


def rotation6d_from_euler(euler):
    r = Rotation.from_euler('xyz', np.asarray(euler).reshape(-1, 3)).as_matrix()
    return np.concatenate((r[..., :, 0], r[..., :, 1]), axis=-1)


def decode_chunk(experiment, actions, anchor):
    """Decode physical (unnormalized) actions into world EE poses.

    A/B/C accumulate spatial rotations on the LEFT, translations in world axes.
    D decodes every future measured-pose target from ONE observation anchor.
    Gripper conversion is separate; this function never commands a robot.
    """
    a = np.asarray(actions, dtype=np.float64)
    dim = 7 if experiment in EXPERIMENTS[:2] else 10
    if experiment not in EXPERIMENTS or a.ndim != 2 or a.shape[1] != dim or not np.isfinite(a).all():
        raise ValueError('Invalid experiment, action shape or finite values')
    if experiment == 'relative_10d':
        return decode_relative_target(anchor, a[:, :9])
    out = []
    current = np.array(anchor, dtype=np.float64, copy=True)
    rotations = (Rotation.from_euler('xyz', a[:, 3:6]).as_matrix()
                 if dim == 7 else rotation6d_to_matrix(a[:, 3:9]))
    for action, rotation in zip(a, rotations):
        current[:3, 3] += action[:3]
        current[:3, :3] = rotation @ current[:3, :3]
        out.append(current.copy())
    return np.asarray(out)
