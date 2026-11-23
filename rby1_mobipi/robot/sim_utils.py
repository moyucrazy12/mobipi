"""Shared RB-Y1 simulator access and pose helpers, preserving existing conventions."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def raw_model(sim):
    return sim.model._model if hasattr(sim.model, '_model') else sim.model

def raw_data(sim):
    return sim.data._data if hasattr(sim.data, '_data') else sim.data

def forward(sim):
    try:
        sim.forward()
    except Exception:
        mujoco.mj_forward(raw_model(sim), raw_data(sim))

def normalize(v, eps=1e-09):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise RuntimeError(f'Cannot normalize vector {v}')
    return v / n

def smoothstep(t):
    return 3.0 * t ** 2 - 2.0 * t ** 3

def site_pose(sim, name):
    m, d = (raw_model(sim), raw_data(sim))
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f'Site not found: {name}')
    pos = np.asarray(d.site_xpos[sid], dtype=float).copy()
    mat = np.asarray(d.site_xmat[sid], dtype=float).reshape(3, 3).copy()
    return (pos, mat)

def make_T(pos, rot):
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T

def interpolate_rotation(R0, R1, t):
    relative = R1 @ R0.T
    rotvec = Rotation.from_matrix(relative).as_rotvec()
    return Rotation.from_rotvec(t * rotvec).as_matrix() @ R0

def arm_joint_names(arm):
    return [f"robot0_{arm}_arm_{i}" for i in range(7)]

def force_tangent_direction(tangent, push_x_direction):
    if push_x_direction != 0 and np.sign(tangent[0]) != push_x_direction:
        tangent = -tangent
    return tangent
