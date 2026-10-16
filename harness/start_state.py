"""The VLA skill's start-state contract, checked by the harness before every step.

Same idea as the cube harness's StartState (scripts/skills/base.py): a skill
declares the robot states it was trained to start from, the harness checks
them before running it, and recovers (harness/recover.py) only when they fail.

LIBERO-90's single-step demos start with the hand at the scene's start pose
and the gripper open, so that is the state a step may start from: the hand
within POS_TOL / ROT_TOL of this episode's start pose (recorded after LIBERO's
settling steps) and the fingers open. The tolerances are looser than the
recovery move's (1 cm / 0.05 rad) so a finished recovery always passes.
"""

import numpy as np
from scipy.spatial.transform import Rotation

POS_TOL = 0.02    # m
ROT_TOL = 0.10    # rad
OPEN_WIDTH = 0.07  # m, finger opening (fully open 0.0775, closed 0.0023)


def gripper_width(env, i):
    inner = env.envs[i]._env.env
    idx = inner.robots[0]._ref_gripper_joint_pos_indexes
    return float(np.abs(inner.sim.data.qpos[idx]).sum())


def check(env, i, start):
    """-> (ok, message). start = (position, rotation matrix) of the hand at episode start."""
    c = env.envs[i]._env.env.robots[0].controller
    pos_err = float(np.linalg.norm(start[0] - c.ee_pos))
    rot_err = float(np.linalg.norm(Rotation.from_matrix(start[1] @ c.ee_ori_mat.T).as_rotvec()))
    width = gripper_width(env, i)
    problems = []
    if pos_err > POS_TOL:
        problems.append(f"hand {100 * pos_err:.1f} cm from start pose")
    if rot_err > ROT_TOL:
        problems.append(f"hand turned {rot_err:.2f} rad from start pose")
    if width < OPEN_WIDTH:
        problems.append(f"gripper not open ({1000 * width:.0f} mm)")
    return not problems, "; ".join(problems)
