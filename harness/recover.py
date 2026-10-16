"""Recovery pose for LIBERO: drive the hand back to where it started.

LIBERO-90's single-step demos start with the arm at its home pose. A step
started from wherever the previous step left the hand is out of the
distribution the VLA learned that step from (harness M2: after "turn on the
stove" the arm stayed by the knob and "put the moka pot on the stove" never
started, though alone it succeeds 100%). Same lesson as the cube harness:
skills have start states; the harness brings the robot back to them.

Scripted, through LIBERO's own operational-space controller (relative
end-effector actions, 1.0 = 5 cm / 0.5 rad per step), never by teleporting:
open the gripper and rise to the start height, then move to the start
position and orientation. Steps count against the episode budget.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from harness.start_state import OPEN_WIDTH, gripper_width

POS_TOL = 0.01   # m
ROT_TOL = 0.05   # rad
MAX_STEPS = 80
OPEN = -1.0      # LIBERO gripper action: -1 open, +1 close


def controller(env, i):
    return env.envs[i]._env.env.robots[0].controller


def start_pose(env, i):
    c = controller(env, i)
    return c.ee_pos.copy(), c.ee_ori_mat.copy()


class Recovery:
    """Per-episode recovery move; act() gives one 7-d action, done says when to stop."""

    def __init__(self, target_pos, target_rot):
        self.target_pos, self.target_rot = target_pos, target_rot
        self.steps = 0
        self.phase = "rise"

    def act(self, env, i):
        c = controller(env, i)
        pos, rot = c.ee_pos, c.ee_ori_mat
        if self.phase == "rise":
            goal = np.array([pos[0], pos[1], max(pos[2], self.target_pos[2])])
            if goal[2] - pos[2] < POS_TOL:
                self.phase = "return"
        if self.phase == "return":
            goal = self.target_pos
        dp = np.clip((goal - pos) / 0.05, -1, 1)
        drot = np.clip(Rotation.from_matrix(self.target_rot @ rot.T).as_rotvec() / 0.5, -1, 1)
        self.steps += 1
        return np.concatenate([dp, drot, [OPEN]]).astype(np.float32)

    def done(self, env, i):
        c = controller(env, i)
        pos_err = np.linalg.norm(self.target_pos - c.ee_pos)
        rot_err = np.linalg.norm(Rotation.from_matrix(self.target_rot @ c.ee_ori_mat.T).as_rotvec())
        at_pose = pos_err < POS_TOL and rot_err < ROT_TOL and gripper_width(env, i) >= OPEN_WIDTH
        return (self.phase == "return" and at_pose) or self.steps >= MAX_STEPS
