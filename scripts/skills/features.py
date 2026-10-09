"""Observation for goal-conditioned skills (place, push), and target parsing.

One function builds the observation for both demo collection (true cube
position) and the skill at run time (world-model estimate), so the two can't
drift apart. It follows the v3 recipe's `both` frame: world-frame features
plus the cube and goal relative to the gripper, in the gripper's frame.
No timer input: the place/push experts have no hidden timers.
"""

import numpy as np

from env import CUBE_HALF, TABLE_HEIGHT

REST_Z = TABLE_HEIGHT + CUBE_HALF

GOAL_OBS_LABELS = (
    [f"arm_q{i+1}" for i in range(7)] + ["grip_width"]
    + ["tcp_x", "tcp_y", "tcp_z"]
    + ["obj_x", "obj_y", "obj_z"]
    + ["obj_rel_x", "obj_rel_y", "obj_rel_z"]          # object - tcp
    + ["goal_x", "goal_y", "goal_z"]                    # goal on the table
    + ["goal_obj_x", "goal_obj_y", "goal_obj_z"]        # goal - object
    + ["obj_g_x", "obj_g_y", "obj_g_z"]                 # object - tcp, gripper frame
    + ["goal_g_x", "goal_g_y", "goal_g_z"]              # goal - tcp, gripper frame
)


def goal_obs(env, obj_pos, goal_xy):
    """29-d observation from proprioception, an object position and a goal."""
    tcp, R = env.tcp_pos, env.tcp_mat
    obj = np.asarray(obj_pos, dtype=float)
    goal = np.array([goal_xy[0], goal_xy[1], REST_Z])
    return np.concatenate([
        env.arm_qpos, [env.gripper_width], tcp, obj, obj - tcp, goal, goal - obj,
        R.T @ (obj - tcp), R.T @ (goal - tcp),
    ]).astype(np.float32)


# Named places on the table (robot frame: +x away from the robot, +y to its
# left), so a planner can say place(cube_red, "left") as well as "0.55,0.10".
ZONES = {
    "center": (0.55, 0.00),
    "left": (0.55, 0.10),
    "right": (0.55, -0.10),
    "near": (0.50, 0.00),
    "far": (0.60, 0.00),
}
TARGET_LO = np.array([0.48, -0.13])
TARGET_HI = np.array([0.62, 0.13])


def parse_target(value):
    """Zone name, "x,y" string, or a 2-sequence -> (xy array, error message)."""
    if isinstance(value, str):
        if value in ZONES:
            return np.array(ZONES[value], dtype=float), ""
        try:
            xy = np.array([float(v) for v in value.split(",")])
        except ValueError:
            return None, f"target {value!r}: expected a zone {sorted(ZONES)} or 'x,y'"
    else:
        xy = np.asarray(value, dtype=float).reshape(-1)
    if xy.shape != (2,):
        return None, f"target {value!r}: expected 2 coordinates"
    return xy, ""


def in_target_range(xy):
    return bool(np.all(xy >= TARGET_LO) and np.all(xy <= TARGET_HI))
