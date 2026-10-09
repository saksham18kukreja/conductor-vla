"""Start states and targets for each skill, shared by demo collection and the
benchmark so both draw from the same distributions.

  pick   arm home, cube on the table in the pick range
  place  cube held ~10 cm up, as pick leaves it; target on the table
  push   arm home, cube on the table; target 5-15 cm away in any direction
  home   arm wherever another skill left it (here: after a scripted place)

Ranges (world frame, metres):

  PICK_RANGE   cube centres the pick policy was trained on
  TARGET_LO/HI where place/push targets may be (skills/features.py; inside
               the pick range, so the next pick can reach the result)
"""

import numpy as np

from controller import ScriptedGrasp
from env import CUBE_HALF, TABLE_HEIGHT
from experts import ScriptedPlace, ScriptedPush
from skills.features import TARGET_HI, TARGET_LO

DEC = 10
REST_Z = TABLE_HEIGHT + CUBE_HALF
PICK_CENTRE = np.array([0.55, 0.0])
PICK_HALF = np.array([0.08, 0.15])
PUSH_DIST = (0.05, 0.15)

# pick's success condition (skills/pick.py), checked on ground truth here
PICK_LIFT, PICK_HOLD, PICK_WIDTH = 0.10, 50, (0.030, 0.050)


def cube_xy(rng):
    return PICK_CENTRE + rng.uniform(-PICK_HALF, PICK_HALF)


def place_target(rng, start_xy=None):
    return rng.uniform(TARGET_LO, TARGET_HI)


def push_target(rng, start_xy):
    """A target 5-15 cm from the cube, inside TARGET_RANGE."""
    for _ in range(1000):
        ang = rng.uniform(-np.pi, np.pi)
        t = start_xy + rng.uniform(*PUSH_DIST) * np.array([np.cos(ang), np.sin(ang)])
        if np.all(t >= TARGET_LO) and np.all(t <= TARGET_HI):
            return t
    raise RuntimeError(f"no push target fits from {start_xy}")


def scripted_pick(env, extra_steps=0, max_steps=1000):
    """Run the pick expert until pick's own success condition holds, plus
    `extra_steps`. Leaves the cube held ~10 cm up, as the pick skill would.
    Returns True if the condition was met."""
    pol = ScriptedGrasp(env)
    held, after = 0, None
    for _ in range(max_steps):
        if not pol.done:  # once the expert finishes its lift, hold that pose
            arm, grip = pol.act()
        env.step(arm, grip, n_substeps=DEC)
        ok = (grip < 128 and PICK_WIDTH[0] < env.gripper_width < PICK_WIDTH[1]
              and env.cube_pos[2] - REST_Z > PICK_LIFT)
        held = held + 1 if ok else 0
        if after is None and held >= PICK_HOLD:
            after = 0
        if after is not None:
            if after >= extra_steps:
                return True
            after += 1
    return False


def run_expert(env, pol, max_steps=1500):
    for _ in range(max_steps):
        arm, grip = pol.act()
        env.step(arm, grip, n_substeps=DEC)
        if pol.done:
            break
    return not pol.timed_out


def scripted_place(env, target_xy):
    return run_expert(env, ScriptedPlace(env, target_xy))


def scripted_push(env, target_xy):
    return run_expert(env, ScriptedPush(env, target_xy))
