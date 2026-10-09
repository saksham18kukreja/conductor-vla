"""Benchmark tasks for skill mode: start states and ground-truth scoring.

Each task says how to set up a held-out episode for one skill (`starts`) and
how to score it from simulator ground truth (`Tracker`). Skills never see the
ground truth; only the benchmark does.

Starts (`benchmark.py --skill NAME --start START`). The first one listed is
the skill's standard start; the others are handoffs from other skills' end
states (PLAN.md Phase 2: skill success vs handoff success).

  pick   home         arm home, cube on the table
         after_place  arm and cube where a scripted pick + place left them
         after_push   arm and cube where a scripted push left them
  place  expert_pick  cube held as the pick expert leaves it (pick's success
                      condition + 0-40 steps)
         pick         cube held by the learned pick skill (run through the
                      Runner; episodes where pick fails are not scored)
  push   home         arm home, cube on the table
         after_place  arm and cube where a scripted pick + place left them
  home   after_place  arm above a cube a scripted place just set down
         after_push   arm above a cube a scripted push just moved
         holding      cube held after a scripted pick (home keeps holding)

Stages are "furthest reached" during the episode, except the last one,
`success`, which is judged on the final state (an object knocked away after
reaching the target doesn't count).

Targets come from rng (HELDOUT_SEED, position index, skill), so every
checkpoint and every start mode sees the same targets.
"""

import zlib

import mujoco
import numpy as np

import scenarios as sc
from env import TABLE_HEIGHT

HELDOUT_SEED = 2026
TOLERANCE = 0.015
REST_Z = sc.REST_Z


def finger_contacts(env):
    """Names of finger bodies touching the cube."""
    m, d = env.model, env.data
    out = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if env.cube_id in (b1, b2):
            other = b2 if b1 == env.cube_id else b1
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other)
            if name in ("left_finger", "right_finger"):
                out.add(name)
    return out


def resting(env):
    return abs(env.cube_pos[2] - REST_Z) < 0.003


def episode_rng(idx, skill):
    return np.random.default_rng([HELDOUT_SEED, idx, zlib.crc32(skill.encode())])


class Tracker:
    stages = []

    def __init__(self, env, goal=None):
        self.env, self.goal = env, goal
        self.start_xy = env.cube_pos[:2].copy()
        self.best = -1

    def hit(self, i):
        self.best = max(self.best, i)

    def goal_err(self):
        if self.goal is None:
            return float("nan")
        return float(np.linalg.norm(self.env.cube_pos[:2] - self.goal))

    def finish(self):
        """Judge the final state; -> success."""
        ok = self.final_ok()
        if ok:
            self.best = len(self.stages) - 1
        else:
            self.best = min(self.best, len(self.stages) - 2)
        return ok


class PlaceTracker(Tracker):
    stages = ["over", "lowered", "released", "placed", "success"]

    def update(self, world):
        env = self.env
        err = self.goal_err()
        touching = bool(finger_contacts(env))
        if err < TOLERANCE and env.cube_pos[2] > REST_Z + 0.01:
            self.hit(0)
        if err < TOLERANCE and env.cube_pos[2] < REST_Z + 0.01 and touching:
            self.hit(1)
        if not touching and resting(env) and self.best >= 0:
            self.hit(2)
        if not touching and resting(env) and err < TOLERANCE:
            self.hit(3)

    def final_ok(self):
        env = self.env
        return (self.goal_err() < TOLERANCE and resting(env) and not finger_contacts(env)
                and env.tcp_pos[2] - env.cube_pos[2] > 0.05)


class PushTracker(Tracker):
    stages = ["contact", "moving", "near", "at_target", "success"]

    def update(self, world):
        env = self.env
        if finger_contacts(env):
            self.hit(0)
        if np.linalg.norm(env.cube_pos[:2] - self.start_xy) > 0.01:
            self.hit(1)
        err = self.goal_err()
        if err < 2 * TOLERANCE:
            self.hit(2)
        if err < TOLERANCE:
            self.hit(3)

    def final_ok(self):
        env = self.env
        return (self.goal_err() < TOLERANCE and resting(env) and not finger_contacts(env)
                and env.tcp_pos[2] > REST_Z + 0.05)


class HomeTracker(Tracker):
    stages = ["clear", "success"]

    def __init__(self, env, goal=None):
        super().__init__(env, goal)
        from experts import ScriptedHome
        self.q_home = ScriptedHome(env).q_home

    def update(self, world):
        if self.env.tcp_pos[2] > REST_Z + 0.08:
            self.hit(0)

    def final_ok(self):
        return np.abs(self.q_home - self.env.arm_qpos).max() < 0.02


# ------------------------------------------------------------------ starts

def _after_place(env, xy, rng):
    if not sc.scripted_pick(env):
        return False
    return sc.scripted_place(env, sc.place_target(rng))


def _after_push(env, xy, rng):
    return sc.scripted_push(env, sc.push_target(rng, xy))


def setup(skill, start, env, world, xy, idx, pick_runner=None):
    """Prepare one episode. -> (skill args, goal xy or None), or None if the
    start state could not be produced (not scored)."""
    rng = episode_rng(idx, skill)
    env.reset(cube_xy=xy)
    world.reset()

    if skill == "pick":
        if start == "after_place" and not _after_place(env, xy, rng):
            return None
        if start == "after_push" and not _after_push(env, xy, rng):
            return None
        return {"object": "cube_red"}, None

    if skill == "place":
        if start == "expert_pick":
            if not sc.scripted_pick(env, extra_steps=int(rng.integers(0, 41))):
                return None
            world.holding, world.gripper_cmd_closed = "cube_red", True
        elif start == "pick":
            res = pick_runner.execute(pick_runner.pick_skill, object="cube_red")
            if not res.ok:
                return None
        goal = sc.place_target(rng)
        return {"object": "cube_red", "target": f"{goal[0]:.4f},{goal[1]:.4f}"}, goal

    if skill == "push":
        if start == "after_place" and not _after_place(env, xy, rng):
            return None
        goal = sc.push_target(rng, env.cube_pos[:2].copy())
        return {"object": "cube_red", "target": f"{goal[0]:.4f},{goal[1]:.4f}"}, goal

    if skill == "home":
        if start == "after_place" and not _after_place(env, xy, rng):
            return None
        if start == "after_push" and not _after_push(env, xy, rng):
            return None
        if start == "holding":
            if not sc.scripted_pick(env):
                return None
            world.holding, world.gripper_cmd_closed = "cube_red", True
        return {}, None
    raise ValueError(skill)


STARTS = {
    "pick": ["home", "after_place", "after_push"],
    "place": ["expert_pick", "pick"],
    "push": ["home", "after_place"],
    "home": ["after_place", "after_push", "holding"],
}
TRACKERS = {"place": PlaceTracker, "push": PushTracker, "home": HomeTracker}
# Scoring protocol ids (compare only rows with the same eval_set and protocol).
# p1: stages above, success judged on the final state, 15 mm target tolerance,
# one call per position, 700 steps.
PROTOCOL = {"place": "place-p1", "push": "push-p1", "home": "home-p1"}
