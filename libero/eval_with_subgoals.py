"""lerobot-eval with every LIBERO goal predicate logged per step.

A LIBERO task succeeds when all predicates of its BDDL goal hold at once,
e.g. (In alphabet_soup_1 basket_1_contain_region) and (In tomato_sauce_1 ...).
lerobot-eval only reports the conjunction. This wraps LeRobot's LiberoEnv so
each episode also records, per predicate: whether it ever held, the first
step it did, and whether it held at the end. That separates "did one of two
objects" from "did nothing" and "undid a step".

Episodes the vector env starts on its own (autoreset after an episode ends,
cut off when the batch finishes) are marked "autoreset": true; only the
others are the evaluated episodes.

Same arguments as lerobot-eval; the log goes to $SUBGOAL_LOG (JSON lines,
one per episode):

  SUBGOAL_LOG=out.jsonl python eval_with_subgoals.py --policy.path=... --env.type=libero ...
"""

import json
import os

from gymnasium.vector import SyncVectorEnv
from lerobot.envs import libero as lib

_in_vector_step = False
_orig_vstep = SyncVectorEnv.step


def _vstep(self, actions):
    """A reset inside SyncVectorEnv.step is an autoreset, not an evaluated episode."""
    global _in_vector_step
    _in_vector_step = True
    try:
        return _orig_vstep(self, actions)
    finally:
        _in_vector_step = False


SyncVectorEnv.step = _vstep

LOG = os.environ.get("SUBGOAL_LOG", "subgoals.jsonl")
_orig_reset, _orig_step, _orig_close = lib.LiberoEnv.reset, lib.LiberoEnv.step, lib.LiberoEnv.close


def _goal(env):
    return env._env.env.parsed_problem["goal_state"]


def _flush(env):
    ep = getattr(env, "_sg", None)
    if ep is None or ep["steps"] == 0:
        return
    with open(LOG, "a") as f:
        f.write(json.dumps(ep) + "\n")
    env._sg = None


def reset(self, seed=None, **kwargs):
    _flush(self)
    init_id = self.init_state_id  # the init state this reset will use
    obs, info = _orig_reset(self, seed=seed, **kwargs)
    goal = _goal(self)
    self._sg = {"task_id": self.task_id, "init_state": int(init_id), "seed": seed,
                "env": id(self), "autoreset": _in_vector_step,
                "predicates": [" ".join(g) for g in goal], "steps": 0, "success": False,
                "success_step": None, "first_true": [None] * len(goal),
                "final": [False] * len(goal), "ever_lost": [False] * len(goal)}
    return obs, info


def step(self, action):
    out = _orig_step(self, action)
    ep = getattr(self, "_sg", None)
    if ep is not None:
        ep["steps"] += 1
        inner = self._env.env
        for i, g in enumerate(_goal(self)):
            ok = bool(inner._eval_predicate(g))
            if ok and ep["first_true"][i] is None:
                ep["first_true"][i] = ep["steps"]
            if not ok and ep["final"][i]:
                ep["ever_lost"][i] = True
            ep["final"][i] = ok
        if out[4].get("is_success") and not ep["success"]:
            ep["success"], ep["success_step"] = True, ep["steps"]
    return out


def close(self):
    _flush(self)
    return _orig_close(self)


lib.LiberoEnv.reset, lib.LiberoEnv.step, lib.LiberoEnv.close = reset, step, close

if __name__ == "__main__":
    import atexit
    import gc

    def _flush_all():
        for o in gc.get_objects():
            if isinstance(o, lib.LiberoEnv):
                _flush(o)

    atexit.register(_flush_all)
    from lerobot.scripts.lerobot_eval import main
    main()
