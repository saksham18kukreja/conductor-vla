"""lerobot-eval with the harness's rollout loop (and per-predicate logging).

Everything else is lerobot-eval's: env creation, seeds, batching, scoring,
videos, eval_info.json. Same arguments; harness switches in $HARNESS_CONFIG
(see harness/config.py), predicate log in $SUBGOAL_LOG. Run from
~/experiments/lerobot with its env:

  MUJOCO_GL=egl LIBERO_CONFIG_PATH=$PWD/.libero SUBGOAL_LOG=out.jsonl \\
      .env/bin/python ../franka_cube_lift/harness/eval_harness.py <lerobot-eval args>
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # import harness.*
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "libero"))

import eval_with_subgoals  # noqa: E402,F401  (patches LiberoEnv to log predicates)
from lerobot.scripts import lerobot_eval  # noqa: E402

from harness.rollout import harness_rollout  # noqa: E402

lerobot_eval.rollout = harness_rollout

if __name__ == "__main__":
    import atexit
    import gc

    def _flush_all():
        for o in gc.get_objects():
            if isinstance(o, eval_with_subgoals.lib.LiberoEnv):
                eval_with_subgoals._flush(o)

    atexit.register(_flush_all)
    lerobot_eval.main()
