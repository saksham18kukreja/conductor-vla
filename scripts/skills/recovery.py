"""Recovery pose: when a skill refuses to start, go home and try again.

The first piece of orchestrator logic (PLAN.md Phase 2, fix (c) for pick after
push). Skills only report why they can't start (SKILLS.md decision 6); this
picks the fix. Home with open fingers is every skill's standard start and the
camera's clearest view of the table, so for the refusals below one `recover`
call followed by a second try is the whole policy:

  TARGET_NOT_DETECTED  the target isn't visible: the arm may be hiding it
  START_STATE          the robot isn't in a start state the skill handles
                       (e.g. pick with closed fingers, as push leaves them)

Failures during a run (grasp missed, dropped) are not recovered here: a retry
after those is a separate experiment (Phase 3), and the object may have moved
out of the skill's range. If the arm is already at the recovery pose and the
target still isn't visible, going there again changes nothing, so it stops.
"""

import numpy as np

from skills.base import Reason
from skills.home import HOME_TOL

RECOVERABLE = (Reason.TARGET_NOT_DETECTED, Reason.START_STATE)


def at_recovery_pose(world):
    from experts import ScriptedHome
    q_home = ScriptedHome(world.env).q_home
    return (np.abs(q_home - world.env.arm_qpos).max() < HOME_TOL
            and (world.holding is not None or world.gripper_width >= 0.075))


def execute_with_recovery(runner, skill, recover_skill, max_recoveries=1, **args):
    """Run `skill`; on a recoverable refusal run `recover_skill` and try again.
    -> (final skill Result, list of every Result in order)."""
    log = [runner.execute(skill, **args)]
    for _ in range(max_recoveries):
        res = log[-1]
        if res.ok or res.reason not in RECOVERABLE:
            break
        if res.reason == Reason.TARGET_NOT_DETECTED and at_recovery_pose(runner.world):
            break
        rec = runner.execute(recover_skill)
        log.append(rec)
        if not rec.ok:
            return res, log
        log.append(runner.execute(skill, **args))
    final = next(r for r in reversed(log) if r.skill == skill.name)
    return final, log
