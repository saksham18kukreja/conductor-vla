"""push(object, target): push an object along the table to a target.

Behaviour: BC policy (v3 recipe) trained on ScriptedPush demos: closed
fingers, an x leg then a y leg with line following, rise clear. Inputs:
proprioception, the world model's object estimate and the target
(skills/features.goal_obs). Termination uses observable signals only.
"""

import os

import numpy as np

from skills.base import Check, Reason, Skill, Status
from skills.features import REST_Z, in_target_range, parse_target

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "..", "..", "models", "bc_push3000_s0.pt")

TOLERANCE = 0.015       # success: object within this of the target (xy)
CLEAR_Z = REST_Z + 0.06 # ... with the TCP risen above this
OVERSHOOT = 0.03        # failure: object this much further from the target than at the start
ON_TABLE = 0.010        # failure: object centre off its resting height (tipped, fell)
SETTLE_STEPS = 10


class PushSkill(Skill):
    name = "push"
    description = "Push an object along the table to a target, without grasping it."
    version = "v1.0"
    params = {"object": "id of the object to push, e.g. 'cube_red'",
              "target": "zone name (center, left, right, near, far) or 'x,y' in metres"}
    effects = "object at the target, gripper clear above it"

    def __init__(self, checkpoint=DEFAULT_CKPT, device=None):
        from skills.learned import GoalPolicy
        self.agent = GoalPolicy(checkpoint, device)

    def validate(self, world, **args):
        ok, msg = super().validate(world, **args)
        if not ok:
            return ok, msg
        if args["object"] not in world.objects:
            return False, f"unknown object {args['object']!r}; known: {sorted(world.objects)}"
        xy, msg = parse_target(args["target"])
        return xy is not None, msg

    def preconditions(self, world, object, target):
        est = world.objects[object]
        if not est.detected:
            return (False, Reason.TARGET_NOT_DETECTED,
                    f"{object} not detected (visible {100 * est.visible:.0f}%)")
        if world.holding is not None:
            return False, Reason.PRECONDITION_FAILED, f"holding {world.holding}; place it first"
        if abs(est.position[2] - REST_Z) > ON_TABLE:
            return False, Reason.PRECONDITION_FAILED, f"{object} is not resting on the table"
        xy, _ = parse_target(target)
        if not in_target_range(xy):
            return False, Reason.UNREACHABLE, f"target {xy.round(3).tolist()} outside the pushing area"
        return True, Reason.NONE, ""

    def reset(self, world, object, target):
        self.object = object
        self.goal, _ = parse_target(target)
        self.agent.reset()
        self.start_err = float(np.linalg.norm(world.objects[object].position[:2] - self.goal))
        self.done_steps = self.off_steps = 0

    def step(self, world):
        return self.agent.act(world.env, world.objects[self.object].position, self.goal)

    def check(self, world):
        est = world.objects[self.object].position
        err = float(np.linalg.norm(est[:2] - self.goal))
        clear = world.tcp_pos[2] > CLEAR_Z
        self.done_steps = self.done_steps + 1 if err < TOLERANCE and clear else 0
        self.off_steps = self.off_steps + 1 if abs(est[2] - REST_Z) > ON_TABLE else 0
        if self.done_steps >= SETTLE_STEPS:
            return Check(Status.SUCCESS)
        if self.off_steps >= SETTLE_STEPS:
            return Check(Status.FAILURE, Reason.OBJECT_DROPPED,
                         f"{self.object} left the table surface (tipped or fell)")
        if err > self.start_err + OVERSHOOT:
            return Check(Status.FAILURE, Reason.TARGET_MISSED,
                         f"{self.object} pushed away from the target ({1000 * err:.0f} mm off)")
        return Check()
