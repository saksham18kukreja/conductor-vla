"""place(object, target): set a held object down at a target on the table.

Behaviour: BC policy (v3 recipe: `both`-frame features, BCE gripper, MLP)
trained on ScriptedPlace demos that start where pick leaves the cube. Inputs:
proprioception, the world model's estimate of the object, and the target
(skills/features.goal_obs). Termination uses observable signals only.
"""

import os

import numpy as np

from skills.base import Check, Reason, Skill, Status
from skills.features import REST_Z, in_target_range, parse_target

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "..", "..", "models", "bc_place3000_s0.pt")

TOLERANCE = 0.015       # success: object within this of the target (xy)
ON_TABLE = 0.010        # object centre within this of its resting height
CLEAR = 0.07            # TCP this far above the object after release
EMPTY_WIDTH, EMPTY_STEPS = 0.025, 10   # fingers closed on nothing
SETTLE_STEPS = 10       # success/miss must hold this long (estimate at 10 Hz)


class PlaceSkill(Skill):
    name = "place"
    description = "Set a held object down at a target on the table and move the hand clear."
    version = "v1.0"
    params = {"object": "id of the held object, e.g. 'cube_red'",
              "target": "zone name (center, left, right, near, far) or 'x,y' in metres"}
    effects = "object resting on the table at the target, gripper open and empty"

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
        if world.holding != object:
            return False, Reason.PRECONDITION_FAILED, f"not holding {object} (holding {world.holding})"
        if not world.objects[object].detected:
            return False, Reason.TARGET_NOT_DETECTED, f"{object} not detected"
        xy, _ = parse_target(target)
        if not in_target_range(xy):
            return False, Reason.UNREACHABLE, f"target {xy.round(3).tolist()} outside the placing area"
        return True, Reason.NONE, ""

    def reset(self, world, object, target):
        self.object = object
        self.goal, _ = parse_target(target)
        self.agent.reset()
        self.released = False
        self.empty_steps = self.done_steps = self.miss_steps = 0

    def step(self, world):
        return self.agent.act(world.env, world.objects[self.object].position, self.goal)

    def check(self, world):
        est = world.objects[self.object].position
        closed = world.gripper_cmd_closed
        if not closed:
            self.released = True
        self.empty_steps = self.empty_steps + 1 if closed and world.gripper_width < EMPTY_WIDTH else 0
        if self.empty_steps >= EMPTY_STEPS:
            return Check(Status.FAILURE, Reason.OBJECT_DROPPED,
                         f"{self.object} slipped out before release")
        if not self.released:
            return Check()
        err = float(np.linalg.norm(est[:2] - self.goal))
        resting = abs(est[2] - REST_Z) < ON_TABLE
        clear = world.tcp_pos[2] - est[2] > CLEAR
        settled = resting and clear
        self.done_steps = self.done_steps + 1 if settled and err < TOLERANCE else 0
        self.miss_steps = self.miss_steps + 1 if settled and err >= TOLERANCE else 0
        if self.done_steps >= SETTLE_STEPS:
            return Check(Status.SUCCESS)
        if self.miss_steps >= SETTLE_STEPS:
            return Check(Status.FAILURE, Reason.TARGET_MISSED,
                         f"{self.object} set down {1000 * err:.0f} mm from the target")
        return Check()

    def apply_effects(self, world, object, target):
        world.holding = None
