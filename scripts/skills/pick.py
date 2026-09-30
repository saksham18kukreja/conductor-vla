"""pick(object): grasp an object on the table and lift it 10 cm.

Behaviour: the v4.0 BC policy (v3.0 checkpoint driven by the world model's
cube estimate). Termination uses observable signals only: gripper command and
width, and the camera's estimate of the object's height.
"""

import os

import numpy as np
import torch

from env import TABLE_HEIGHT, CUBE_HALF
from skills.base import Check, Reason, Skill, Status

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "..", "..", "models", "bc_clean3000_both_bce_time_s0.pt")

REST_Z = TABLE_HEIGHT + CUBE_HALF
LIFT_HEIGHT = 0.10       # success: estimated centre this far above rest
HOLD_STEPS = 50          # ... for 1 s
HOLD_WIDTH = (0.030, 0.050)  # fingers stopped on a 40 mm cube
EMPTY_WIDTH = 0.025      # fingers closed past this: nothing between them
EMPTY_STEPS = 10         # ... for 0.2 s before declaring it


class PickSkill(Skill):
    name = "pick"
    description = "Grasp an object on the table and lift it 10 cm."
    version = "v4.0"
    params = {"object": "id of the object to pick, e.g. 'cube_red'"}
    effects = "gripper holding the object, object ~10 cm above the table"

    def __init__(self, checkpoint=DEFAULT_CKPT, device=None):
        from benchmark import Agent  # checkpoint loading + observation/action mapping
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = Agent(checkpoint, device)

    def validate(self, world, **args):
        ok, msg = super().validate(world, **args)
        if ok and args["object"] not in world.objects:
            return False, f"unknown object {args['object']!r}; known: {sorted(world.objects)}"
        return ok, msg

    def preconditions(self, world, object):
        est = world.objects[object]
        if not est.detected:
            return (False, Reason.TARGET_NOT_DETECTED,
                    f"{object} not detected (visible {100 * est.visible:.0f}%)")
        if world.holding is not None:
            return False, Reason.PRECONDITION_FAILED, f"already holding {world.holding}"
        return True, Reason.NONE, ""

    def reset(self, world, object):
        self.object = object
        self.agent.reset()
        self.held_steps = 0
        self.empty_steps = 0
        self.was_lifted = False

    def step(self, world):
        arm, grip, _ = self.agent.act(world.env, world.objects[self.object].position)
        return arm, grip

    def check(self, world):
        est = world.objects[self.object].position
        closed = world.gripper_cmd_closed
        width = world.gripper_width
        holding = closed and HOLD_WIDTH[0] < width < HOLD_WIDTH[1]
        height = est[2] - REST_Z

        if holding and height > 0.03:
            self.was_lifted = True
        self.held_steps = self.held_steps + 1 if holding and height > LIFT_HEIGHT else 0
        self.empty_steps = self.empty_steps + 1 if closed and width < EMPTY_WIDTH else 0

        if self.held_steps >= HOLD_STEPS:
            return Check(Status.SUCCESS)
        if self.empty_steps >= EMPTY_STEPS:
            if self.was_lifted:
                return Check(Status.FAILURE, Reason.OBJECT_DROPPED,
                             f"{self.object} slipped out after being lifted")
            return Check(Status.FAILURE, Reason.GRASP_MISSED,
                         f"fingers closed on nothing (width {1000 * width:.0f} mm)")
        return Check()

    def apply_effects(self, world, object):
        world.holding = object
