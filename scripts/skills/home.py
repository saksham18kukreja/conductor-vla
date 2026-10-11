"""home(): move the arm back to its start posture.

Behaviour: scripted (experts.ScriptedHome): rise clear of the table, then a
joint-space ramp home. A learned version would add nothing here: the motion
needs no perception and the scripted one already succeeds. The gripper keeps
its current command, so homing while holding keeps holding.
"""

import numpy as np

from env import GRIPPER_CLOSE, GRIPPER_OPEN
from skills.base import Check, Reason, Skill, Status

HOME_TOL = 0.02  # rad, every joint


class HomeSkill(Skill):
    name = "home"
    description = "Move the arm back to its start posture (gripper unchanged)."
    version = "v1.0 (scripted)"
    params = {}
    effects = "arm at the start posture"

    def __init__(self, checkpoint=None, device=None):
        self.agent = None  # scripted

    def reset(self, world):
        from experts import ScriptedHome
        grip = GRIPPER_CLOSE if world.gripper_cmd_closed else GRIPPER_OPEN
        self.expert = ScriptedHome(world.env, gripper_ctrl=grip)

    def step(self, world):
        return self.expert.act()

    def check(self, world):
        if np.abs(self.expert.q_home - world.env.arm_qpos).max() < HOME_TOL:
            return Check(Status.SUCCESS)
        if self.expert.timed_out:
            return Check(Status.FAILURE, Reason.TIMEOUT, "arm did not reach home")
        return Check()


class RecoverSkill(HomeSkill):
    """The recovery pose: home with the fingers open (unless holding).

    Home is every skill's standard start and the camera's clearest view of the
    table, so an orchestrator can return here when a skill refuses to start
    (target not visible, fingers closed) and try again. See skills/recovery.py.
    """

    name = "recover"
    description = "Open the gripper (unless holding) and move the arm to the start posture."
    version = "v1.0 (scripted)"
    effects = "arm at the start posture, fingers open unless holding"

    def reset(self, world):
        from experts import ScriptedHome
        grip = GRIPPER_CLOSE if world.holding is not None else GRIPPER_OPEN
        self.expert = ScriptedHome(world.env, gripper_ctrl=grip)

    def check(self, world):
        c = super().check(world)
        if c.status == Status.SUCCESS and world.holding is None and world.gripper_width < 0.075:
            return Check()  # home, but the fingers are still opening
        return c
