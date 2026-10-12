"""The skill abstraction: a behaviour with a contract, run by a shared Runner.

A skill declares what it needs (parameters, preconditions), produces one action
per control step, decides from observable signals when it has finished, and
promises effects on success. Everything outside a skill sees only this
contract, so a scripted controller and a learned policy look the same.

Lifecycle (Runner.execute):

  validate args -> preconditions -> start state -> reset -> [step -> check]* -> Result
       |                |               |                            |
   INVALID_ARGS   PRECONDITION_FAILED  START_STATE          success / failure / timeout

Start states are part of the contract, declared as data (StartState) and
checked by the Runner, not by the skill: they say which robot states the
skill was trained or designed to start from. A learned skill is only
trustworthy from those (pick after push 35%, push after place 75%), and
an orchestrator can reach them with the recovery pose (skills/recovery.py).

Skills read the world model (perception + memory) and the robot's own state.
They never read the simulator's object state; ground truth is only for scoring
(scripts/benchmark.py --skill).
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

DEC = 10  # physics substeps per control step (500 Hz -> 50 Hz)
HOME_START_TOL = 0.05  # rad, every joint: "arm at home" for a start state
OPEN_WIDTH = 0.070     # m: "fingers open" (fully open: 80 mm; a 40 mm cube fits)


class Status(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


class Reason(str, Enum):
    """Shared failure codes, so an orchestrator or planner can react to them."""
    NONE = "none"
    INVALID_ARGS = "invalid_args"
    PRECONDITION_FAILED = "precondition_failed"
    # the robot is not in a start state the skill handles (e.g. fingers closed);
    # an orchestrator can fix it with the recovery pose (skills/recovery.py)
    START_STATE = "start_state"
    TARGET_NOT_DETECTED = "target_not_detected"
    UNREACHABLE = "unreachable"
    GRASP_MISSED = "grasp_missed"
    OBJECT_DROPPED = "object_dropped"
    TARGET_MISSED = "target_missed"
    TIMEOUT = "timeout"


@dataclass
class Check:
    """Outcome of one termination check."""
    status: Status = Status.RUNNING
    reason: Reason = Reason.NONE
    message: str = ""


@dataclass(frozen=True)
class StartState:
    """The robot states a skill may start from, checked by the Runner.

    arm      "any", or "home": every joint within HOME_START_TOL of the
             start posture
    fingers  "any", "open" (width >= OPEN_WIDTH) or "closed" (narrower)
    """
    arm: str = "any"
    fingers: str = "any"

    def check(self, world):
        """-> (ok, message), from proprioception only."""
        env = world.env
        if self.arm == "home":
            q_home = env.model.key_qpos[env.home_key][: len(env.arm_qpos_adr)]
            err = float(np.abs(env.arm_qpos - q_home).max())
            if err > HOME_START_TOL:
                return False, f"arm not at home (joint error up to {err:.2f} rad)"
        width = world.gripper_width
        if self.fingers == "open" and width < OPEN_WIDTH:
            return False, f"fingers not open (width {1000 * width:.0f} mm)"
        if self.fingers == "closed" and width >= OPEN_WIDTH:
            return False, f"fingers not closed (width {1000 * width:.0f} mm)"
        return True, ""

    def __str__(self):
        return f"arm {self.arm}, fingers {self.fingers}"


@dataclass
class Result:
    skill: str
    args: dict
    status: Status
    reason: Reason = Reason.NONE
    message: str = ""
    steps: int = 0
    world: dict = field(default_factory=dict)  # world-model snapshot at the end

    @property
    def ok(self):
        return self.status == Status.SUCCESS

    def __str__(self):
        why = f" [{self.reason.value}] {self.message}" if not self.ok else ""
        return f"{self.skill}({self.args}) -> {self.status.value} in {self.steps} steps{why}"


class Skill:
    """Base class. Subclasses fill in the contract."""

    name = ""
    description = ""
    version = ""
    params = {}   # parameter name -> description (for validation and planners)
    effects = ""  # what the world looks like after success
    start_state = StartState()  # robot states it may start from (Runner checks)

    def validate(self, world, **args):
        """-> (ok, message). Default: exactly the declared parameters."""
        missing = set(self.params) - set(args)
        extra = set(args) - set(self.params)
        if missing or extra:
            return False, f"expected {sorted(self.params)}, got {sorted(args)}"
        return True, ""

    def preconditions(self, world, **args):
        """-> (ok, reason, message), checked on the world model before starting."""
        return True, Reason.NONE, ""

    def reset(self, world, **args):
        """Clear internal state and bind the arguments for this run."""

    def step(self, world):
        """-> (arm joint targets, gripper ctrl) for one control step."""
        raise NotImplementedError

    def check(self, world):
        """-> Check. Observable signals only; called after every step."""
        return Check()

    def apply_effects(self, world, **args):
        """Update the world model after success."""


class Runner:
    """Owns the control loop: perception update, step, termination, timeout."""

    def __init__(self, env, world, timeout_steps=700, on_step=None):
        self.env = env
        self.world = world
        self.timeout_steps = timeout_steps
        self.on_step = on_step  # on_step(skill, t, check), e.g. for video

    def execute(self, skill, **args):
        world = self.world

        def result(status, reason=Reason.NONE, message="", steps=0):
            return Result(skill.name, args, status, reason, message, steps, world.snapshot())

        world.observe()  # fresh perception before deciding whether to start
        ok, msg = skill.validate(world, **args)
        if not ok:
            return result(Status.FAILURE, Reason.INVALID_ARGS, msg)
        ok, reason, msg = skill.preconditions(world, **args)
        if not ok:
            return result(Status.FAILURE, reason, msg)
        ok, msg = skill.start_state.check(world)
        if not ok:
            return result(Status.FAILURE, Reason.START_STATE,
                          f"{msg}; {skill.name} starts from: {skill.start_state}")

        skill.reset(world, **args)
        for t in range(self.timeout_steps):
            world.update()
            arm, grip = skill.step(world)
            self.env.step(arm, grip, n_substeps=DEC)
            world.after_action(grip)
            c = skill.check(world)
            if self.on_step is not None:
                self.on_step(skill, t, c)
            if c.status != Status.RUNNING:
                if c.status == Status.SUCCESS:
                    skill.apply_effects(world, **args)
                return result(c.status, c.reason, c.message, t + 1)
        return result(Status.TIMEOUT, Reason.TIMEOUT,
                      f"no success or failure within {self.timeout_steps} steps",
                      self.timeout_steps)
