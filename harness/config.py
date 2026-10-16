"""Harness switches. Each one is a component that can be ablated.

Read from $HARNESS_CONFIG (JSON) so lerobot-eval's own command line stays
unchanged. All off = M0: the VLA with the task's full instruction, exactly
what lerobot-eval runs.
"""

import json
import os
from dataclasses import asdict, dataclass


@dataclass
class HarnessConfig:
    # M1/M2: run the task as a list of step instructions (harness/plans.py)
    # instead of the full instruction.
    steps: bool = False
    # step wording: "trained" (phrasing from SmolVLA's single-object training
    # tasks), "split" (the task's sentence cut at "and") or "libero90" (the
    # same-scene LIBERO-90 instruction, where one exists)
    wording: str = "trained"
    # M1: run only this step of the plan (0-based), alone from the task's
    # initial state, for the whole step budget
    only_step: int = None
    # M2: a step counts as done once its check has held for this many
    # consecutive control steps (20 Hz), then the next step's instruction is
    # given and the VLA's action queue cleared
    confirm_steps: int = 5
    # M4: before each step the harness checks the skill's start state (hand at
    # the episode's start pose, gripper open; harness/start_state.py) and, only
    # if it fails, drives the hand back there first (harness/recover.py)
    recover: bool = False
    # M3: retry a step that has not been confirmed within step_timeout control
    # steps of starting (95% of successful steps finish by ~153), at most
    # max_retries times per step and only while at least min_budget steps are
    # left (a recovery + a typical step); a finished step whose check stops
    # holding is gone back to (undo check). Retries go through the start-state
    # check, so they recover first when recover is on.
    retry: bool = False
    step_timeout: int = 160
    max_retries: int = 1
    min_budget: int = 160

    @classmethod
    def from_env(cls):
        raw = os.environ.get("HARNESS_CONFIG", "").strip()
        cfg = cls(**json.loads(raw)) if raw else cls()
        if cfg.wording not in ("trained", "split", "libero90"):
            raise ValueError(f"wording must be trained, split or libero90, got {cfg.wording!r}")
        return cfg

    def describe(self):
        if not self.steps:
            return "harness off (M0: full instruction)"
        if self.only_step is not None:
            return f"M1: step {self.only_step} alone, {self.wording} wording"
        parts = ["steps in sequence"] + (["recovery pose"] if self.recover else []) + (
            [f"retry (timeout {self.step_timeout}, max {self.max_retries})"] if self.retry else [])
        tag = "M3" if self.retry else ("M4" if self.recover else "M2")
        return f"{tag}: {' + '.join(parts)}, {self.wording} wording"
