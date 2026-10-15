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
    # tasks) or "split" (the task's sentence cut at "and")
    wording: str = "trained"
    # M1: run only this step of the plan (0-based), alone from the task's
    # initial state, for the whole step budget
    only_step: int = None

    @classmethod
    def from_env(cls):
        raw = os.environ.get("HARNESS_CONFIG", "").strip()
        cfg = cls(**json.loads(raw)) if raw else cls()
        if cfg.wording not in ("trained", "split"):
            raise ValueError(f"wording must be trained or split, got {cfg.wording!r}")
        return cfg

    def describe(self):
        if not self.steps:
            return "harness off (M0: full instruction)"
        if self.only_step is not None:
            return f"M1: step {self.only_step} alone, {self.wording} wording"
        return f"steps, {self.wording} wording"
