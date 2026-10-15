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
    # M1/M2: run the task as a list of step instructions (orchestrator plans)
    # instead of the full instruction.
    steps: bool = False

    @classmethod
    def from_env(cls):
        raw = os.environ.get("HARNESS_CONFIG", "").strip()
        cfg = cls(**json.loads(raw)) if raw else cls()
        return cfg

    def describe(self):
        on = [k for k, v in asdict(self).items() if v]
        return "harness off (M0: full instruction)" if not on else "harness: " + ", ".join(on)
