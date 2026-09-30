"""Skill library. See skills/base.py for the abstraction and SKILLS.md."""

from skills.base import Check, Reason, Result, Runner, Skill, Status  # noqa: F401
from skills.pick import PickSkill

LIBRARY = {
    PickSkill.name: PickSkill,
}
