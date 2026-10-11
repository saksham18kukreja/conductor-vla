"""Skill library. See skills/base.py for the abstraction and SKILLS.md."""

from skills.base import Check, Reason, Result, Runner, Skill, Status  # noqa: F401
from skills.home import HomeSkill, RecoverSkill
from skills.pick import PickSkill
from skills.place import PlaceSkill
from skills.push import PushSkill

LIBRARY = {
    PickSkill.name: PickSkill,
    PlaceSkill.name: PlaceSkill,
    PushSkill.name: PushSkill,
    HomeSkill.name: HomeSkill,
    RecoverSkill.name: RecoverSkill,
}
