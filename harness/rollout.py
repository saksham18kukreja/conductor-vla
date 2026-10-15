"""The harness's rollout loop, a drop-in for lerobot_eval.rollout.

Same signature and return value as LeRobot's rollout, so lerobot-eval's
seeding, batching, scoring and videos stay exactly the same; what changes is
who decides what the VLA is told and when. Each env runs an Episode: a plan
(a list of Steps, each an instruction) and the index of the current step.

With every switch in HarnessConfig off (M0) the plan is one step, the task's
own instruction, and the loop must reproduce lerobot-eval episode for episode.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import trange

from lerobot.envs import check_env_attributes_and_types
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.utils.constants import ACTION

from harness.config import HarnessConfig
from harness.vla_skill import VLASkill


@dataclass
class Step:
    instruction: str


@dataclass
class Episode:
    plan: list
    current: int = 0
    log: list = field(default_factory=list)

    @property
    def instruction(self):
        return self.plan[self.current].instruction


def task_descriptions(env):
    try:
        return list(env.call("task_description"))
    except (AttributeError, NotImplementedError):
        try:
            return list(env.call("task"))
        except (AttributeError, NotImplementedError):
            return [""] * env.num_envs


def make_plan(task, cfg):
    if not cfg.steps:
        return [Step(task)]
    raise NotImplementedError("step plans come in M1/M2")


def successes_from_info(info, n):
    """Same extraction as lerobot-eval (gymnasium >= 1.0 and older)."""
    if "final_info" in info:
        fi = info["final_info"]
        if isinstance(fi, dict):
            s = fi.get("is_success", [False] * n)
            return s.tolist() if hasattr(s, "tolist") else [bool(s)] * n
        return [bool(x["is_success"]) if isinstance(x, dict) and "is_success" in x else False
                for x in fi]
    if "is_success" in info:
        s = info["is_success"]
        return s.tolist() if hasattr(s, "tolist") else [bool(s)] * n
    return [False] * n


def harness_rollout(env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor,
                    seeds=None, return_observations=False, render_callback=None, **_unused):
    if return_observations:
        raise NotImplementedError("harness rollout does not return observations")
    cfg = HarnessConfig.from_env()
    skill = VLASkill(policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor)
    n = env.num_envs

    skill.reset()
    observation, info = env.reset(seed=seeds, options={NEW_ROLLOUT_OPTION: True})
    if render_callback is not None:
        render_callback(env)
    episodes = [Episode(make_plan(t, cfg)) for t in task_descriptions(env)]

    all_actions, all_rewards, all_successes, all_dones = [], [], [], []
    step = 0
    done = np.array([False] * n)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(max_steps, desc=f"Harness rollout ({cfg.describe()}), at most {max_steps} steps",
                     disable=False, leave=False)
    check_env_attributes_and_types(env)
    while not np.all(done) and step < max_steps:
        action = skill.act(observation, [e.instruction for e in episodes])
        observation, reward, terminated, truncated, info = env.step(action)
        if render_callback is not None:
            render_callback(env)
        successes = successes_from_info(info, n)

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)
        all_actions.append(torch.from_numpy(action))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))
        step += 1
        rate = torch.stack(all_successes, dim=1).any(dim=1).float().mean().item()
        progbar.set_postfix({"running_success_rate": f"{rate * 100:.1f}%"})
        progbar.update()
    progbar.close()

    return {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
