"""The harness's rollout loop, a drop-in for lerobot_eval.rollout.

Same signature and return value as LeRobot's rollout, so lerobot-eval's
seeding, batching, scoring and videos stay exactly the same; what changes is
who decides what the VLA is told and when. Each env runs an Episode: a plan
(a list of Steps, each an instruction) and the index of the current step.

With every switch in HarnessConfig off (M0) the plan is one step, the task's
own instruction, and the loop must reproduce lerobot-eval episode for episode.

M2 (sequencing): after every control step the verifier evaluates the current
step's check (a LIBERO goal predicate, read from the simulator: privileged,
so runs with it are upper bounds); once it has held for confirm_steps steps,
the episode moves to the next step and the VLA's action queue is cleared. The
VLA keeps one queue for the whole batch, so sequencing needs batch size 1.
After the last step it keeps the last instruction until the episode ends.
M4 (recover): between steps the hand is first driven back to its start pose
(harness/recover.py); those control steps count against the budget.
Step switches are written to $HARNESS_LOG (JSON lines, one per episode).
"""

import json
import os

from dataclasses import dataclass, field

import numpy as np
import torch
from tqdm import trange

from lerobot.envs import check_env_attributes_and_types
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.utils.constants import ACTION

from harness.config import HarnessConfig
from harness.plans import PLANS
from harness.recover import Recovery, start_pose
from harness.vla_skill import VLASkill


@dataclass
class Step:
    instruction: str
    check: tuple = None  # goal predicate that verifies the step (None: no check)


@dataclass
class Episode:
    plan: list
    task_id: int = None
    current: int = 0
    held: int = 0  # consecutive control steps the current step's check has held
    start: tuple = None  # hand pose at the start of the episode (recovery target)
    recovering: object = None  # active Recovery, if any
    log: list = field(default_factory=list)  # (control step, event)

    @property
    def instruction(self):
        return self.plan[self.current].instruction

    @property
    def step(self):
        return self.plan[self.current]


def check_holds(env, i, check):
    """Evaluate a LIBERO goal predicate on env i's simulator state (privileged)."""
    inner = env.envs[i]._env.env
    return bool(inner._eval_predicate(list(check)))


def task_descriptions(env):
    try:
        return list(env.call("task_description"))
    except (AttributeError, NotImplementedError):
        try:
            return list(env.call("task"))
        except (AttributeError, NotImplementedError):
            return [""] * env.num_envs


def make_plan(task, task_id, cfg):
    if not cfg.steps:
        return [Step(task)]
    steps = [Step(s.instruction(cfg.wording), s.check) for s in PLANS[task_id]]
    if cfg.only_step is not None:
        if cfg.only_step >= len(steps):
            raise ValueError(f"task {task_id} has {len(steps)} steps; no step {cfg.only_step}")
        steps = [steps[cfg.only_step]]
    return steps


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
    task_ids = list(env.get_attr("task_id"))
    episodes = [Episode(make_plan(t, tid, cfg), tid) for t, tid in zip(task_descriptions(env), task_ids)]
    sequencing = any(len(e.plan) > 1 for e in episodes)
    if sequencing and n > 1:
        raise ValueError("sequencing steps needs --eval.batch_size=1 (the VLA's action queue is shared)")
    for i, e in enumerate(episodes):
        e.start = start_pose(env, i)
    if cfg.steps:
        print(f"[harness] {cfg.describe()}: " + " | ".join(
            sorted({f"task {tid}: {e.instruction!r}" for tid, e in zip(task_ids, episodes)})), flush=True)

    all_actions, all_rewards, all_successes, all_dones = [], [], [], []
    step = 0
    done = np.array([False] * n)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(max_steps, desc=f"Harness rollout ({cfg.describe()}), at most {max_steps} steps",
                     disable=False, leave=False)
    check_env_attributes_and_types(env)
    while not np.all(done) and step < max_steps:
        if any(e.recovering for e in episodes):  # batch size 1 when sequencing
            action = np.stack([e.recovering.act(env, i) for i, e in enumerate(episodes)])
        else:
            action = skill.act(observation, [e.instruction for e in episodes])
        observation, reward, terminated, truncated, info = env.step(action)
        if render_callback is not None:
            render_callback(env)
        successes = successes_from_info(info, n)

        if sequencing:
            for i, e in enumerate(episodes):
                if e.recovering is not None:
                    if e.recovering.done(env, i):
                        e.log.append((step + 1, f"recovered in {e.recovering.steps} steps"))
                        e.recovering = None
                        skill.reset()
                    continue
                if done[i] or e.current >= len(e.plan) - 1 or e.step.check is None:
                    continue
                e.held = e.held + 1 if check_holds(env, i, e.step.check) else 0
                if e.held >= cfg.confirm_steps:
                    e.log.append((step + 1, f"step {e.current} done"))
                    e.current += 1
                    e.held = 0
                    skill.reset()  # batch size 1: only this episode's queue
                    if cfg.recover:
                        e.recovering = Recovery(*e.start)

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

    log_path = os.environ.get("HARNESS_LOG")
    if log_path and sequencing:
        with open(log_path, "a") as f:
            for i, e in enumerate(episodes):
                f.write(json.dumps({"task_id": e.task_id, "seed": seeds[i] if seeds else None,
                                    "plan": [s.instruction for s in e.plan], "reached_step": e.current,
                                    "events": e.log, "steps": step,
                                    "success": bool(torch.stack(all_successes, dim=1)[i].any())}) + "\n")

    return {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
