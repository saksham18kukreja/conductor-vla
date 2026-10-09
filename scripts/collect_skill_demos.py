"""Demonstrations for the goal-conditioned skills (place, push).

Same idea as collect_demos.py (privileged expert, student-visible
observation logged separately), with the start states a skill really sees:

  place  the cube held as pick leaves it: the pick expert runs until pick's
         own success condition holds, plus 0-40 extra steps so the handoff
         point varies; then ScriptedPlace to a random target
  push   arm home, cube on the table, target 5-15 cm away (ScriptedPush)

Observation: skills/features.goal_obs (29-d), from the true cube position,
or with --obj-noise S a perturbed one: like a 10 Hz detector, an error drawn
every 5 steps and held in between, with sigma drawn per episode from
U(0, S) mm. The expert still acts on the truth, so the labels stay correct
and the policy learns to ignore estimate noise (place v1.0 trained on exact
positions went from 38/40 to 1/40 with 1 mm of noise).
With --joint-noise R the executed joint targets get N(0, R) rad noise while
the label stays the expert's clean action (DART, as collect_demos.py
--noise), so the data shows how to recover from small deviations, including
along the arm's redundant direction.
Action: joint-target delta + gripper (0 closed, 1 open), as for pick.
Episodes run in parallel worker processes; episode i always uses rng seed
(seed, i), so a run is reproducible for any --workers.

  python collect_skill_demos.py --skill place --episodes 3000 --out ../demos/place3000.npz
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scenarios as sc  # noqa: E402
from collect_demos import ACT_LABELS, DEC, GRIPPER_SCALE  # noqa: E402
from env import FrankaCubeLift  # noqa: E402
from experts import ScriptedPlace, ScriptedPush  # noqa: E402
from skills.features import GOAL_OBS_LABELS, goal_obs  # noqa: E402

EXTRA_HANDOFF_STEPS = 40
_env = None


def _init():
    global _env
    _env = FrankaCubeLift()


def start_state(skill, env, rng):
    """Set up the scene; -> (expert, goal_xy, start_xy) or None if setup failed."""
    xy = sc.cube_xy(rng)
    env.reset(cube_xy=xy)
    if skill == "place":
        if not sc.scripted_pick(env, extra_steps=int(rng.integers(0, EXTRA_HANDOFF_STEPS + 1))):
            return None
        goal = sc.place_target(rng)
        return ScriptedPlace(env, goal), goal, xy
    goal = sc.push_target(rng, xy)
    return ScriptedPush(env, goal), goal, xy


NOISE_EVERY = 5  # control steps per simulated detection (10 Hz)


def episode(job):
    skill, seed, i, max_steps, obj_noise, joint_noise = job
    env = _env
    rng = np.random.default_rng([seed, i])
    setup = start_state(skill, env, rng)
    if setup is None:
        return None
    pol, goal, start_xy = setup
    sigma = rng.uniform(0.0, obj_noise) / 1000.0
    noise = np.zeros(3)
    obs, act = [], []
    for t in range(max_steps):
        if t % NOISE_EVERY == 0:
            noise = rng.normal(0.0, sigma, 3)
        o = goal_obs(env, env.cube_pos + noise, goal)
        q = env.arm_qpos
        arm, grip = pol.act()
        obs.append(o)
        act.append(np.concatenate([arm - q, [grip / GRIPPER_SCALE]]).astype(np.float32))
        if joint_noise > 0:
            arm = np.clip(arm + rng.normal(0.0, joint_noise, 7),
                          env.arm_limits[:, 0], env.arm_limits[:, 1])
        env.step(arm, grip, n_substeps=DEC)
        if pol.done:
            break
    err = float(np.linalg.norm(env.cube_pos[:2] - goal))
    resting = abs(env.cube_pos[2] - sc.REST_Z) < 0.003
    ok = (not pol.timed_out) and err < 0.015 and resting
    return dict(obs=np.array(obs), act=np.array(act), ok=ok, err=err,
                meta=(start_xy[0], start_xy[1], goal[0], goal[1], float(ok), err))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", choices=["place", "push"], required=True)
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--joint-noise", type=float, default=0.0,
                    help="std (rad) of noise on the executed joint targets (DART)")
    ap.add_argument("--obj-noise", type=float, default=0.0,
                    help="max sigma (mm) of the object-position noise in the "
                         "observation (see module doc); 0 = exact positions")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    t0 = time.time()
    jobs = [(args.skill, args.seed, i, args.max_steps, args.obj_noise, args.joint_noise)
            for i in range(args.episodes)]
    with Pool(args.workers, initializer=_init) as pool:
        eps = pool.map(episode, jobs, chunksize=8)

    n_setup = sum(e is None for e in eps)
    kept = [e for e in eps if e is not None and e["ok"]]
    n_fail = len(eps) - n_setup - len(kept)
    lengths = np.array([len(e["obs"]) for e in kept])
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    obs = np.concatenate([e["obs"] for e in kept])
    act = np.concatenate([e["act"] for e in kept])
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out, obs=obs, act=act, episode_starts=starts, episode_lengths=lengths,
        meta=np.array([e["meta"] for e in kept], dtype=np.float32),
        meta_labels=np.array(["start_x", "start_y", "goal_x", "goal_y", "ok", "err"]),
        obs_labels=np.array(GOAL_OBS_LABELS), act_labels=np.array(ACT_LABELS),
        skill=np.array(args.skill),
        collection=np.array({"skill": args.skill, "episodes_requested": args.episodes,
                             "seed": args.seed, "extra_handoff_steps": EXTRA_HANDOFF_STEPS,
                             "obj_noise_mm": args.obj_noise,
                             "joint_noise_rad": args.joint_noise}),
    )
    errs = 1000 * np.array([e["err"] for e in kept])  # expert's final error
    print(f"{args.skill}: requested {args.episodes}, kept {len(kept)}, expert failed {n_fail}, "
          f"setup failed {n_setup}")
    print(f"transitions {len(obs)}  obs {obs.shape}  length mean {lengths.mean():.0f} "
          f"max {lengths.max()}  final error median {np.median(errs):.1f} mm "
          f"max {errs.max():.1f} mm")
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
