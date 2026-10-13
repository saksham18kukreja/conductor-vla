"""Headless check of the LIBERO-Long evaluation pipeline, no policy and no GPU.

Uses LeRobot's LiberoEnv (the wrapper lerobot-eval scores with) under
MUJOCO_GL=osmesa (CPU rendering, no display) and checks, per task:

  1. both cameras render (non-blank images of the expected size)
  2. LIBERO's success predicate is False after reset (fixed init state 0)
  3. a zero-motion action steps the sim and keeps it False
  4. for the pick-and-place tasks, teleporting the objects named in the goal
     into their target region makes the predicate True: the metric reacts
     to the goal, so a harness can be scored with the same code

Run from ~/experiments/lerobot (its env has LIBERO):
  MUJOCO_GL=osmesa LIBERO_CONFIG_PATH=$PWD/.libero \\
      .env/bin/python ../franka_cube_lift/libero/check_headless.py
"""

import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv

SUITE = "libero_10"
# task -> [(object, target object, height above target centre in m)]
TELEPORT = {
    0: [("alphabet_soup_1", "basket_1", 0.06), ("tomato_sauce_1", "basket_1", 0.06)],
    1: [("cream_cheese_1", "basket_1", 0.06), ("butter_1", "basket_1", 0.06)],
    7: [("alphabet_soup_1", "basket_1", 0.06), ("cream_cheese_1", "basket_1", 0.06)],
}


def inner(env):
    """The robosuite problem env inside LeRobot's wrapper."""
    return env._env.env


def teleport(env, obj, target, dz, k):
    sim = inner(env).sim
    tgt = sim.data.body_xpos[sim.model.body_name2id(f"{target}_main")].copy()
    jnt = sim.model.get_joint_qpos_addr(f"{obj}_joint0")
    q = sim.data.qpos
    # spread two objects apart inside the container
    q[jnt[0]:jnt[0] + 3] = tgt + np.array([0.0, (-0.03, 0.03)[k % 2], dz])
    q[jnt[0] + 3:jnt[0] + 7] = [1, 0, 0, 0]
    sim.forward()


def main():
    suite = benchmark.get_benchmark_dict()[SUITE]()
    tasks = [int(a) for a in sys.argv[1:]] or list(range(suite.n_tasks))
    ok_all = True
    for tid in tasks:
        t0 = time.time()
        env = LiberoEnv(suite, tid, SUITE)
        obs, info = env.reset(seed=0)
        imgs = obs["pixels"]
        shapes = {k: v.shape for k, v in imgs.items()}
        blank = {k: float(v.std()) < 5 for k, v in imgs.items()}
        s0 = env._env.check_success()
        for _ in range(5):
            obs, _, _, _, info = env.step(np.zeros(7, dtype=np.float32))
        s1 = info["is_success"]
        line = (f"task {tid}: '{env.task}' | cameras {shapes} blank {blank} | "
                f"success after reset {s0}, after 5 steps {s1}")
        good = not any(blank.values()) and not s0 and not s1
        if tid in TELEPORT:
            for k, (obj, tgt, dz) in enumerate(TELEPORT[tid]):
                teleport(env, obj, tgt, dz, k)
            for _ in range(40):  # let them settle
                obs, _, _, _, info = env.step(np.zeros(7, dtype=np.float32))
            line += f" | after teleporting into {TELEPORT[tid][0][1]}: success {info['is_success']}"
            good = good and info["is_success"]
        print(line + f" | {time.time() - t0:.0f} s | {'OK' if good else 'FAIL'}", flush=True)
        if os.environ.get("SAVE_FRAMES"):
            import imageio
            imageio.imwrite(os.path.join(os.environ["SAVE_FRAMES"], f"task{tid}.png"),
                            np.concatenate([imgs["image"], imgs["image2"]], axis=1))
        ok_all &= good
        env.close()
    print("ALL OK" if ok_all else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
