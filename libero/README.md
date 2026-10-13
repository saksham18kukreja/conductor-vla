# LIBERO-Long: harness vs VLA

The harness (orchestrator + skills + verifier + recovery) is evaluated on
LIBERO-Long (`libero_10`, 10 multi-step household tasks, robosuite/MuJoCo,
Franka Panda) and compared against VLAs on the same tasks and scoring. Then
LIBERO-Plus (perturbed robot start states, camera, layout, ...). See PLAN.md.

## Setup (this laptop)

LeRobot in its own env (needs Python 3.12), next to this repo:

```bash
cd ~/experiments
git clone https://github.com/huggingface/lerobot.git
conda create -p lerobot/.env python=3.12
lerobot/.env/bin/pip install -e "lerobot[smolvla,libero]"
cp franka_cube_lift/libero/run_libero_eval.sh lerobot/
# LIBERO asks a question on first import; answer "n" once to write its config
cd lerobot && LIBERO_CONFIG_PATH=$PWD/.libero sh -c 'echo n | .env/bin/python -c "import libero.libero"'
```

## Headless check (no display, no policy)

`check_headless.py` runs LeRobot's `LiberoEnv` (the wrapper `lerobot-eval`
scores with) with OSMesa CPU rendering and checks every LIBERO-Long task:
both cameras render, LIBERO's success predicate is False after reset and
after stepping, and for tasks 0, 1 and 7 teleporting the goal objects into
the basket makes it True.

```bash
cd ~/experiments/lerobot
env -u DISPLAY MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LIBERO_CONFIG_PATH=$PWD/.libero \
    .env/bin/python ../franka_cube_lift/libero/check_headless.py      # ALL OK (2026-10-12)
```

Speed with both cameras: 10.5 env steps/s headless (OSMesa) vs 47.8 with GLX,
so a 520-step episode costs up to ~50 s headless. Camera images come out
upside down (as LIBERO renders them; the VLA checkpoints are trained on the
same), so harness perception must flip them.

`lerobot-eval` runs headless the same way (`env -u DISPLAY MUJOCO_GL=osmesa`).

Pitfalls found on the way:

- **Rendering:** EGL fails here (`nvidia-smi`: driver/library mismatch), and
  robosuite forces EGL unless `MUJOCO_GL` is `glx` or `osmesa`.
  `run_libero_eval.sh` renders with GLX through the X display.
- **Memory:** 10 parallel envs (async or in one process) exhaust the 15 GB
  and stall or crash. Batches of 5 in one process (`--eval.batch_size=5
  --eval.use_async_envs=false`) peak at ~1.6 GB free.
- **Checkpoint:** use `lerobot/smolvla_libero` (what LeRobot's CI tests),
  with its camera mapping and `--policy.empty_cameras=1`. The older
  `HuggingFaceVLA/smolvla_libero` (n_action_steps 1, different camera
  keys) got 1/7 on tasks 0-1 vs 5/10, and runs ~6x slower.

## Baseline: SmolVLA

`run_baseline.sh`: `lerobot/smolvla_libero`, LIBERO-Long, 10 episodes per
task (fixed LIBERO init states), seeds 1000 / 2000 / 3000, settings from
LeRobot CI. Not run yet (stopped on 2026-10-12); probe only: 5/10 episodes on
tasks 0-1. Published reference points (other checkpoints and setups):
SmolVLA 77% (paper), pi0.5 92.4% (OpenPI) / 96% (LeRobot).
