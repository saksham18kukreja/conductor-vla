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

- **Rendering:** EGL (GPU, headless) is the default in `run_libero_eval.sh`:
  97.7 env steps/s with both cameras, vs 47.8 with GLX and 10.5 with OSMesa.
  On 2026-10-12 EGL failed (`nvidia-smi`: driver/library mismatch after a
  driver update); a reboot fixed it. If it breaks again, run with
  `MUJOCO_GL=glx` (needs the X display) or `osmesa` (CPU): robosuite forces
  EGL for any other value.
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
LeRobot CI, EGL rendering, batches of 5. Run 2026-10-14 (~17 min per seed);
raw results in `results/libero/smolvla_libero10_s*.json`.

| Task | Instruction | Seeds 1000 / 2000 / 3000 | Mean |
|---|---|---|---|
| 0 | put both the alphabet soup and the tomato sauce in the basket | 40 / 30 / 10% | 26.7% |
| 1 | put both the cream cheese box and the butter in the basket | 70 / 80 / 20% | 56.7% |
| 2 | turn on the stove and put the moka pot on it | 70 / 90 / 70% | 76.7% |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 70 / 80 / 80% | 76.7% |
| 4 | put the white mug on the left plate and the yellow and white mug on the right plate | 20 / 30 / 20% | 23.3% |
| 5 | pick up the book and place it in the back compartment of the caddy | 100 / 70 / 90% | 86.7% |
| 6 | put the white mug on the plate and the chocolate pudding to the right of the plate | 10 / 50 / 40% | 33.3% |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 40 / 20 / 20% | 26.7% |
| 8 | put both moka pots on the stove | 20 / 80 / 50% | 50.0% |
| 9 | put the yellow and white mug in the microwave and close it | 90 / 80 / 80% | 83.3% |
| **All** | | **53 / 61 / 48%** | **54.0%** |

Published reference points (other checkpoints and setups): SmolVLA 77%
(paper), pi0.5 92.4% (OpenPI) / 96% (LeRobot). This checkpoint is 23 points
below the paper's number; possible reasons are the checkpoint (25k steps,
batch 32, trained on all four suites) and the protocol (10 episodes per
task). Either way, 54.0% is the number our harness is compared against.

The tasks that move two objects (0, 1, 4, 6, 7, 8) are the weakest
(23–57%); the single-object tasks, including the articulated ones (2, 3,
5, 9), reach 77–87%.
