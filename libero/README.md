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

### Step instructions need same-scene training data

`lerobot/smolvla_libero` mostly ignores step instructions (13.7% of steps
done when given one step alone; harness M1): it never saw single steps in
the LIBERO-Long scenes. `Kesvill/smolvla_libero_90`, a SmolVLA trained on
LIBERO-90 (single-step tasks in the same scenes), scores 0% on LIBERO-Long
with the full instruction but follows step instructions: 82.4% on the 15
steps whose instruction exists in LIBERO-90 for that scene, 0% on the 3 that
don't (`run_kesvill.sh`, `results/libero/kesvill_l90/`). Run it with
`--policy.n_action_steps=10` (default 1: same success, 3x slower).

Per-subgoal logging (same arguments as `lerobot-eval`):

```bash
cd ~/experiments/lerobot
MUJOCO_GL=egl LIBERO_CONFIG_PATH=$PWD/.libero SUBGOAL_LOG=subgoals.jsonl \
    .env/bin/python ../franka_cube_lift/libero/eval_with_subgoals.py <lerobot-eval args>
python ../franka_cube_lift/libero/analyze_subgoals.py subgoals.jsonl
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

### Where SmolVLA fails: per-subgoal breakdown

A LIBERO task succeeds when all predicates of its goal hold at once.
`eval_with_subgoals.py` (lerobot-eval with each predicate logged per step)
reran the baseline with identical per-episode outcomes on all 3 seeds;
`analyze_subgoals.py` sorts the 300 episodes:

| Outcome | Share |
|---|---|
| success (all predicates) | 54.0% |
| partial (some predicates held at the end) | 24.3% |
| none (no predicate ever held) | 20.3% |
| undone (a predicate held, then was lost) | 1.3% |

| Task | Success | Partial (what got done) | None | Where it fails |
|---|---|---|---|---|
| 0 soup + sauce → basket | 8/30 | 6 (soup 5, sauce 1) | 16 | mostly the first object |
| 1 cream cheese + butter → basket | 17/30 | 8 (butter 6, cheese 2) | 5 | second object |
| 2 stove on + moka pot on it | 23/30 | 7 (stove on) | 0 | placing the pot (stove on 100%) |
| 3 bowl in drawer + close it | 23/30 | 7 (bowl in) | 0 | closing the drawer (bowl in 100%) |
| 4 two mugs → two plates | 7/30 | 12 (one mug each way, 6/6) | 9 | second mug, often the first |
| 5 book → caddy | 26/30 | — | 4 | single step |
| 6 mug → plate, pudding right of it | 10/30 | 14 (mug 11, pudding 3) | 5 | the pudding |
| 7 soup + cream cheese → basket | 8/30 | 3 (soup) | 19 | mostly the first object |
| 8 both moka pots → stove (+ on) | 15/30 | 15 (stove on always; one pot 11) | 0 | the second pot |
| 9 mug in microwave + close it | 25/30 | 1 | 3 | — |

What it says for the harness:

- **A quarter of all episodes finish part of the task and then stall.** In
  the stove, drawer and moka-pot tasks the first step succeeds every time
  and the failure is the next one. Sequencing, a verifier that sees which
  step is missing, and a retry of just that step target exactly this.
- **SmolVLA almost never undoes progress** (1.3%), so a retry of the missing
  step starts from a sensible state.
- **The basket tasks with the soup can (0, 7) mostly fail at the first
  object** (16 and 19 of 30 do nothing). That is a skill failure (grasping
  that object), not a sequencing one; the harness needs a reliable pick.
