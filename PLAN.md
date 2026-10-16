# Project plan

Living roadmap. Read it before starting an experiment and update it after one:
tick boxes, add a Log row, and move items between phases as priorities change.
README.md shows the results; this file holds the direction, the rules, and
what's next.

## Goal

Build a **harness for an agentic robotics system**. A high-level model (a
VLA, VLM or LLM planner) acts as the orchestrator. Around it sit the same
pieces that made agentic AI work: skills (specialised policies), memory,
verification and recovery.

Build it one component at a time, as with the cube-lift policy, and measure
each addition. The point isn't only to build a working system but to find
**where this approach hits its limits** and which component is the bottleneck.

| Agentic AI | Robotics harness |
|---|---|
| LLM orchestrator | VLA / VLM / LLM planner choosing the next step |
| Tools / skills | Specialised policies (e.g. cube lift), motion planners, grasp detectors |
| Memory | World state (objects, poses, progress), episode history, skill track records |
| Tests / error messages | Success detectors, failure monitors |
| Retry / replan | Recovery behaviours, replanning after a failed skill |

```
            task (language)
                  │
          ┌───────▼────────┐        ┌──────────────┐
          │  orchestrator   │◄──────►│    memory     │
          │ (planner / VLA) │        │ world state,  │
          └───────┬────────┘        │ history       │
         skill call│  ▲ result      └──────▲───────┘
          ┌───────▼──┴─────┐               │
          │ skill library   │── state ─────┘
          │ lift, place, …  │
          └───────┬────────┘
       50 Hz actions│  ▲ observations
          ┌───────▼──┴─────┐
          │ robot / sim     │◄── verifier: did it work?
          └────────────────┘
```

## Expected bottlenecks

These are the hypotheses the phases are designed to test.

1. **Skill handoffs.** Skill B must start from wherever skill A stopped. BC
   policies are brittle outside their training states (v1.0 collapsed from a
   small drift), so chain success may fall well below the product of
   individual skill success rates.
2. **Verification.** A robot has to *perceive* failure, and physical mistakes
   can't be undone. The harness can only recover from failures it detects.
3. **Timescales.** Planners and VLAs run at roughly 1–10 Hz and skills at
   50 Hz, so passing control between the layers is a design problem.
4. **Grounding.** A planner can produce steps that make sense on paper but
   can't be carried out from the current physical state.
5. **Compute.** A laptop RTX 4060 (8 GB) runs small skills and small VLAs
   (~0.5B, e.g. SmolVLA) but can't fine-tune 7B models. The orchestrator can
   be an API model.

## Rules for every experiment

1. **One change at a time**, with a written hypothesis ("X fails because Y;
   changing Z should fix it").
2. **At least 3 seeds.** Single-seed results swung from 4% to 98% here. Report
   all seeds and the mean, never just the best.
3. **Judge by task success in rollouts**, not by training or validation loss.
4. **Diagnose before fixing.** Break failures into stages, and confirm the
   cause with an intervention before changing anything.
5. **Ablate every harness component.** Measure task success with and without
   it (memory, verifier, retries, planner). The difference is its measured
   contribution.
6. **Attribute every failure** to one of: plan, skill execution, handoff,
   perception, verification. Track the counts; they show where the
   bottleneck is.
7. **Keep defaults backward compatible.** New behaviour goes behind a flag.
8. **Log it.** Benchmark every new version with `scripts/benchmark.py
   --wandb` (one row per checkpoint in `results/benchmark.tsv`, a per-position
   record in `results/episodes/`, a regression report against the
   ground-truth baseline, and a W&B run), train with `train_bc.py --wandb`,
   put checkpoints in `models/`, and add a line to the Log below. Compare only
   results with the same `eval_set` and `protocol`.
9. **Versioning.** Minor bump for eval or rollout changes, major bump for a
   new model, input, loss, dataset or harness component. Tag the commit.

## Metrics

| Metric | Meaning |
|---|---|
| Skill success | One skill, from its standard start states |
| Handoff success | The same skill, started from another skill's end states |
| Chain success vs product | Multi-step task success compared with the product of the individual skill rates. The gap is the harness's loss. |
| Failure attribution | Share of failures from plan / execution / handoff / perception / verification |
| Recovery rate | Share of detected failures the harness recovers from |
| Verifier precision/recall | How often "success" is right, and how many failures it catches |
| Ablation delta | Task success with a component minus success without it |

## Where we are

**Skill #1: cube lift, v3.0.** State-based MLP, 21 inputs, BCE gripper,
steps-since-closed counter. **100% Success@1 on 3/3 seeds on the held-out
benchmark** (200 unseen positions, strict hold-for-1-s success; 600/600).

**v4.0: the same policy from camera input**: 99.3% Success@1, 99.8%
Success@3 on the held-out benchmark (fixed RGB-D camera + color detector).

**v5.1: pick without the hidden timer** (the pick skill's default): 20
inputs, no counter, 100% with true positions and 95% through the skill with
the camera, on 3/3 seeds.

Limits:

- A start-of-episode blind spot: the hand at its start pose hides cubes close
  to the robot (all v4.0 failures).
- The color detector relies on the cube being the only red object; a learned
  detector is needed once the scene has more objects.
- Narrow task: unrotated cube, no noise or disturbances, and positions only
  within the training range. A dropped cube can land outside it, and the
  policy can't recover from there (seen in v2.0 retries).

## Roadmap

### Phase 0: Evaluation you can trust (done)

The harness is only as good as its success signal, so this comes first.

- [x] **Held-out test set:** 200 new cube positions (rng seed 2026) used
      only to report versions. v1.0–v3.0 re-scored (`scripts/benchmark.py`).
- [x] **Strict success:** cube above the threshold for 50 consecutive steps
      (1 s) with both fingers in contact ("hold" stage).
- [x] **Stage score, Success@1/@3 with retries, mean attempts** (defined in
      README → Evaluation).
- [x] **Results log:** `results/benchmark.tsv`, one row per checkpoint, with
      the git commit. Plot: `scripts/plot_benchmark.py`.
- [ ] **Git tags** v1.0, v1.1, v2.0, v3.0.

Done when: v3.0 has test-set and strict-success numbers, both logged. **Met;
only the tags remain.**

### Phase 1: Perception (v4.0 done; one fix open)

Question: *how much does replacing the simulator's true cube position with a
camera and detector cost?*

Modular design: `table_cam` (RGB-D) → detector → 3D cube position → the
v3.0 policy, unchanged. Detection at 10 Hz; the last good estimate is kept when
the cube is hidden (the first bit of harness memory). Every result is compared
with true positions on the held-out benchmark.

- [x] **Error budget**: v3.0 tolerates ~2 mm random error and ~5 mm fixed
      offset; at σ 10 mm it drops to 64% (seed 2: 23.5%).
- [x] **Camera module**: fixed `table_cam`, RGB/depth/segmentation, pixel →
      3D; depth agrees with geometry to 0.56 mm.
- [x] **Color baseline detector**: 0.2 mm median error; detections with
      < 60% of the cube visible are rejected and the last estimate held.
- [x] **Integrate and benchmark** (v4.0): 99.3% Success@1, 99.8% Success@3
      from camera input; rejection/memory ablated (no measurable effect).
- [x] **Occlusion analysis**: occlusion only matters at the episode start
      (hand over cubes near the robot); rare along the policy's path.
- [x] **System video** with detector overlay (`make_bc_video.py --overlay`).
- [ ] **Look before acting**: don't start the skill without a valid
      detection; move the arm out of the camera's view first if needed. Fixes
      the remaining start-of-episode blind spot.
- Deferred: **learned detector (YOLO)**, until the scene has several or
  unfamiliar objects; the color detector is already 0.2 mm here.
- Not needed: **retraining with position noise**; the detector is well inside
  the error budget.

Done when: v3.0 runs from camera input on the benchmark, with success,
position error and failure rate logged for each perception mode. **Met
(v4.0).**

### Phase 2: Skill interface and library

Question: *what does a skill need to expose to be usable by a harness?*

- [x] **Skill API**: `start conditions → run → result`, with a success
      signal, failure signals, a timeout and a result object
      (status, final state, reason for failure). (v4.1)
- [x] **Wrap cube-lift v3.0** as the first skill. (v4.1)
- [x] **Remove the expert's hidden timer** (v5.1): the expert lifts on the
      first step the fingers reach the cube's width; no counter input. 100%
      with true positions, 95% with the camera (3 seeds), same as v3.0/v4.1.
      "Fingers stopped for 5 steps" (v5.0) still froze: any wait is a timer.
      Open: seed 2 after place drops to 94.5% (11 failures); diagnose, and
      run v3.0 s2 after_place for a baseline.
- [x] **Add 2–3 skills**: place, push (BC), home (scripted). See the
      2026-10-07 Log row and SKILLS.md. Full 3-seed camera benchmark only
      partly run (see "Open from 2026-10-07").
- [ ] **Handoff tests**: start each skill from other skills' end states,
      plus perturbed starts. Record skill success vs handoff success.
      Starts exist (`benchmark.py --start`, skill_tasks.py); partly run.

**Open from 2026-10-07** (stopped mid-benchmark):
- Run the 200-position benchmarks still missing: place color s2; place
  `--start pick` s1, s2; pick `--start after_place` s2.
- [x] Push after place: 74.8% (gt 71 / 75.5 / 78%, camera 70 / 75 / 75.5%) vs
  99.7% from home. Diagnosed 2026-10-12: the arm's start pose, not the
  cube's position (same scenes: 40/60 direct, 59/60 after recover). Push
  only trained from home; place leaves the hand above the cube with the
  wrist turned ~90° (q7 −2.40 vs −0.79). The policy's first move is the
  same for every target, so pushes toward −x (pusher must reach the far
  side) fail most (12/29; other directions 28/31). Closing the fingers
  first doesn't help (40/60); turning only the wrist back helps partly
  (50/60). Fixed (push v1.1): skills declare start states as data
  (`StartState`), the Runner checks them, and the recovery rule reaches
  them. Push declares arm at home + fingers open: gt 99.0 / 100 / 99.5%,
  camera 99.0 / 99.0 / 98.0%. Training push from varied starts stays an
  option if trips home get expensive in chains.
- Push camera failures beyond the start blind spot (30 refusals over 3
  seeds): 5 `object_dropped` (3 after reaching the target) and 1 false
  success (s1). Likely the hidden cube's estimate behind the pusher.
- [x] Pick after push: fixed in the orchestrator (option c, v4.2). Pick
  refuses closed fingers (`START_STATE`), and the recovery rule sends the arm
  to the recovery pose (home, fingers open) and retries: 35% → 100% (600/600,
  3 seeds). See SKILLS.md → Recovery pose.
- Commit (nothing from this session is committed; result rows carry the old
  commit 55e818d) and tag.

Done when: each skill has skill success and handoff success, logged.

### Phase 3: Harness v0, fixed sequences

Question: *how much does chaining lose, and why?*

- [x] **Multi-object scene** (2026-10-12): `multi_cube_scene.xml` (red,
      blue, green), objects by id in env and world model, one colour
      detector per cube (0.3 mm median), `benchmark.py --distractors N`.
- [x] **Skills in clutter** (single-object skills, 1–2 distractor cubes ≥ 8
      cm from the red cube, same positions and targets; 3 seeds, gt and
      camera). See the Log row. Findings:
      - pick unaffected (100% gt, 95% camera; never touches a distractor).
      - place loses ~4 points per distractor (gt 100 → 96.0 → 91.7%):
        fails when a distractor is within 40 mm of the target (53.5%),
        knocks it within 60 mm. Its "done" is wrong 0.5–2% of the time.
      - push still reaches its target (gt 99.5 / 99.0%) by bulldozing the
        distractor (knocked in 81% of episodes with one within 40 mm of the
        target) and doesn't report it. With the camera the red cube tilts
        ~5 mm against the distractor, the estimate adds a few mm, and push
        reports `object_dropped` (93.7 → 88.2 → 85.0%).
      - So skills need a spatial precondition (target area free) and the
        harness needs a verifier that checks side effects on other objects.
- [ ] **2–3 step tasks** in the multi-cube scene (sort into zones, clear a
      zone then place, rearrange).
- [ ] **Scripted orchestrator**: a fixed skill sequence, a verifier after
      each step, and retry on failure.
- [ ] **Measure** chain success vs product, and attribute every failure.
- [ ] **Ablate** the verifier and the retries.

### LIBERO-Long track (from 2026-10-12): harness vs VLA

Question: *on a standard benchmark VLAs report, does specialist skills +
harness beat an end-to-end VLA, and where does each fail?* Practical
household tasks instead of our own cube scenes; LIBERO-Plus next (robot
start states and camera shifts are where VLAs collapse: OpenVLA-OFT 97.1%
→ 31.9% / 56.4%, pi0 6.0% / 13.8%). Setup and pitfalls: `libero/README.md`.

The 10 tasks: 7 are pick-and-place only (0, 1, 4, 5, 6, 7, 8: into a
basket, onto a plate or stove, into a caddy compartment, "to the right of"
a plate); 3 also need an articulated object (2 stove knob, 3 close a drawer,
9 close the microwave).

Fairness rules (decide what a comparison claims):
1. Same test-time inputs as the VLA: the two cameras and proprioception.
   No simulator object state in the harness at test time; ground truth
   only for scoring and for labelling training data.
2. Report the training data used. LIBERO gives 50 human demos per task; any
   scripted demos are extra supervision and reported as a separate variant.
3. LIBERO's own success predicates (BDDL), its fixed init states, 10
   episodes per task, 3 seeds.

- [x] **Setup and headless check** (2026-10-12): LeRobot + LIBERO in their
      own env; `libero/check_headless.py` passes on all 10 tasks with OSMesa
      (no display, no GPU): cameras render, success predicate False at
      reset, True after teleporting goal objects (tasks 0, 1, 7). 10.5 env
      steps/s headless vs 47.8 with GLX.
- [x] **Baseline: SmolVLA** (`lerobot/smolvla_libero`) on LIBERO-Long, 3 seeds
      (2026-10-14, after a reboot fixed EGL and NVML): **54.0%** (53 / 61 /
      48%). Two-object tasks 23–57%, single-object tasks 77–87%. Per-task
      table in `libero/README.md`. Below the paper's 77% (different
      checkpoint/protocol); 54.0% is our comparison point.
- [x] **Task and failure analysis** (2026-10-14): every goal predicate logged
      per step (`libero/eval_with_subgoals.py`, identical per-episode outcomes
      to the baseline). 54.0% success, 24.3% partial (some predicates done,
      then stalled), 20.3% none, 1.3% undone. Stove, drawer and moka-pot
      tasks: the first step succeeds 100% and the next one fails. Basket
      tasks with the soup can (0, 7): mostly fail at the first object (16 and
      19 of 30 do nothing). So the harness can gain from sequencing,
      per-step verification and retrying the missing step, plus a reliable
      pick for the objects SmolVLA can't grasp.
- [ ] **Harness on LIBERO, SmolVLA as the skill** (decided 2026-10-14): the
      harness keeps its job (plan steps, verify each, retry, recover) and
      SmolVLA executes each step, called with a step instruction. A VLA
      never signals "done", so termination moves to the verifier. Code:
      `harness/` (`vla_skill.py`, `rollout.py` = a drop-in for lerobot-eval's
      rollout, `config.py` = one switch per component, `eval_harness.py`).
      Fairness: the whole task keeps the 520-step budget; rows that use
      LIBERO's predicates or hand-written plans are labelled upper bounds.
      Milestones, each switching on one component:
  - [x] **M0** loop only, harness off (full instruction): reproduces
        lerobot-eval episode for episode (tasks 0-1, seed 1000: 4/10 and
        7/10, identical success steps and predicate times).
  - [x] **M1** step instructions, each alone (2026-10-15): **SmolVLA mostly
        ignores them.** Commanded predicate held in 13.7% (trained wording,
        18 steps) and 3.0% (split wording, 9 steps), vs 64.4% / 57.4% for
        the same predicates with the full task instruction. It often does
        the other step instead (told cream cheese → butter 53%; told the
        right moka pot → left one 83%), and even a training-set instruction
        ("turn on the stove", 3%) fails in a new scene. The instruction acts
        like a task id, not a command, so step-by-step sequencing through
        language (M2–M6 as planned) won't work with this checkpoint.
        Plans in `harness/plans.py`, summary in `results/libero/m1/`.
        Options: (1) keep the full instruction and add verifier + retry +
        recovery around it (the 24% partial episodes), (2) fine-tune SmolVLA
        on step-segmented LIBERO-Long demos so it follows step instructions,
        (3) a VLA that follows language better (likely too big for 8 GB).
  - [x] **M1b: a SmolVLA trained on LIBERO-90** (2026-10-15). Why M1 failed:
        `lerobot/smolvla_libero` trained on 40 tasks (Spatial/Object/Goal/
        Long); our step wordings are in it verbatim, but only in other
        scenes, while the LIBERO-Long scenes only ever came with the full
        two-step sentence. LIBERO-90 (the training half of LIBERO-100) has
        single-step tasks in the same scenes. No larger SmolVLA exists for
        LIBERO (every checkpoint uses the 500M backbone). `Kesvill/
        smolvla_libero_90` (SmolVLA, ~1,900 LIBERO-90 demos, 100k steps; run
        with n_action_steps 10, same sanity success as 1): 28/30 on its own
        LIBERO-90 tasks; **0% on LIBERO-Long with the full instruction**
        (300 episodes, 30% partial); with LIBERO-90 step wording
        (`plans.py` `libero90`), **commanded step 68.7%** (vs 13.7% for
        smolvla_libero), other-object 4.1%; **82.4% on the 15 steps whose
        instruction exists in LIBERO-90 for that scene, 0% on the 3 that
        don't**. The language failure was a data gap, not the model.
        Product of step rates on the 6 fully covered tasks (0, 2, 3, 4, 6,
        7): 69.0% vs 43.9% for smolvla_libero on the same tasks (step 2
        measured from the initial scene, not after step 1). Results in
        `results/libero/kesvill_l90/`.
  - [x] **M2** + sequencing (2026-10-15): LIBERO-90 checkpoint, hand-written
        steps, a step is done when its predicate holds for 5 control steps,
        batch size 1; the 6 fully covered tasks (0, 2, 3, 4, 6, 7), 3 seeds.
        **0% (0/180)**: step 1 done 57–100%, step 2 never, and step 2 undoes
        step 1 in 23% (stove switched back off 19/30, first mug knocked off
        17/30). The same steps from the initial pose: 60–100% (M1b). The
        handoff bottleneck, at full strength.
  - [ ] **M3** + retry a failed step
  - [x] **M4** + recovery pose between steps (2026-10-15, done before M3
        because M2 showed the handoff was the failure): `harness/recover.py`
        drives the hand back to its start pose, gripper open, through
        LIBERO's own controller (28–39 control steps, inside the 520 budget).
        **56.1%** (56.7 / 56.7 / 55.0) vs 43.9% for smolvla_libero end to end
        on the same 6 tasks, and 0% without recovery. Per task: 23 / 77 / 87 /
        50 / 60 / 40% vs 27 / 77 / 77 / 23 / 33 / 27%. Below the 69% estimate
        from steps run alone: in the failures the arm reaches the next
        object but misses it, likely the changed scene (e.g. the stove now
        on, which the step's demos never showed). Upper bound: privileged
        predicates and hand-written plans; different training data from the
        end-to-end baseline (LIBERO-90 vs the 4 suites incl. LIBERO-Long).
        **Start-state contract** (2026-10-16): recovery is now triggered by
        a check, as in the cube harness, not by "a step finished":
        `harness/start_state.py` declares the VLA skill's start state (hand
        within 2 cm / 0.1 rad of the episode's start pose, gripper ≥ 70 mm
        open) and the harness checks it before every step, recovering only
        on failure. Same result: 56.1% (56.7 / 56.7 / 55.0); 180/180 first
        steps passed, 146 failed checks before step 2 all met after
        recovery.
  - [ ] **M5** visual verifier instead of predicates (first fair comparison)
  - [ ] **M6** LLM orchestrator instead of hand-written steps
  - [ ] **M7** LIBERO-Plus
- [ ] **Compare** per task, then LIBERO-Plus perturbations.

### Phase 4: Planner as orchestrator

Question: *are failures in the plan or in carrying it out?*

- [ ] **LLM/VLM planner** that calls skills as tools from a text state
      description, on language-specified tasks.
- [ ] **Task suite** with measurable success, including tasks that need
      reasoning (ordering, conditions).
- [ ] **Measure** plan failures vs execution failures; compare against the
      scripted orchestrator.

### Phase 5: Memory and image-based skills

Question: *what does memory buy you?*

- [ ] **Image-based skills** (end-to-end, images in, actions out), compared
      with the Phase 1 modular pipeline.
- [ ] **World-state memory**: object poses, task progress, skill history.
- [ ] **Ablate memory** on tasks with objects that go out of view or with
      long horizons.

### Phase 6: Recovery

Question: *which failures can the harness recover from, and which can't it?*

- [ ] **Replanning** on verifier failure.
- [ ] **Disturbances**: bump the cube mid-skill, drop it during lift.
- [ ] **Recovery data**: DART-style noisy demos (`collect_demos.py --noise`).
- [ ] **Measure** recovery rate per failure type.

### Phase 7: Generalist vs specialists

Question: *when does a generalist beat specialists plus a harness?*

- [ ] **Small pretrained VLA** (e.g. SmolVLA), fine-tuned on the task suite.
- [ ] **Compare** on the same tasks: end-to-end VLA vs specialist skills plus
      harness vs hybrid (VLA as a skill or fallback).
- [ ] **Sim-to-real**: decide whether real hardware is in scope.

## Backlog

Useful, but not on the harness's critical path. Pull an item in when a phase
needs it.

- Data efficiency: how few demos reach ≥95% with the v3 recipe.
- Frame question (world / gripper / both), 5 seeds each.
- Action chunking or diffusion policy for single skills.
- Physics and visual randomization (friction, mass, lighting, textures).
- Cube yaw randomization with a yaw-aligning expert.

## Decisions to make

- [ ] **Orchestrator** for Phase 4: API model (e.g. Claude) or a local
      small model.
- [x] **Simulator** for multi-object scenes (2026-10-12): extend the current
      MuJoCo setup (`multi_cube_scene.xml`: red, blue, green cubes). Revisit
      LIBERO (robosuite, MuJoCo, Franka) for Phase 4/7: it is where VLAs
      publish results (LIBERO-Long for multi-step, LIBERO-Plus for robot
      start states and camera shifts), so the harness vs VLA comparison
      could use the same tasks and scoring.
- [ ] **Compute** for Phase 7: whether cloud GPUs are available for VLA
      fine-tuning.

## Lessons so far

- **Validation loss didn't predict rollout success.** Only rollouts count.
- **Single seeds mislead.** The same config gave 4%–98% across seeds.
- **Binary actions need a classification loss.** MSE blurs the switch, and
  in-between commands push the policy into states it never trained on.
- **Hidden state in the expert makes imitation ambiguous.** When the expert
  acts on something the observation doesn't contain (here, a timer), the
  policy can't copy it. Either expose it or remove it. This is a memory
  problem in miniature, and it will come back at the harness level.
- **Any wait the policy can't see is a timer.** Replacing the 120-step hold
  with "lift once the fingers have stopped for 5 steps" still froze (89% of
  the identical-looking frames said stay). Triggering on the current width
  alone fixed it with no extra input. Check a trigger by asking whether two
  identical observations can get different labels.
- **A skill's success says nothing about the rest of the scene.** Push
  "succeeded" while shoving the other cube out of the way, and nothing in
  its signal shows it. Single-object skills need preconditions on free
  space, and the harness needs a verifier that checks every object, not
  just the target: side effects are a failure mode of their own.
- **Diagnose by stage.** Breaking rollouts into reach / close / grasp / rise
  / lift found both failures quickly. The harness needs the same kind of
  failure attribution.
- **Measure the error budget before building perception.** Degrading the
  true position first showed the policy needs ~2 mm, so a simple detector was
  enough and a learned one could wait.
- **Memory can't help at step 0.** Holding the last estimate covers occlusion
  mid-task, but the only real occlusion was at the start. A skill needs a
  perception precondition before it starts.
- **Normalise with a floor.** A feature that never varies in the demos (the
  cube's y offset in the fingers) turned a few mm at rollout into hundreds
  of standard deviations; place went 0% until `--obs-std-floor`.
- **Estimate rate and noise matter more for moving objects.** A 10 Hz
  estimate of a carried cube lags ~12 mm (place 0/30 vs 27/30 at 50 Hz), and
  a policy trained on exact positions failed with 1 mm noise. Fixes: track
  held objects from proprioception, train with observation noise.
- **Joint-space BC drifts along the arm's redundancy.** Fixed with a posture
  term in the expert plus DART noise.
- **Occlusion is skill-specific.** Pushing toward the robot hides the cube
  behind the hand; a contact motion model in the world model fixed it.
- **Handoffs fail on state the next skill never trained from**: pick after
  push (gripper closed, arm low) succeeds 20–35%. The cube was visible every
  time (all failures `grasp_missed`), so "not visible → recover" alone
  wouldn't have caught it; the skill has to declare its start state.
- **A learned skill can't tell it is outside its training starts.** Push
  from place's end pose didn't fail fast; it hovered for 14 s, nudged the
  cube up to 145 mm, and with the camera sometimes reported `object_dropped`
  or false success. Declaring start states in the contract and checking them
  in the harness turns that into an immediate, honest refusal.
- **The handoff lesson transfers to VLAs.** A LIBERO-90 SmolVLA does each
  step 60–100% from the robot's start pose and 0% right after the previous
  step (M2); returning the hand to the start pose between steps takes the
  chain from 0% to 56.1%, above the end-to-end VLA (43.9%). Same fix as the
  cube harness: skills have start states, the harness brings the robot to
  them.
- **A recovery pose plus declared start states fixes handoffs cheaply.**
  Going home with open fingers before retrying took pick after push to 100%
  without retraining anything. It can't fix the start blind spot, though:
  there the arm is already home.
- **A retry only helps if the failure leaves a state the skill can handle.**
  v2.0's dropped cubes landed outside the training range and retries couldn't
  recover them. This is the handoff bottleneck, seen already with one skill.

## Log

| Date | Version | What | Result |
|---|---|---|---|
| 2026-09-18 | — | Dataset sweep (300 / 1000 demos), world frame | ~50%, high seed variance |
| 2026-09-26 | v1.0 | 3000 demos, world frame | 58.7% (86 / 12 / 78) |
| 2026-09-26 | — | Observation frame comparison (world / gripper / both) | Within seed noise |
| 2026-09-26 | v1.1 | Snap gripper at rollout | 90.0% |
| 2026-09-26 | v2.0 | BCE gripper, `both` frame | 90.7%; the "grasp then freeze" failure appears |
| 2026-09-27 | v3.0 | Steps-since-closed input | 100% (3/3 seeds, standard eval) |
| 2026-09-28 | — | Project goal set: agentic robotics harness; plan rewritten | — |
| 2026-09-28 | — | Held-out benchmark (200 positions, stages, strict hold, 3 attempts) | Success@1: v1.0 58.7, v1.1 89.0, v2.0 86.8, v3.0 100% |
| 2026-09-28 | — | Perception moved to Phase 1 (camera + detector before the skill interface) | — |
| 2026-09-28 | — | Perception error budget (noisy cube position, 7 settings) | ≤2 mm random / ≤5 mm offset keeps ~100%; σ 10 mm → 64% |
| 2026-09-29 | v4.0 | Camera + color detector replace the simulator's cube position | Success@1 99.3%, Success@3 99.8%; failures = start-of-episode blind spot |
| 2026-09-29 | v4.1 | Skill abstraction + `pick` skill (preconditions, observable termination), evaluated in the unified benchmark | Success@1 95.0% (3/3 seeds); all 10 failures = `target_not_detected` refusals; success signal precision/recall 100% |
| 2026-10-08 | push v1.0 | 3 seeds, 200 held-out positions | gt 99.0 / 100 / 100% (mean 99.7%); camera 92.0 / 94.0 / 95.0% (mean 93.7%), 10 of each seed's failures = start blind-spot refusals; ~7 mm final error |
| 2026-10-07 | place/push/home v1.0 | Skill library grows to 4: scripted experts without hidden timers (place: joint-space tuck first; push: x leg then y leg, line following), BC on 29-d goal_obs with obs noise (≤4 mm) + DART joint noise (0.01 rad) + std floor; world model tracks held objects with the gripper and pushes hidden objects with the pusher | place 100% gt (3/3 seeds) and 100% camera (2 seeds run), 3 mm error; push s0 96% camera (50 pos); home 100%; pick after place 100%, after push 20% (handoff) |
| 2026-10-10 | v4.2 | Recovery pose (`recover` = home + open fingers) and the first orchestrator rule: on a `TARGET_NOT_DETECTED` / `START_STATE` refusal, recover and retry once (`benchmark.py --recover`); pick's new start-state precondition: fingers open | Pick after push (camera) 35% (20 pos) → 100 / 100 / 100% (600/600, 3 seeds, all via recovery); standard and after_place starts unchanged (20/20, 0 recoveries) |
| 2026-10-10 | v5.0 | Pick expert lifts once the fingers stop (5 still steps), no counter input (`--close-trigger settled`) | gt 95.0 / 100 / 98.0%; camera skill 85.0 / 94.0 / 69.5%; failures = grasp then freeze (timeouts), traced to 89% "stay" labels in identical settled-grip frames |
| 2026-10-10 | v5.1 | Pick expert lifts on the first step the fingers reach the cube's width (`--close-trigger contact`), no counter input; pick skill default | gt 100 / 100 / 100% (600/600); camera skill 95.0 / 95.0 / 95.0%, all failures start blind-spot refusals; expert 200/200, episodes 25% shorter. Handoffs (camera): after_place 100 / 100 / 94.5% (s2: 7 timeouts, 4 drops; v3.0 s2 after_place never run, so no baseline); after_push + recover 100 / 100 / 100% |
| 2026-10-12 | push v1.0 | Handoff: push `--start after_place`, 3 seeds, gt and camera; start-state interventions (60 scenes, gt) | gt 71.0 / 75.5 / 78.0%, camera 70.0 / 75.0 / 75.5% (vs 99.7% from home); direct 40/60, fingers closed first 40/60, wrist turned back 50/60, recover first 59/60 → the arm's start pose; −x pushes 12/29 |
| 2026-10-12 | push v1.1 | Start states in the skill contract: `StartState(arm, fingers)` declared per skill, checked by the Runner (`START_STATE`), not by the skill; pick's open-fingers check moved there; push declares arm at home + fingers open | Push after place + recover: gt 99.0 / 100 / 99.5%, camera 99.0 / 99.0 / 98.0% (was 74.8% / 73.5%); 20-position checks: push from home and pick after place unchanged, pick after push + recover 20/20 |
| 2026-10-12 | — | Multi-cube scene (red, blue, green), per-colour detectors, `--distractors N`; single-object skills with 1–2 distractors, 3 seeds, gt and camera (36 runs) | pick: gt 100 / 100%, camera 95 / 95% (d1 / d2), no distractor touched. place: gt 96.0 / 91.7%, camera 96.2 / 92.3%, distractor knocked 12 / 23%; < 40 mm from target 53.5% success. push: gt 99.5 / 99.0% but knocked 11 / 21% (81% when < 40 mm from target), camera 88.2 / 85.0% (`object_dropped` 37 / 57: cube tilts against the distractor + estimate error) |
| 2026-10-12 | — | Benchmark chosen: LIBERO-Long, then LIBERO-Plus. LeRobot + LIBERO installed; SmolVLA probes | `HuggingFaceVLA/smolvla_libero` 1/7 on tasks 0-1 (stale with current LeRobot); `lerobot/smolvla_libero` with CI settings 5/10; EGL broken → GLX; batch 10 runs out of memory → batch 5 |
| 2026-10-14 | — | SmolVLA baseline on LIBERO-Long (`lerobot/smolvla_libero`, 10 episodes per task, seeds 1000 / 2000 / 3000, EGL) | 53 / 61 / 48%, mean 54.0%; tasks moving two objects 23.3–56.7%, single-object tasks 76.7–86.7% |
| 2026-10-14 | — | SmolVLA LIBERO-Long failure breakdown: per-predicate logging rerun (same seeds; identical per-episode outcomes) | success 54.0%, partial 24.3%, none 20.3%, undone 1.3%; stove on 100% / pot on stove 77% (task 2); bowl in drawer 100% / drawer closed 77% (task 3); tasks 0 and 7 do nothing in 16 and 19 of 30 |
| 2026-10-14 | harness M0 | Harness rollout loop (`harness/`) plugged into lerobot-eval, all components off; tasks 0-1, seed 1000 | identical to the baseline episode for episode: 4/10 and 7/10, same success steps and predicate times |
| 2026-10-15 | harness M1 | Each plan step alone from the task's start, 520 steps, trained and split wording, 3 seeds (`run_m1.sh`) | commanded predicate 13.7% (trained) / 3.0% (split) vs 64.4% / 57.4% with the full instruction; other step done instead up to 53% (task 1) and 83% (task 8) |
| 2026-10-15 | harness M1b | `Kesvill/smolvla_libero_90` (SmolVLA trained on LIBERO-90) on LIBERO-Long: full instruction, and each step alone in LIBERO-90 wording; 3 seeds | full instruction 0% (30% partial); commanded step 68.7% (82.4% on 15 same-scene LIBERO-90 steps, 0% on 3 without), other-object 4.1%; product of step rates on the 6 covered tasks 69.0% vs 43.9% for smolvla_libero |
| 2026-10-15 | harness M2 | Steps in sequence, no recovery; LIBERO-90 SmolVLA, 6 covered tasks, batch 1, 3 seeds | 0/180; step 1 done 57–100%, step 2 never; step 1 undone in 23% |
| 2026-10-15 | harness M4 | + recovery pose between steps (hand back to start pose, gripper open, 28–39 steps) | 56.1% (56.7 / 56.7 / 55.0) vs 43.9% smolvla_libero end to end on the same tasks; best on tasks 3 (87%), 6 (60%), 4 (50%) |
| 2026-10-16 | harness M4 | Recovery triggered by a start-state check before every step (as in the cube harness) instead of after every finished step | 56.1% (56.7 / 56.7 / 55.0), same as M4; 180/180 first steps pass, 146 failed checks before step 2 all met after recovery |
| 2026-09-29 | — | Unified evaluation (skill mode, per-episode records, regression vs ground truth) + W&B tracking; 48 results rows backfilled | — |
