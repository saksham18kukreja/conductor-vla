# Skills

The skill library is the harness's toolbox: behaviours with a contract that an
orchestrator (and later a language planner) can call. Code: `scripts/skills/`.

## The abstraction

A skill is a self-contained behaviour with a contract. Outside code sees only
the contract, so a scripted controller and a learned policy look the same.

| Part | Meaning |
|---|---|
| **Identity** | name, one-line description, version |
| **Parameters** | typed inputs, validated before starting (objects by id) |
| **Preconditions** | checks on the world model before starting → ok, or a reason |
| **Start states** | the robot states it was trained or designed to start from (arm pose, fingers), declared as data; the Runner checks them |
| **Behaviour** | one action per control step (scripted or learned) |
| **Termination** | per step: running, success, or failure with a reason; observable signals only |
| **Effects** | the world change promised on success |
| **Result** | status, reason code, message, steps, world snapshot |
| **Internal state** | cleared on reset (e.g. a learned skill's policy state) |

**Lifecycle** (`Runner.execute`):

```
validate args ─► preconditions ─► start state ─► reset ─► [perception → step → check]* ─► Result
     │                 │                 │                              │
 INVALID_ARGS   TARGET_NOT_DETECTED /  START_STATE           success / failure(reason) /
                PRECONDITION_FAILED                           timeout
```

**Design decisions:**

1. **The Runner owns the control loop**; skills expose `step()`. The Runner
   updates perception, enforces the timeout, and can record or stop any skill
   the same way.
2. **Termination:** the skill decides success and its own failures; the
   Runner adds the timeout (and later, generic safety checks).
3. **Skills see the world model + the robot's own state.** Never the
   simulator's object state. Perception stays a separate, measurable part.
4. **Targets by reference:** objects by id from the world model (e.g.
   `cube_red`); places by named zone or (x, y) once a skill needs them.
5. **Interruptible only by the Runner** (timeout, safety); an orchestrator
   waits for the Result.
6. **Preconditions report ok/fail + reason**, not a fix. Choosing the fix
   (e.g. run a `look` skill) is the orchestrator's job.
7. **Retries belong to the orchestrator**, not to skills or the executor.
8. **Start states are part of the contract, checked by the harness.** A
   learned skill is only trustworthy from the states it trained from, and
   it can't tell when it is outside them: pick from closed fingers (35%)
   and push from where place stops (75%) ran on, missed and timed out
   instead of failing fast. So each skill declares
   `start_state = StartState(arm=..., fingers=...)` and the Runner refuses
   with `START_STATE` before the first step. The skill's own code never
   checks it; preconditions stay about the task (object visible, holding).

   | Skill | Start state | Why |
   |---|---|---|
   | pick | fingers open, any arm pose | from closed fingers 13/20 grasps missed; after place 100% |
   | place | any (holding is a precondition) | trained from pick's end states |
   | push | arm at home, fingers open | trained from home only; after place 40/60, pushes toward −x 12/29 |
   | home, recover | any | scripted |

**Failure codes** (shared, so an orchestrator/planner can react):
`INVALID_ARGS`, `PRECONDITION_FAILED`, `START_STATE` (the robot isn't in one of
the skill's declared start states), `TARGET_NOT_DETECTED`, `UNREACHABLE`,
`GRASP_MISSED`, `OBJECT_DROPPED`, `TIMEOUT`.

**Two different success checks.** A skill's termination uses only what the
robot can observe. Grading uses simulator ground truth, only in
`benchmark.py`. Comparing them gives the precision and recall of each skill's
own success signal: how far the harness can trust a skill's "done".

**Evaluation.** Skills are evaluated with the same framework as every other
experiment: `benchmark.py --skill NAME` runs the skill through the Runner on
the 200 held-out positions, one call each, with the same ground-truth stages
and success definition as the policy benchmark. It adds the skill's reported
success, the precision/recall of its success signal and its failure reasons,
and a regression report against the ground-truth baseline.

**World model** (`skills/world.py`): object estimates by id (position,
visible fraction, last seen) from `table_cam` + detectors at 10 Hz, keeping the
last good estimate when a detection is rejected; gripper command and width;
what is held. `snapshot()` gives a plain-data view for Results and, later, a
planner prompt.

## Library

### `pick(object)`

Grasp an object on the table and lift it 10 cm.

| | |
|---|---|
| Version | v5.1: BC policy from an expert with no hidden timer (lifts once the fingers reach the cube's width), driven by the world model's estimate; no steps-since-closed input |
| Parameters | `object`: object id, e.g. `cube_red` |
| Preconditions | object detected (`TARGET_NOT_DETECTED` otherwise); gripper empty (`PRECONDITION_FAILED`) |
| Start state | fingers open ≥ 70 mm, any arm pose |
| Success | gripper commanded closed, fingers stopped at 30–50 mm, estimated object centre ≥ 10 cm above rest, for 50 steps (1 s) |
| Failures | `GRASP_MISSED`: fingers closed below 25 mm for 10 steps before any lift; `OBJECT_DROPPED`: the same after a lift; `TIMEOUT`: 700 steps |
| Effects | `world.holding = object` |

Evaluate it:

```bash
cd scripts
python benchmark.py ../models/bc_contact3000_both_bce_s0.pt --label v5.1 \
    --skill pick --skill-arg object=cube_red
```

```python
from skills import LIBRARY, Runner
from skills.world import WorldModel

world = WorldModel(env)
result = Runner(env, world).execute(LIBRARY["pick"](), object="cube_red")
print(result)   # pick({'object': 'cube_red'}) -> success in 380 steps
```

## Recovery pose

`recover()` (`skills/home.py`) is home with the fingers opened (kept closed
if holding). Home is every skill's standard start and the camera's clearest
view of the table, so it is the place to go back to when a skill can't start.

`skills/recovery.py` holds the first orchestrator rule: if a skill refuses
with `TARGET_NOT_DETECTED` or `START_STATE`, run `recover` and try the skill
once more. It stops if the arm is already at the recovery pose and the
target is still hidden (going there again changes nothing; this is the case
for the start-of-episode blind spot, which needs a different viewing pose).
Failures during a run (grasp missed, dropped) are not recovered by this rule.

```python
from skills.recovery import execute_with_recovery
result, log = execute_with_recovery(runner, LIBRARY["pick"](), LIBRARY["recover"](),
                                    object="cube_red")
```

Benchmark it with `benchmark.py --skill NAME --recover` (mode
`skill:NAME+recover`).

## Adding a skill

1. Subclass `Skill` in `scripts/skills/<name>.py`: fill in identity, `params`,
   `start_state` (the states its demos or design start from), `preconditions`,
   `reset`, `step`, `check`, `effects`/`apply_effects`.
2. Termination from observable signals only; use the shared failure codes.
3. Register it in `LIBRARY` (`scripts/skills/__init__.py`).
4. Add its ground-truth stages/success to `benchmark.py` and evaluate it with
   `--skill <name>`.
