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
| **Behaviour** | one action per control step (scripted or learned) |
| **Termination** | per step: running, success, or failure with a reason; observable signals only |
| **Effects** | the world change promised on success |
| **Result** | status, reason code, message, steps, world snapshot |
| **Internal state** | cleared on reset (e.g. pick's steps-closed counter) |

**Lifecycle** (`Runner.execute`):

```
validate args ─► check preconditions ─► reset ─► [perception → step → check]* ─► Result
     │                   │                                  │
 INVALID_ARGS    TARGET_NOT_DETECTED /             success / failure(reason) /
                 PRECONDITION_FAILED                timeout
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

**Failure codes** (shared, so an orchestrator/planner can react):
`INVALID_ARGS`, `PRECONDITION_FAILED`, `TARGET_NOT_DETECTED`, `UNREACHABLE`,
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
| Version | v4.0: the v3.0 BC policy driven by the world model's estimate |
| Parameters | `object`: object id, e.g. `cube_red` |
| Preconditions | object detected (`TARGET_NOT_DETECTED` otherwise); gripper empty (`PRECONDITION_FAILED`) |
| Success | gripper commanded closed, fingers stopped at 30–50 mm, estimated object centre ≥ 10 cm above rest, for 50 steps (1 s) |
| Failures | `GRASP_MISSED`: fingers closed below 25 mm for 10 steps before any lift; `OBJECT_DROPPED`: the same after a lift; `TIMEOUT`: 700 steps |
| Effects | `world.holding = object` |

Evaluate it:

```bash
cd scripts
python benchmark.py ../models/bc_clean3000_both_bce_time_s0.pt --label v4.1 \
    --skill pick --skill-arg object=cube_red
```

```python
from skills import LIBRARY, Runner
from skills.world import WorldModel

world = WorldModel(env)
result = Runner(env, world).execute(LIBRARY["pick"](), object="cube_red")
print(result)   # pick({'object': 'cube_red'}) -> success in 380 steps
```

## Adding a skill

1. Subclass `Skill` in `scripts/skills/<name>.py`: fill in identity, `params`,
   `preconditions`, `reset`, `step`, `check`, `effects`/`apply_effects`.
2. Termination from observable signals only; use the shared failure codes.
3. Register it in `LIBRARY` (`scripts/skills/__init__.py`).
4. Add its ground-truth stages/success to `benchmark.py` and evaluate it with
   `--skill <name>`.
