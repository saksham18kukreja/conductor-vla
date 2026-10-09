"""Held-out benchmark: 200 unseen cube positions, stage scores, retries.

Report versions with this; debug with eval_bc.py / diag_collapse.py on the
old 50-position set. These positions (rng seed 2026) are never used for
debugging.

Per position the policy gets up to --attempts tries of --max-steps each. The
sim and policy are deterministic, so a retry from the same start would replay
the same failure. A retry therefore starts from where the last attempt left
things: the arm is returned home with the gripper open and the cube stays
wherever it ended up (pushed, dropped, tilted).

Stages, each worth one point (furthest stage reached counts):

  reach  TCP within 15 mm of the cube centre in xy and 20 mm in z
  close  gripper commanded closed while at the grasp pose
  grasp  finger opening 30-50 mm while at the grasp pose
  rise   cube centre 10 mm above its resting height
  lift   cube centre 100 mm above its resting height (env.lifted())
  hold   lift sustained for 50 steps (1 s) with both fingers on the cube

Metrics:

  stage score  mean over positions of stages reached on attempt 1, out of 6
  success@1    hold reached on the first attempt
  success@K    hold reached within K attempts
  attempts     mean attempts used, over positions that succeeded
  lifted@1     old lenient metric (lift on attempt 1), for comparison

Perception (--perception) sets where the policy's cube position comes from:

  gt     the simulator's true position every step (the original setup)
  noisy  true position + Gaussian noise (--pos-noise, mm per axis) + a fixed
         offset (--pos-bias, mm, random direction per position), refreshed at
         --perception-hz and held in between. Used to find the error budget a
         detector must meet.
  color  table_cam RGB-D + perception.detectors.ColorDetector at
         --perception-hz. A rejected detection (cube too hidden) keeps the last
         good estimate; before the first good one, the workspace centre is used.

Skill mode (--skill NAME --skill-arg k=v) runs a library skill through the
skills Runner instead of the bare policy: one call per position (retries are
the orchestrator's job) and the skill's own perception (world model; with
--perception gt the world model is fed true positions, to separate perception
failures from skill failures). It also reports the skill's self-reported
success, the precision/recall of that signal against ground truth, and failure
reasons. pick uses the stages above; place, push and home have their own
stages, start states (--start, standard or a handoff from another skill's end
state) and targets, defined in skill_tasks.py. They also report the final
distance from the target (goal error).

Every logged run writes a per-position record to results/episodes/ and, unless
it is itself the ground-truth baseline, prints a regression report against the
latest ground-truth policy run for the same checkpoint: metric deltas and the
positions that regressed.

  python benchmark.py ../models/bc_clean3000_both_bce_time_s0.pt --label v3.0
  python benchmark.py <ckpt> --label v3.0 --perception noisy --pos-noise 5
  python benchmark.py <ckpt> --label v4.1 --skill pick --skill-arg object=cube_red
  python benchmark.py ../models/bc_place3000_s0.pt --label place-v1.0 --skill place
  python benchmark.py ../models/bc_place3000_s0.pt --label place-v1.0 --skill place --start pick
  python benchmark.py --label home-v1.0 --skill home --start after_push
"""

import argparse
import collections
import csv
import datetime
import json
import os
import subprocess
import sys

import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import FrankaCubeLift, TABLE_HEIGHT, CUBE_HALF  # noqa: E402
from collect_demos import GRIPPER_SCALE  # noqa: E402
from train_bc import MLPPolicy, policy_obs, GRIP_THRESHOLD  # noqa: E402

DEC = 10
HELDOUT_SEED = 2026
STAGES = ["reach", "close", "grasp", "rise", "lift", "hold"]
REST_Z = TABLE_HEIGHT + CUBE_HALF
HOLD_STEPS = 50
PERCEPTION_COLS = ["perception", "pos_noise_mm", "pos_bias_mm", "perception_hz",
                   "pos_err_med_mm", "pos_err_p95_mm", "detect_reject_pct"]
SETTLE_STEPS = 25  # control steps with the arm at home before a retry
CONTROL_HZ = 50
WORKSPACE_CENTRE = np.array([0.55, 0.0, TABLE_HEIGHT + CUBE_HALF])
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                       "benchmark.tsv")
EPISODES = os.path.join(os.path.dirname(RESULTS), "episodes")


def heldout_positions(n=200):
    rng = np.random.default_rng(HELDOUT_SEED)
    return [np.array([0.55, 0.0]) + rng.uniform([-0.08, -0.15], [0.08, 0.15])
            for _ in range(n)]


class Perception:
    """Where the policy's cube position comes from. None means ground truth."""

    def __init__(self, mode="gt", noise_mm=0.0, bias_mm=0.0, hz=10, min_visible=0.6):
        self.mode = mode
        self.min_visible = min_visible
        self.noise = noise_mm / 1000.0
        self.bias_mag = bias_mm / 1000.0
        self.every = max(1, round(CONTROL_HZ / hz))
        self.errors = []
        self.calls = self.rejects = 0
        self.detector = None
        self.last = {}

    def reset_position(self, idx):
        # Seeded by position, so every checkpoint sees the same noise.
        self.rng = np.random.default_rng(HELDOUT_SEED * 1000 + idx)
        d = self.rng.normal(size=3)
        self.bias = self.bias_mag * d / np.linalg.norm(d)
        self.estimate = None

    def reset_attempt(self):
        self.t = 0

    def _detect(self, env):
        if self.detector is None:
            from perception.camera import Camera
            from perception.detectors import ColorDetector
            self.detector = ColorDetector(Camera(env), min_visible=self.min_visible)
        self.calls += 1
        p, info = self.detector.detect()
        if p is None:
            self.rejects += 1
            fallback = self.estimate is None
            self.last = dict(info, status="fallback" if fallback else "rejected")
            return WORKSPACE_CENTRE if fallback else self.estimate
        self.last = dict(info, status="accepted")
        return p

    def cube_pos(self, env):
        if self.mode == "gt":
            return None
        if self.t % self.every == 0 or self.estimate is None:
            if self.mode == "color":
                self.estimate = self._detect(env)
            else:
                self.estimate = env.cube_pos + self.bias + self.rng.normal(0, self.noise, 3)
        self.t += 1
        self.errors.append(np.linalg.norm(self.estimate - env.cube_pos))
        return self.estimate


class Agent:
    """A saved BC checkpoint plus the per-attempt state its inputs need."""

    def __init__(self, path, device, raw_grip=False):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.policy = MLPPolicy(ck["obs_dim"], ck["act_dim"], hidden=ck["hidden"]).to(device)
        self.policy.load_state_dict(ck["state_dict"])
        self.policy.eval()
        self.norm = ck["norm"]
        self.frame = ck.get("frame", "world")
        self.grip_loss = ck.get("grip_loss", "mse")
        self.time_feature = ck.get("time_feature", False)
        # raw_grip reproduces the v1.0 rollout: the MSE gripper output sent as-is.
        self.raw_grip = raw_grip
        self.device = device
        self.n_closed = 0

    def reset(self):
        self.n_closed = 0

    def act(self, env, cube_pos=None):
        om, osd, am, asd = self.norm
        o = policy_obs(env, self.frame, self.time_feature, self.n_closed, cube_pos)
        with torch.no_grad():
            x = torch.as_tensor((o - om) / osd, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            a = self.policy(x).squeeze(0).cpu().numpy() * asd + am
        arm = np.clip(env.arm_qpos + a[:7], env.arm_limits[:, 0], env.arm_limits[:, 1])
        if self.raw_grip:
            g = float(np.clip(a[7], 0.0, 1.0))
        else:
            g = float(a[7] > GRIP_THRESHOLD[self.grip_loss])
        closed = g < 0.5
        self.n_closed = self.n_closed + 1 if closed else 0
        return arm, g * GRIPPER_SCALE, closed


def fingers_on_cube(env):
    m, d = env.model, env.data
    touching = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if env.cube_id in (b1, b2):
            other = b2 if b1 == env.cube_id else b1
            touching.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other))
    return {"left_finger", "right_finger"} <= touching


def return_home(env):
    """Arm and fingers back to the start pose; the cube keeps its current state."""
    adr = env.cube_qpos_adr
    dadr = env.cube_dof_adr
    cube_q = env.data.qpos[adr : adr + 7].copy()
    cube_v = env.data.qvel[dadr : dadr + 6].copy()
    env.reset()
    env.data.qpos[adr : adr + 7] = cube_q
    env.data.qvel[dadr : dadr + 6] = cube_v
    mujoco.mj_forward(env.model, env.data)
    home = env.arm_qpos
    for _ in range(SETTLE_STEPS):  # let a released cube fall and settle
        env.step(home, GRIPPER_SCALE, n_substeps=DEC)


class StageTracker:
    """Ground-truth stage tracking, shared by policy and skill modes."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.best, self.held = -1, 0

    def update(self, env, closed):
        rel = env.cube_pos - env.tcp_pos
        at_pose = np.linalg.norm(rel[:2]) < 0.015 and abs(rel[2]) < 0.020
        lifted = env.lifted()
        self.held = self.held + 1 if lifted and fingers_on_cube(env) else 0
        hits = [
            at_pose,
            at_pose and closed,
            at_pose and 0.030 < env.gripper_width < 0.050,
            env.cube_pos[2] > REST_Z + 0.010,
            lifted,
            self.held >= HOLD_STEPS,
        ]
        self.best = max([self.best] + [i for i, h in enumerate(hits) if h])
        return self.best

    @property
    def done(self):
        return self.best == len(STAGES) - 1


def attempt(env, agent, perception, max_steps, on_step=None):
    """One try from the current state. Returns the index of the furthest stage.

    on_step(best, closed, n_closed) is called after every control step (used
    by make_bc_video.py).
    """
    agent.reset()
    perception.reset_attempt()
    tracker = StageTracker()
    for _ in range(max_steps):
        arm, grip, closed = agent.act(env, perception.cube_pos(env))
        env.step(arm, grip, n_substeps=DEC)
        best = tracker.update(env, closed)
        if on_step is not None:
            on_step(best, closed, agent.n_closed)
        if tracker.done:
            break
    return tracker.best


def evaluate(env, agent, perception, positions, attempts, max_steps):
    """Policy mode. One record per position."""
    rows = []
    for idx, xy in enumerate(positions):
        env.reset(cube_xy=xy)
        perception.reset_position(idx)
        first, used, success = None, attempts, False
        for k in range(attempts):
            if k > 0:
                if env.cube_pos[2] < TABLE_HEIGHT - 0.05:  # fell off the table
                    break
                return_home(env)
            best = attempt(env, agent, perception, max_steps)
            if first is None:
                first = best
            if best == len(STAGES) - 1:
                used, success = k + 1, True
                break
        rows.append(dict(idx=idx, first=first, success=success, attempts=used,
                         reported="", reason=""))
    return rows


def evaluate_skill(env, skill, skill_name, start, skill_args, positions, max_steps,
                   min_visible, oracle):
    """Skill mode: the skill runs through the Runner, one call per position
    (retries are the orchestrator's job). pick is scored with the same
    StageTracker as policy mode; place/push/home with skill_tasks trackers.
    Episodes whose start state could not be produced are recorded as skipped
    and left out of the metrics."""
    import skill_tasks
    from skills import LIBRARY, Runner
    from skills.world import WorldModel

    world = WorldModel(env, min_visible=min_visible, oracle=oracle)
    tracker = None
    errors = []

    def on_step(_skill, _t, _check):
        if skill_name == "pick":
            tracker.update(env, world.gripper_cmd_closed)
        else:
            tracker.update(world)
        for est in world.objects.values():
            if est.detected:
                errors.append(np.linalg.norm(est.position - env.cube_pos))

    runner = Runner(env, world, timeout_steps=max_steps, on_step=on_step)
    pick_runner = None
    if start == "pick":  # handoff: the learned pick skill produces the start
        pick_runner = Runner(env, world, timeout_steps=max_steps)
        pick_runner.pick_skill = LIBRARY["pick"]()
    rows = []
    calls = rejects = 0
    for idx, xy in enumerate(positions):
        prepared = skill_tasks.setup(skill_name, start, env, world, xy, idx, pick_runner)
        if prepared is None:
            rows.append(dict(idx=idx, first=-1, success=False, attempts=1, reported="",
                             reason="start_failed", goal_err=float("nan"), skipped=True))
            continue
        args, goal = prepared
        args.update(skill_args)
        if skill_name == "pick":
            tracker = StageTracker()
        else:
            tracker = skill_tasks.TRACKERS[skill_name](env, goal)
        n0, r0 = world.detect_calls, world.detect_rejects
        res = runner.execute(skill, **args)
        calls += world.detect_calls - n0  # world.reset() per episode clears the counters
        rejects += world.detect_rejects - r0
        if skill_name == "pick":
            success, goal_err = tracker.done, float("nan")
        else:
            success, goal_err = tracker.finish(), tracker.goal_err()
        rows.append(dict(idx=idx, first=tracker.best, success=success, attempts=1,
                         reported=res.ok, reason=res.reason.value, goal_err=goal_err,
                         skipped=False))
    info = {"errors": errors, "calls": calls, "rejects": rejects}
    return rows, info


def perception_tag(args):
    if args.skill:
        # skills see the world model: table_cam + color detector, or the
        # simulator's true positions with --perception gt
        tag = "gt" if args.perception == "gt" else "color"
    else:
        tag = args.perception
    if tag == "color" and args.min_visible != 0.6:
        return f"color_minvis{args.min_visible:g}"
    return tag


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(RESULTS), text=True).strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------ results files

COLUMNS = (["date", "commit", "label", "checkpoint", "frame", "grip_loss", "raw_grip",
            "time_feature", "positions", "attempts", "stage_score", "success_1",
            "success_k", "mean_attempts", "lifted_1"]
           + [f"rate_{s}" for s in STAGES] + PERCEPTION_COLS
           + ["mode", "reported_success", "signal_precision", "signal_recall",
              "failure_reasons", "episodes", "eval_set", "protocol",
              "start", "stage_rates", "goal_err_med_mm", "goal_err_p95_mm", "start_failed"])


def summarize(rows, attempts, stages=STAGES):
    skipped = sum(bool(r.get("skipped")) for r in rows)
    rows = [r for r in rows if not r.get("skipped")]
    first = np.array([r["first"] for r in rows])
    success = np.array([r["success"] for r in rows])
    used = np.array([r["attempts"] for r in rows])
    m = {
        "stage_score": 100 * np.mean((first + 1) / len(stages)),
        "success_1": 100 * np.mean(success & (used == 1)),
        "success_k": 100 * np.mean(success),
        "mean_attempts": float(np.mean(used[success])) if success.any() else float("nan"),
        "lifted_1": 100 * np.mean(first >= STAGES.index("lift")) if stages is STAGES
        else float("nan"),
        "start_failed": skipped,
    }
    m["stage_rates"] = {s: 100 * np.mean(first >= i) for i, s in enumerate(stages)}
    if stages is STAGES:
        for s in STAGES:
            m[f"rate_{s}"] = m["stage_rates"][s]
    errs = np.array([r.get("goal_err", np.nan) for r in rows], dtype=float)
    errs = errs[~np.isnan(errs)] * 1000
    m["goal_err_med_mm"] = float(np.median(errs)) if len(errs) else float("nan")
    m["goal_err_p95_mm"] = float(np.percentile(errs, 95)) if len(errs) else float("nan")
    m["att_hist"] = [int(np.sum(success & (used == k))) for k in range(1, attempts + 1)]
    m["failed"] = int((~success).sum())
    if rows and rows[0]["reported"] != "":
        claimed = np.array([bool(r["reported"]) for r in rows])
        tp = int((claimed & success).sum())
        m["reported_success"] = 100 * claimed.mean()
        m["signal_precision"] = 100 * tp / max(1, claimed.sum())
        m["signal_recall"] = 100 * tp / max(1, success.sum())
        reasons = collections.Counter(r["reason"] for r in rows if not r["reported"])
        m["failure_reasons"] = json.dumps(dict(reasons))
    return m


def migrate_results():
    """Add columns introduced since a results file was started (blank for old rows)."""
    if not os.path.exists(RESULTS):
        return
    with open(RESULTS) as f:
        header = f.readline().rstrip("\n").split("\t")
        rows = list(csv.DictReader(f, fieldnames=header, delimiter="\t"))
    if header == COLUMNS:
        return
    with open(RESULTS, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c) or "") for c in COLUMNS) + "\n")


def write_episodes(run_id, rows, stages=STAGES):
    os.makedirs(EPISODES, exist_ok=True)
    path = os.path.join(EPISODES, f"{run_id}.tsv")
    goal = any("goal_err" in r and not np.isnan(r["goal_err"]) for r in rows)
    with open(path, "w") as f:
        f.write("idx\tfirst_stage\tsuccess\tattempts\treported\treason"
                + ("\tgoal_err_mm" if goal else "") + "\n")
        for r in rows:
            stage = stages[r["first"]] if r["first"] >= 0 else "none"
            f.write(f"{r['idx']}\t{stage}\t{int(r['success'])}\t{r['attempts']}\t"
                    f"{r['reported']}\t{r['reason']}")
            if goal:
                f.write(f"\t{1000 * r['goal_err']:.1f}")
            f.write("\n")
    return os.path.basename(path)


def read_episodes(name):
    path = os.path.join(EPISODES, name)
    if not name or not os.path.exists(path):
        return None
    with open(path) as f:
        return {int(r["idx"]): r for r in csv.DictReader(f, delimiter="\t")}


def find_baseline(checkpoint, mode="policy", start=""):
    """Latest ground-truth row for this checkpoint: policy mode for the pick
    policy and pick skill (whose standard start is the policy benchmark's);
    otherwise the same skill and start with --perception gt."""
    if not os.path.exists(RESULTS):
        return None
    with open(RESULTS) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if mode in ("policy", "skill:pick") and start in ("", "home"):
        rows = [r for r in rows if r["checkpoint"] == checkpoint and r["perception"] == "gt"
                and r.get("mode", "policy") == "policy" and r["raw_grip"] == "False"]
    else:
        rows = [r for r in rows if r["checkpoint"] == checkpoint and r["perception"] == "gt"
                and r.get("mode") == mode and r.get("start", "") == start]
    return rows[-1] if rows else None


def regression_report(m, rows, checkpoint, is_baseline, mode="policy", start="",
                      stages=STAGES):
    """Compare a run with the ground-truth baseline for the same checkpoint."""
    if is_baseline:
        return None
    base = find_baseline(checkpoint, mode, start)
    if base is None:
        print("   regression vs ground truth: no baseline row for this checkpoint "
              "(run it with --perception gt)")
        return None
    keys = ["stage_score", "success_1"]
    if "rate_hold" in m:
        keys += ["lifted_1"] + [f"rate_{s}" for s in STAGES]
    print(f"   regression vs ground truth ({base['label']}, perception gt, {base['date']}):")
    print("     " + "  ".join(
        f"{k.replace('rate_', '')} {m[k] - float(base[k]):+.1f}" for k in keys))
    report = {"baseline": f"{base['label']} {base['checkpoint']} gt ({base['date']})",
              "deltas": {k: m[k] - float(base[k]) for k in keys}}
    eps = read_episodes(base.get("episodes", ""))
    if eps is None:
        print("     (no per-episode record for the baseline; re-run it to list positions)")
        return report
    now = {r["idx"]: r for r in rows if not r.get("skipped")}
    first_ok = lambda r: bool(int(r["success"])) and int(r["attempts"]) == 1
    worse = [i for i in now if i in eps and first_ok(eps[i])
             and not (now[i]["success"] and now[i]["attempts"] == 1)]
    better = [i for i in now if i in eps and not first_ok(eps[i])
              and now[i]["success"] and now[i]["attempts"] == 1]
    print(f"     first-attempt regressions: {len(worse)}   improvements: {len(better)}")
    for i in worse:
        r = now[i]
        stage = stages[r["first"]] if r["first"] >= 0 else "none"
        why = f", {r['reason']}" if r["reason"] else ""
        print(f"       pos {i:3d}: reached {stage}{why}")
    report.update(regressed=worse, improved=better)
    return report


def log_eval_to_wandb(args, path, name, mode, ptag, attempts, eval_set, agent, m, rows,
                      report, err_med, err_p95, reject, skill_args, stages=STAGES,
                      protocol=None, start=""):
    """Mirror one evaluation to W&B: config, metrics, funnel, episodes, regressions."""
    import tracking
    import wandb

    protocol = protocol or tracking.PROTOCOL
    ck = torch.load(path, map_location="cpu", weights_only=False) if agent else {}
    label = args.label or "nolabel"
    stem = name.replace(".pt", "")
    config = {
        "label": label, "checkpoint": name, "mode": mode, "skill_args": skill_args,
        "perception": ptag, "pos_noise_mm": args.pos_noise, "pos_bias_mm": args.pos_bias,
        "perception_hz": args.perception_hz, "min_visible": args.min_visible,
        "attempts": attempts, "max_steps": args.max_steps, "raw_grip": args.raw_grip,
        "frame": getattr(agent, "frame", ""), "grip_loss": getattr(agent, "grip_loss", ""),
        "time_feature": getattr(agent, "time_feature", ""), "start": start,
        "eval_set": eval_set, "protocol": protocol, "commit": tracking.git_commit(),
        "train_data": tracking.data_reference(path, ck) if agent else "scripted",
    }
    run_name = f"eval-{label}-{stem}-{mode.replace(':', '-')}-{ptag}" + (f"-{start}" if start else "")
    run = tracking.init("eval", run_name, config, group=label,
                        tags=[t for t in [label, mode, ptag, eval_set, protocol, start] if t])
    if agent:
        tracking.use_checkpoint(run, path, {"train_data": config["train_data"]})

    summary = {k: m[k] for k in ["stage_score", "success_1", "success_k", "mean_attempts",
                                 "lifted_1", "goal_err_med_mm", "goal_err_p95_mm",
                                 "start_failed"]}
    summary.update({f"rate_{s}": v for s, v in m["stage_rates"].items()})
    summary.update(pos_err_med_mm=err_med, pos_err_p95_mm=err_p95, detect_reject_pct=reject)
    for k in ["reported_success", "signal_precision", "signal_recall"]:
        if k in m:
            summary[k] = m[k]
    if report:
        summary.update({f"vs_gt/{k}": v for k, v in report["deltas"].items()})
        summary["vs_gt/baseline"] = report["baseline"]
        if "regressed" in report:
            summary["vs_gt/regressed_positions"] = report["regressed"]
            summary["vs_gt/improved_positions"] = report["improved"]
            summary["vs_gt/n_regressed"] = len(report["regressed"])
    run.summary.update(summary)

    funnel = wandb.Table(columns=["stage", "reached_pct"],
                         data=[[s, v] for s, v in m["stage_rates"].items()])
    episodes = wandb.Table(columns=["position", "first_stage", "success", "attempts",
                                    "skill_reported_success", "failure_reason", "goal_err_mm"],
                           data=[[r["idx"], stages[r["first"]] if r["first"] >= 0 else "none",
                                  bool(r["success"]), r["attempts"], str(r["reported"]),
                                  r["reason"], 1000 * r.get("goal_err", float("nan"))]
                                 for r in rows])
    run.log({"stage_funnel": wandb.plot.bar(funnel, "stage", "reached_pct",
                                            title="Stage funnel (attempt 1)"),
             "episodes": episodes})
    run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="*",
                    help="policy checkpoints (none for a scripted skill, e.g. home)")
    ap.add_argument("--label", type=str, default="",
                    help="version label for the results row, e.g. v3.0")
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--raw-grip", action="store_true",
                    help="send the MSE gripper output unsnapped (v1.0 rollout)")
    ap.add_argument("--perception", choices=["gt", "noisy", "color"], default=None,
                    help="source of the policy's cube position (see module doc); "
                         "default gt for the policy, color for a skill")
    ap.add_argument("--pos-noise", type=float, default=0.0,
                    help="noisy: Gaussian noise per axis, mm")
    ap.add_argument("--pos-bias", type=float, default=0.0,
                    help="noisy: fixed offset magnitude, mm")
    ap.add_argument("--perception-hz", type=float, default=10.0,
                    help="noisy: estimate refresh rate (control runs at 50 Hz)")
    ap.add_argument("--min-visible", type=float, default=0.6,
                    help="color / skill: reject detections below this visible "
                         "fraction (0 accepts every detection)")
    ap.add_argument("--skill", type=str, default=None,
                    help="skill mode: run this library skill (with the checkpoint) "
                         "through the Runner instead of the bare policy")
    ap.add_argument("--skill-arg", action="append", default=[],
                    help="skill mode: name=value, repeatable (e.g. object=cube_red)")
    ap.add_argument("--start", type=str, default=None,
                    help="skill mode: start state (skill_tasks.STARTS; default: the "
                         "skill's standard start, others are handoffs)")
    ap.add_argument("--no-log", action="store_true", help="don't append to results")
    ap.add_argument("--wandb", action="store_true",
                    help="also log each evaluation to W&B (tracking.py)")
    args = ap.parse_args()
    if args.perception is None:
        args.perception = "color" if args.skill else "gt"
    import tracking
    eval_set = tracking.eval_set_id(HELDOUT_SEED, args.positions)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    positions = heldout_positions(args.positions)
    env = FrankaCubeLift()
    attempts = 1 if args.skill else args.attempts
    mode = f"skill:{args.skill}" if args.skill else "policy"
    ptag = perception_tag(args)
    stages, start, protocol = STAGES, "", tracking.PROTOCOL
    if args.skill:
        import skill_tasks
        start = args.start or skill_tasks.STARTS[args.skill][0]
        if start not in skill_tasks.STARTS[args.skill]:
            ap.error(f"--start for {args.skill}: one of {skill_tasks.STARTS[args.skill]}")
        if args.skill != "pick":
            stages = skill_tasks.TRACKERS[args.skill].stages
            protocol = skill_tasks.PROTOCOL[args.skill]
    checkpoints = args.checkpoints or (["scripted"] if args.skill == "home" else [])
    if not checkpoints:
        ap.error("give at least one checkpoint")

    for path in checkpoints:
        name = os.path.basename(path)
        if args.skill:
            from skills import LIBRARY
            skill = LIBRARY[args.skill](checkpoint=None if path == "scripted" else path)
            skill_args = dict(a.split("=", 1) for a in args.skill_arg)
            rows, info = evaluate_skill(env, skill, args.skill, start, skill_args, positions,
                                        args.max_steps, args.min_visible, oracle=ptag == "gt")
            agent = skill.agent
            errs_raw, calls, rejects = info["errors"], info["calls"], info["rejects"]
        else:
            agent = Agent(path, device, raw_grip=args.raw_grip)
            perception = Perception(args.perception, args.pos_noise, args.pos_bias,
                                    args.perception_hz, args.min_visible)
            rows = evaluate(env, agent, perception, positions, args.attempts, args.max_steps)
            errs_raw, calls, rejects = perception.errors, perception.calls, perception.rejects
        errs = np.array(errs_raw) * 1000 if len(errs_raw) else np.zeros(1)
        err_med, err_p95 = np.median(errs), np.percentile(errs, 95)
        reject = 100 * rejects / max(1, calls)
        m = summarize(rows, attempts, stages)

        frame = getattr(agent, "frame", "scripted")
        print(f"\n== {name}  {args.label}  mode={mode}{'  start=' + start if start else ''}  "
              f"(frame={frame}, grip={getattr(agent, 'grip_loss', '-')}"
              f"{', raw' if args.raw_grip else ''}, time={getattr(agent, 'time_feature', '-')}, "
              f"perception={ptag})")
        retry = (f"success@{attempts} {m['success_k']:5.1f}%   attempts {m['mean_attempts']:.2f}   "
                 if attempts > 1 else "")
        lifted = f"lifted@1 {m['lifted_1']:5.1f}%" if stages is STAGES else ""
        print(f"   stage score {m['stage_score']:5.1f}   success@1 {m['success_1']:5.1f}%   "
              f"{retry}{lifted}")
        print("   attempt 1 stage rates: " + "  ".join(
            f"{s} {v:.0f}%" for s, v in m["stage_rates"].items()))
        print(f"   succeeded on attempt 1/2/…: {m['att_hist']}   failed: {m['failed']}"
              + (f"   start state failed (not scored): {m['start_failed']}"
                 if m["start_failed"] else ""))
        if not np.isnan(m["goal_err_med_mm"]):
            print(f"   final distance to target: median {m['goal_err_med_mm']:.1f} mm  "
                  f"p95 {m['goal_err_p95_mm']:.1f} mm")
        if ptag != "gt":
            print(f"   cube position error: median {err_med:.1f} mm  p95 {err_p95:.1f} mm"
                  + (f"   detections rejected {reject:.1f}%" if ptag.startswith("color") else ""))
        if args.skill:
            print(f"   skill-reported success {m['reported_success']:.1f}%   success signal: "
                  f"precision {m['signal_precision']:.1f}%  recall {m['signal_recall']:.1f}%   "
                  f"failure reasons {m['failure_reasons']}")
        is_baseline = ptag == "gt" and not args.raw_grip and (
            mode == "policy" or (args.skill and args.skill != "pick"))
        report = regression_report(m, rows, name, is_baseline, mode, start, stages)
        if args.wandb:
            log_eval_to_wandb(args, path, name, mode, ptag, attempts, eval_set, agent, m, rows,
                              report, err_med, err_p95, reject,
                              dict(a.split("=", 1) for a in args.skill_arg), stages,
                              protocol, start)

        if not args.no_log:
            stem = name.replace(".pt", "")
            standard = not args.skill or start == skill_tasks.STARTS[args.skill][0]
            run_id = "__".join(x for x in [args.label or "nolabel", stem,
                                           mode.replace(":", "-"), ptag,
                                           "" if standard else start] if x)
            if ptag == "noisy":
                run_id += f"_n{args.pos_noise:g}_b{args.pos_bias:g}"
            episodes = write_episodes(run_id, rows, stages)
            values = {
                "date": datetime.date.today().isoformat(), "commit": git_commit(),
                "label": args.label, "checkpoint": name, "frame": frame,
                "grip_loss": getattr(agent, "grip_loss", ""), "raw_grip": args.raw_grip,
                "time_feature": getattr(agent, "time_feature", ""), "positions": args.positions,
                "attempts": attempts, "perception": ptag,
                "pos_noise_mm": f"{args.pos_noise:g}", "pos_bias_mm": f"{args.pos_bias:g}",
                "perception_hz": f"{args.perception_hz:g}",
                "pos_err_med_mm": f"{err_med:.2f}", "pos_err_p95_mm": f"{err_p95:.2f}",
                "detect_reject_pct": f"{reject:.1f}", "mode": mode, "episodes": episodes,
                "eval_set": eval_set, "protocol": protocol, "start": start,
                "stage_rates": json.dumps({k: round(v, 1) for k, v in m["stage_rates"].items()}),
                "goal_err_med_mm": "" if np.isnan(m["goal_err_med_mm"])
                else f"{m['goal_err_med_mm']:.1f}",
                "goal_err_p95_mm": "" if np.isnan(m["goal_err_p95_mm"])
                else f"{m['goal_err_p95_mm']:.1f}",
                "start_failed": m["start_failed"],
            }
            for k in ["stage_score", "success_1", "success_k", "lifted_1"] + \
                     [f"rate_{s}" for s in STAGES]:
                values[k] = f"{m[k]:.1f}" if k in m and not np.isnan(m[k]) else ""
            values["mean_attempts"] = f"{m['mean_attempts']:.2f}"
            for k in ["reported_success", "signal_precision", "signal_recall"]:
                values[k] = f"{m[k]:.1f}" if k in m else ""
            values["failure_reasons"] = m.get("failure_reasons", "")
            migrate_results()
            new = not os.path.exists(RESULTS)
            with open(RESULTS, "a") as f:
                if new:
                    f.write("\t".join(COLUMNS) + "\n")
                f.write("\t".join(str(values[c]) for c in COLUMNS) + "\n")
    env.close()


if __name__ == "__main__":
    main()
