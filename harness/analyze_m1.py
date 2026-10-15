"""M1: does SmolVLA follow single-step instructions?

For every (task, step, wording) run of run_m1.sh, from the per-predicate logs:

  commanded   % of episodes where the step's own predicate held at some point
  baseline    % of baseline episodes (full task instruction) where that same
              predicate held: does the step instruction help or hurt it?
  other       % of episodes where a predicate the step did NOT ask for held
              (e.g. the soup went in the basket when told the tomato sauce):
              is it following the instruction or replaying the task?

  python harness/analyze_m1.py  [M1 dir]  [baseline logs dir]
"""

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from harness.plans import PLANS  # noqa: E402

M1 = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/experiments/libero_runs/m1")
BASE = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/experiments/libero_runs")


def episodes(paths):
    out = []
    for p in paths:
        out += [e for e in map(json.loads, open(p)) if not e["autoreset"]]
    return out


def held(e, pred):
    i = e["predicates"].index(pred)
    return e["first_true"][i] is not None


def pct(xs):
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def main():
    base = episodes(glob.glob(os.path.join(BASE, "subgoals_s*.jsonl")))
    rows = []
    for step in (0, 1):
        for wording in ("trained", "split"):
            eps = episodes(glob.glob(os.path.join(M1, f"step{step}_{wording}_s*.jsonl")))
            for t in sorted({e["task_id"] for e in eps}):
                plan = PLANS[t][step]
                pred = " ".join(plan.check)
                te = [e for e in eps if e["task_id"] == t]
                others = [p for p in te[0]["predicates"]
                          if p != pred and p in [" ".join(s.check) for s in PLANS[t]]]
                rows.append(dict(task=t, step=step, wording=wording, n=len(te),
                                 instruction=plan.instruction(wording),
                                 commanded=pct([held(e, pred) for e in te]),
                                 baseline=pct([held(e, pred) for e in base if e["task_id"] == t]),
                                 other=pct([any(held(e, o) for o in others) for e in te])))
    print(f"{'task':>4} {'step':>4} {'wording':8} {'n':>3} {'commanded':>9} {'baseline':>8} {'other':>6}  instruction")
    for r in rows:
        print(f"{r['task']:>4} {r['step'] + 1:>4} {r['wording']:8} {r['n']:>3} {r['commanded']:8.0f}% "
              f"{r['baseline']:7.0f}% {r['other']:5.0f}%  {r['instruction']}")
    for w in ("trained", "split"):
        rs = [r for r in rows if r["wording"] == w]
        if rs:
            print(f"\n{w}: mean commanded {sum(r['commanded'] for r in rs) / len(rs):.1f}% vs baseline "
                  f"{sum(r['baseline'] for r in rs) / len(rs):.1f}% on the same steps; "
                  f"other-object {sum(r['other'] for r in rs) / len(rs):.1f}%  ({len(rs)} steps)")


if __name__ == "__main__":
    main()
