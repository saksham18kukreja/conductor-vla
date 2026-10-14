"""Per-predicate breakdown of LIBERO-Long episodes (eval_with_subgoals.py logs).

Evaluated episodes are those not marked "autoreset" (the vector env starts
those itself after an episode ends; they are cut short when the batch ends).
Each evaluated episode falls in one outcome:

  success     every goal predicate held at once
  partial     some predicates held at the end (which ones are listed)
  undone      a predicate held at some point but not at the end
  none        no goal predicate ever held

  python analyze_subgoals.py ~/experiments/libero_runs/subgoals_s*.jsonl
"""

import collections
import json
import sys


def real_episodes(path):
    eps = [json.loads(line) for line in open(path)]
    if "autoreset" not in eps[0]:
        raise SystemExit(f"{path}: logged before autoreset marking; rerun eval_with_subgoals.py")
    return [e for e in eps if not e["autoreset"]]


def outcome(e):
    if e["success"]:
        return "success"
    if any(e["final"]):
        return "partial"
    if any(t is not None for t in e["first_true"]):
        return "undone"
    return "none"


def short(p):
    # "in alphabet_soup_1 basket_1_contain_region" -> "in alphabet_soup basket"
    return " ".join(w.replace("_1_contain_region", "").replace("_contain_region", "")
                    .replace("_region", "").rstrip("_1").removesuffix("_1") for w in p.split())


def main(paths):
    by_task = collections.defaultdict(list)
    for p in paths:
        eps = real_episodes(p)
        n_tasks = len({e["task_id"] for e in eps})
        print(f"{p}: {len(eps)} real episodes over {n_tasks} tasks, "
              f"success {100 * sum(e['success'] for e in eps) / max(1, len(eps)):.1f}%")
        for e in eps:
            by_task[e["task_id"]].append(e)

    total = collections.Counter()
    print()
    for t in sorted(by_task):
        eps = by_task[t]
        preds = [short(p) for p in eps[0]["predicates"]]
        c = collections.Counter(outcome(e) for e in eps)
        total.update(c)
        held = collections.Counter()
        for e in eps:
            if outcome(e) == "partial":
                held[" + ".join(preds[i] for i, f in enumerate(e["final"]) if f)] += 1
        first = collections.Counter()  # which predicate was achieved first, when both were
        for e in eps:
            ft = e["first_true"]
            if all(x is not None for x in ft) and len(ft) > 1:
                first[preds[min(range(len(ft)), key=lambda i: ft[i])]] += 1
        n = len(eps)
        print(f"task {t}: {' & '.join(preds)}")
        print(f"   n={n}  success {c['success']}  partial {c['partial']}  undone {c['undone']}  "
              f"none {c['none']}" + (f"   partial holds: {dict(held)}" if held else ""))
        for i, p in enumerate(preds):
            ever = sum(e["first_true"][i] is not None for e in eps)
            print(f"   - {p}: ever held {100 * ever / n:.0f}%, at end {100 * sum(e['final'][i] for e in eps) / n:.0f}%")
    n = sum(total.values())
    print(f"\nall tasks, {n} episodes: " + ", ".join(
        f"{k} {100 * total[k] / n:.1f}%" for k in ("success", "partial", "undone", "none")))


if __name__ == "__main__":
    main(sys.argv[1:])
