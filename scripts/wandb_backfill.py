"""One-off: mirror the existing results/benchmark.tsv rows into W&B.

Each row becomes an `eval` run (tagged `backfill`) with the same config and
summary keys that benchmark.py --wandb logs, the per-position episode table
when a record exists, and a link to the checkpoint artifact. Training curves
from before tracking existed are not recoverable; checkpoints' training data
is inferred from their names and marked as inferred.

  python wandb_backfill.py            # all rows
  python wandb_backfill.py --dry-run  # list what would be logged
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tracking  # noqa: E402
from benchmark import RESULTS, STAGES, read_episodes  # noqa: E402

MODELS = os.path.join(tracking.REPO, "models")
NUMERIC = (["stage_score", "success_1", "success_k", "mean_attempts", "lifted_1"]
           + [f"rate_{s}" for s in STAGES]
           + ["pos_err_med_mm", "pos_err_p95_mm", "detect_reject_pct",
              "reported_success", "signal_precision", "signal_recall"])


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    import wandb

    rows = list(csv.DictReader(open(RESULTS), delimiter="\t"))
    print(f"{len(rows)} rows in {RESULTS}")
    for i, r in enumerate(rows):
        label = r["label"] or "nolabel"
        mode = r.get("mode") or "policy"
        ptag = r["perception"]
        if ptag == "noisy":
            ptag = f"noisy_n{r['pos_noise_mm']}_b{r['pos_bias_mm']}"
        raw = "_raw" if r["raw_grip"] == "True" else ""
        stem = r["checkpoint"].replace(".pt", "")
        name = f"eval-{label}-{stem}-{mode.replace(':', '-')}-{ptag}{raw}"
        path = os.path.join(MODELS, r["checkpoint"])
        print(f"[{i + 1:2d}/{len(rows)}] {name}")
        if args.dry_run:
            continue

        ck = torch.load(path, map_location="cpu", weights_only=False) if os.path.exists(path) else {}
        config = {k: r[k] for k in ["label", "checkpoint", "frame", "grip_loss", "raw_grip",
                                    "time_feature", "positions", "attempts", "perception",
                                    "pos_noise_mm", "pos_bias_mm", "perception_hz",
                                    "commit", "date"]}
        config.update(mode=mode, eval_set=r.get("eval_set") or "heldout2026_n200",
                      protocol=r.get("protocol") or tracking.PROTOCOL,
                      train_data=tracking.data_reference(path, ck), backfilled=True)
        run = tracking.init("eval", name, config, group=label,
                            tags=[label, mode, r["perception"], config["eval_set"],
                                  config["protocol"], "backfill"])
        if os.path.exists(path):
            tracking.use_checkpoint(run, path, {"train_data": config["train_data"]})
        run.summary.update({k: as_float(r.get(k)) for k in NUMERIC if as_float(r.get(k)) is not None})

        funnel = wandb.Table(columns=["stage", "reached_pct"],
                             data=[[s, float(r[f"rate_{s}"])] for s in STAGES])
        log = {"stage_funnel": wandb.plot.bar(funnel, "stage", "reached_pct",
                                              title="Stage funnel (attempt 1)")}
        eps = read_episodes(r.get("episodes", ""))
        if eps:
            log["episodes"] = wandb.Table(
                columns=["position", "first_stage", "success", "attempts",
                         "skill_reported_success", "failure_reason"],
                data=[[int(k), e["first_stage"], e["success"] == "1", int(e["attempts"]),
                       e["reported"], e["reason"]] for k, e in sorted(eps.items())])
        run.log(log)
        run.finish()
    print("done")


if __name__ == "__main__":
    main()
