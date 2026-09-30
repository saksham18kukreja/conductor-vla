"""Experiment tracking: Weights & Biases mirror of training and evaluation.

Local files stay the source of truth (results/benchmark.tsv, results/episodes/,
models/); W&B mirrors them for dashboards and lineage. Everything is opt-in
(--wandb on train_bc.py / benchmark.py).

What goes to W&B: configs, metrics, per-position episode tables, and
checkpoints (small model artifacts). What stays local: demonstration data
(referenced by content hash and metadata only) and videos.

Comparability: every evaluation carries EVAL_SET (which positions) and
PROTOCOL (how they are scored). Only results sharing both are directly
comparable. Bump PROTOCOL whenever scoring changes.
"""

import hashlib
import os
import subprocess

import numpy as np

ENTITY = "saksham-kukreja18-robotics"
PROJECT = "Agentic robotics"

# p1: 6 stages (reach, close, grasp, rise, lift, hold); success = lift held
# 50 steps with both fingers on the cube; 700 steps per attempt; retries start
# from where the last attempt left the cube (arm home).
PROTOCOL = "p1"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def eval_set_id(seed, n):
    return f"heldout{seed}_n{n}"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dataset_fingerprint(path):
    """Reference to a local demo file: identity and shape, never the data."""
    d = np.load(path, allow_pickle=True)
    fp = {
        "file": os.path.basename(path),
        "sha256": file_sha256(path),
        "episodes": int(len(d["episode_lengths"])),
        "transitions": int(len(d["obs"])),
        "mean_episode_len": float(np.mean(d["episode_lengths"])),
    }
    if "collection" in d.files:  # written by collect_demos.py since tracking was added
        fp["collection"] = {k: v.item() if hasattr(v, "item") else v
                            for k, v in d["collection"].item().items()}
    return fp


def checkpoint_artifact_name(checkpoint_path):
    return os.path.basename(checkpoint_path).replace(".pt", "")


def init(job_type, name, config, group=None, tags=None, enabled=True):
    """Start a W&B run, or return None when tracking is off."""
    if not enabled:
        return None
    import wandb
    return wandb.init(entity=ENTITY, project=PROJECT, job_type=job_type, name=name,
                      group=group, tags=tags or [], config=config, reinit="finish_previous")


def log_checkpoint(run, path, metadata):
    """Upload a checkpoint as a model artifact (versioned by content)."""
    import wandb
    art = wandb.Artifact(checkpoint_artifact_name(path), type="model", metadata=metadata)
    art.add_file(path)
    run.log_artifact(art)
    return art


def use_checkpoint(run, path, metadata=None):
    """Link an evaluation to the checkpoint it used. Logs the file if W&B has
    no artifact with identical content yet, so lineage is never broken."""
    import wandb
    if run.settings.mode == "offline":
        return  # W&B can't link artifacts offline; the checkpoint name is in the config
    art = wandb.Artifact(checkpoint_artifact_name(path), type="model",
                         metadata=metadata or {})
    art.add_file(path)
    run.use_artifact(art)


# Checkpoints saved before tracking existed don't record their data; infer the
# demo file from the checkpoint name and mark it as inferred.
_NAME_TO_DEMOS = [("clean3000", "clean3000.npz"), ("clean1000", "clean1000.npz"),
                  ("1000_long", "clean1000.npz"), ("noisy02", "noisy02.npz"),
                  ("noisy05", "noisy05.npz"), ("state_demos", "state_demos.npz"),
                  ("300_long", "state_demos.npz"), ("bc_policy", "state_demos.npz")]
_fingerprints = {}


def data_reference(checkpoint_path, ck):
    """Training-data reference for a checkpoint: stored, or inferred from its name."""
    if "data" in ck:
        return dict(ck["data"], inferred=False)
    stem = os.path.basename(checkpoint_path)
    for key, demo in _NAME_TO_DEMOS:
        if key in stem:
            path = os.path.join(REPO, "demos", demo)
            if not os.path.exists(path):
                return {"file": demo, "inferred": True, "sha256": "missing locally"}
            if path not in _fingerprints:
                _fingerprints[path] = dataset_fingerprint(path)
            return dict(_fingerprints[path], inferred=True)
    return {"file": "unknown", "inferred": True}
