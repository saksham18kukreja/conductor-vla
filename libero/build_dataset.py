"""Build one LeRobot dataset for fine-tuning SmolVLA to follow LIBERO step instructions.

Three Apache-2.0 sources, re-written through LeRobot's own writer so every
episode has the same format, timing and video encoding (lerobot-train in this
LeRobot version takes one dataset; merging needs identical features):

  libero4   lerobot/libero: Spatial, Object, Goal, Long (1,693 demos), each with
            its own instruction (LIBERO-Long with the full two-step sentence).
  libero90  IPEC-COMMUNITY/libero_90_no_noops_lerobot (3,921 demos, v2.1, 20 fps
            label): single-step tasks in the LIBERO-Long scenes. Converted:
            wrist_image -> image2, extra state features dropped, gripper action
            1 = open / 0 = closed -> -1 = open / +1 = closed (as lerobot/libero
            and the LIBERO env), fps label 20 -> 10 (one frame per control step
            in both; verified by identical episode lengths).
  l10steps  lerobot/libero_10_subtask (500 LIBERO-Long demos with a step id per
            frame): every frame relabelled with the instruction of its step, in
            the wording Conductor sends (STEP_WORDING). Episodes with a step id
            that isn't mapped are skipped.

Output: fps 10, robot "panda", observation.images.image / image2 (256x256 RGB,
AV1 video), observation.state (8), action (7), one task string per frame.

  python build_dataset.py OUT_DIR [--sources libero4,libero90,l10steps] [--max-episodes N]
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchcodec.decoders import VideoDecoder

FPS = 10
CAMERAS = {"observation.images.image": "observation.images.image",
           "observation.images.image2": "observation.images.image2"}
FEATURES = {
    "observation.images.image": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.image2": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": [f"state_dim_{i}" for i in range(8)]},
    "action": {"dtype": "float32", "shape": (7,), "names": [f"action_dim_{i}" for i in range(7)]},
}

# (LIBERO-Long task, step id in lerobot/libero_10_subtask) -> step instruction.
# Same-scene LIBERO-90 wording where it exists, the same wording as harness/plans.py.
# Step ids found from each task's step order; in "put both moka pots on the stove"
# step 5 grasps the pot at x ~ 0.04 (moka_pot_2, LIBERO's left pot) and step 4
# the one at x ~ -0.09 (moka_pot_1, the right pot).
SOUP = "pick up the alphabet soup and put it in the basket"
CHEESE = "pick up the cream cheese box and put it in the basket"
STEP_WORDING = {
    ("turn on the stove and put the moka pot on it", 15): "turn on the stove",
    ("turn on the stove and put the moka pot on it", 4): "put the moka pot on the stove",
    ("put the black bowl in the bottom drawer of the cabinet and close it", 8):
        "put the black bowl in the bottom drawer of the cabinet",
    ("put the black bowl in the bottom drawer of the cabinet and close it", 2):
        "close the bottom drawer of the cabinet",
    ("put the yellow and white mug in the microwave and close it", 14):
        "put the yellow and white mug in the microwave",
    ("put the yellow and white mug in the microwave and close it", 1): "close the microwave",
    ("put both moka pots on the stove", 15): "turn on the stove",
    ("put both moka pots on the stove", 5): "put the left moka pot on the stove",
    ("put both moka pots on the stove", 4): "put the right moka pot on the stove",
    ("put both the alphabet soup and the cream cheese box in the basket", 9): SOUP,
    ("put both the alphabet soup and the cream cheese box in the basket", 12): CHEESE,
    ("put both the alphabet soup and the tomato sauce in the basket", 9): SOUP,
    ("put both the alphabet soup and the tomato sauce in the basket", 13):
        "pick up the tomato sauce and put it in the basket",
    ("put both the cream cheese box and the butter in the basket", 12): CHEESE,
    ("put both the cream cheese box and the butter in the basket", 11):
        "pick up the butter and put it in the basket",
    ("put the white mug on the left plate and put the yellow and white mug on the right plate", 6):
        "put the white mug on the left plate",
    ("put the white mug on the left plate and put the yellow and white mug on the right plate", 7):
        "put the yellow and white mug on the right plate",
    ("put the white mug on the plate and put the chocolate pudding to the right of the plate", 6):
        "put the white mug on the plate",
    ("put the white mug on the plate and put the chocolate pudding to the right of the plate", 3):
        "put the chocolate pudding to the right of the plate",
    ("pick up the book and place it in the back compartment of the caddy", 10):
        "pick up the book and place it in the back compartment of the caddy",
}


def frames_v3(root, ep, key, fps):
    """All frames of one episode from a v3 dataset video file, as HWC uint8."""
    path = os.path.join(root, f"videos/{key}/chunk-{int(ep[f'videos/{key}/chunk_index']):03d}/"
                              f"file-{int(ep[f'videos/{key}/file_index']):03d}.mp4")
    t0, n = float(ep[f"videos/{key}/from_timestamp"]), int(ep["length"])
    d = VideoDecoder(path)
    batch = d.get_frames_played_at(seconds=[t0 + k / fps for k in range(n)])
    return batch.data.permute(0, 2, 3, 1).numpy()


def v3_episodes(repo):
    root = snapshot_download(repo, repo_type="dataset")
    info = json.load(open(f"{root}/meta/info.json"))
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet"))])
    tasks = pd.read_parquet(f"{root}/meta/tasks.parquet")
    task_name = {int(i): str(t) for t, i in zip(tasks.index, tasks["task_index"])}
    data = {}
    for _, ep in eps.sort_values("episode_index").iterrows():
        key = (int(ep["data/chunk_index"]), int(ep["data/file_index"]))
        if key not in data:
            data = {key: pd.read_parquet(f"{root}/data/chunk-{key[0]:03d}/file-{key[1]:03d}.parquet")}
        df = data[key]
        rows = df[df.episode_index == ep["episode_index"]].sort_values("frame_index")
        yield root, info["fps"], ep, rows, task_name


def source_libero4(max_eps):
    for k, (root, fps, ep, rows, names) in enumerate(v3_episodes("lerobot/libero")):
        if max_eps and k >= max_eps:
            return
        imgs = {c: frames_v3(root, ep, c, fps) for c in CAMERAS}
        tasks = [names[int(t)] for t in rows["task_index"]]
        yield imgs, rows, tasks


def source_l10steps(max_eps, skipped):
    k = 0
    for root, fps, ep, rows, names in v3_episodes("lerobot/libero_10_subtask"):
        full = names[int(rows["task_index"].iloc[0])]
        labels = [STEP_WORDING.get((full, int(s))) for s in rows["subtask_index"]]
        if any(l is None for l in labels):
            skipped.append((int(ep["episode_index"]), full, sorted(set(int(s) for s in rows["subtask_index"]))))
            continue
        if max_eps and k >= max_eps:
            return
        k += 1
        imgs = {c: frames_v3(root, ep, c, fps) for c in CAMERAS}
        yield imgs, rows, labels


def source_libero90(max_eps):
    root = snapshot_download("IPEC-COMMUNITY/libero_90_no_noops_lerobot", repo_type="dataset")
    tasks = {d["task_index"]: d["task"] for d in map(json.loads, open(f"{root}/meta/tasks.jsonl"))}
    eps = [json.loads(l) for l in open(f"{root}/meta/episodes.jsonl")]
    info = json.load(open(f"{root}/meta/info.json"))
    for k, ep in enumerate(sorted(eps, key=lambda e: e["episode_index"])):
        if max_eps and k >= max_eps:
            return
        i = ep["episode_index"]
        chunk = i // info["chunks_size"]
        rows = pd.read_parquet(f"{root}/data/chunk-{chunk:03d}/episode_{i:06d}.parquet").sort_values("frame_index")
        imgs = {}
        for dst, src in (("observation.images.image", "observation.images.image"),
                         ("observation.images.image2", "observation.images.wrist_image")):
            d = VideoDecoder(f"{root}/videos/chunk-{chunk:03d}/{src}/episode_{i:06d}.mp4")
            imgs[dst] = d[:].data.permute(0, 2, 3, 1).numpy()
        act = np.stack(rows["action"].to_numpy()).astype(np.float32)
        act[:, 6] = 1.0 - 2.0 * act[:, 6]  # 1 open / 0 closed -> -1 open / +1 closed
        rows = rows.assign(action=list(act))
        yield imgs, rows, [tasks[int(t)] for t in rows["task_index"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sources", default="libero4,libero90,l10steps")
    ap.add_argument("--max-episodes", type=int, default=0, help="per source (for testing)")
    ap.add_argument("--repo-id", default="saksham18kukreja/libero_conductor")
    args = ap.parse_args()

    ds = LeRobotDataset.create(args.repo_id, fps=FPS, features=FEATURES, root=args.out,
                               robot_type="panda", use_videos=True, image_writer_threads=8)
    skipped, counts, t0 = [], {}, time.time()
    gens = {"libero4": lambda: source_libero4(args.max_episodes),
            "libero90": lambda: source_libero90(args.max_episodes),
            "l10steps": lambda: source_l10steps(args.max_episodes, skipped)}
    for name in args.sources.split(","):
        n_ep = n_fr = 0
        for imgs, rows, tasks in gens[name]():
            n = len(rows)
            for c, f in imgs.items():
                if len(f) != n:
                    raise RuntimeError(f"{name}: {c} has {len(f)} frames for {n} rows")
            states = np.stack(rows["observation.state"].to_numpy()).astype(np.float32)
            actions = np.stack(rows["action"].to_numpy()).astype(np.float32)
            for j in range(n):
                ds.add_frame({"observation.images.image": imgs["observation.images.image"][j],
                              "observation.images.image2": imgs["observation.images.image2"][j],
                              "observation.state": states[j], "action": actions[j], "task": tasks[j]})
            ds.save_episode()
            n_ep += 1
            n_fr += n
            if n_ep % 100 == 0:
                print(f"{name}: {n_ep} episodes, {n_fr} frames, {time.time() - t0:.0f} s", flush=True)
        counts[name] = (n_ep, n_fr)
        print(f"{name} done: {n_ep} episodes, {n_fr} frames, {time.time() - t0:.0f} s", flush=True)
    ds.finalize()
    print("counts:", counts)
    print(f"skipped l10steps episodes (unmapped step ids): {len(skipped)} {skipped[:5]}")
    json.dump({"counts": counts, "skipped": skipped}, open(os.path.join(args.out, "build_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
