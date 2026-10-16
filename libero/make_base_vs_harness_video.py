"""Side-by-side video: the base model alone vs the same model inside Conductor.

Left: Kesvill/smolvla_libero_90 given the task's full instruction (0% on
LIBERO-Long). Right: the same model with the full harness (step instructions,
start-state check + recovery, retry; harness M3). Same task, seed and init
state. The right side is labelled with the step in force and the harness's
last event from its log.

  python make_base_vs_harness_video.py OUT.mp4 [RUNS_DIR]   # ~/experiments/libero_runs/kesvill
"""

import json
import os
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_retry_video import BAR, F_TEXT, F_TITLE, FPS, W, harness_log, panel  # noqa: E402

RUNS = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/experiments/libero_runs/kesvill")
CLIPS = [(1000, 2, 7), (1000, 0, 5)]  # (seed, task, episode), both succeed with Conductor
TASKS = {0: "put both the alphabet soup and the tomato sauce in the basket",
         2: "turn on the stove and put the moka pot on it"}


def base_panel(frame, task, t, n_frames, success):
    img = Image.new("RGB", (W, W + BAR), (20, 22, 26))
    img.paste(Image.fromarray(frame), (0, BAR))
    d = ImageDraw.Draw(img)
    d.text((8, 6), "Model alone", fill=(230, 140, 90), font=F_TITLE)
    words, lines, line = TASKS[task].split(), [], ""
    for w in words:  # wrap the full instruction over two lines
        if len(line) + len(w) > 52:
            lines.append(line)
            line = ""
        line += w + " "
    lines.append(line)
    for k, l in enumerate(lines[:2]):
        d.text((8, 28 + 16 * k), ("> " if k == 0 else "  ") + l.strip(), fill=(235, 235, 235), font=F_TEXT)
    if t >= n_frames - 1:
        msg, c = ("SUCCESS", (90, 200, 120)) if success else ("FAILED (out of time)", (230, 110, 90))
        d.rectangle([0, BAR + W - 30, W, BAR + W], fill=(0, 0, 0))
        d.text((8, BAR + W - 24), msg, fill=c, font=F_TITLE)
    return np.asarray(img)


def main(out):
    frames = []
    for seed, task, ep in CLIPS:
        name = f"libero_10_{task}/eval_episode_{ep}.mp4"
        base = [f for f in imageio.get_reader(f"{RUNS}/subgoals_s{seed}/videos/{name}")]
        harn = [f for f in imageio.get_reader(f"{RUNS}/m3/s{seed}/videos/{name}")]
        base_ok = json.load(open(f"{RUNS}/subgoals_s{seed}/eval_info.json"))["per_task"]
        base_ok = [p for p in base_ok if p["task_id"] == task][0]["metrics"]["successes"][ep]
        h = harness_log("m3", seed, task, ep)
        for t in range(max(len(base), len(harn)) + FPS):
            pa = base_panel(base[min(t, len(base) - 1)], task, t, len(base), base_ok)
            pb = panel(harn[min(t, len(harn) - 1)], "Same model + Conductor", (110, 190, 255),
                       h, t, len(harn), h["success"])
            frames.append(np.concatenate([pa, np.full((W + BAR, 6, 3), 20, np.uint8), pb], axis=1))
    imageio.mimwrite(out, frames, fps=FPS, quality=7)
    print(f"wrote {out}: {len(frames)} frames")


if __name__ == "__main__":
    main(sys.argv[1])
