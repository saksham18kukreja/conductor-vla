"""Side-by-side video: the same LIBERO episode without and with retry.

Left: recovery only (harness M4). Right: recovery + retry (M3). Same seed, task
and init state; the runs drift apart slightly after the first recovery (the
policy and recovery timing aren't bit-identical across runs). Each side is
labelled with the step instruction in force and the harness's last event,
read from the harness logs (one video frame per control step).

  python make_retry_video.py OUT.mp4 [RUNS_DIR]   # RUNS_DIR: ~/experiments/libero_runs/kesvill
"""

import json
import os
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUNS = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/experiments/libero_runs/kesvill")
# (seed, task, episode): episodes M4 failed and M3 rescued with a retry
CLIPS = [(1000, 2, 7), (1000, 0, 5)]
FPS = 40  # control runs at 20 Hz: 2x real time
W = 360
BAR = 74


def harness_log(run, seed, task, episode):
    eps = [json.loads(l) for l in open(f"{RUNS}/{run}/harness_s{seed}.jsonl")]
    return [h for h in eps if h["task_id"] == task][episode]


def font(size):
    for f in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


F_TITLE, F_TEXT = font(15), font(12)


def state_at(h, t):
    """Instruction in force and the last harness event at control step t."""
    step, last, recovering = 0, "", False
    for et, msg in h["events"]:
        if et > t:
            break
        if msg.endswith(" done") and msg.startswith("step "):
            step = int(msg.split()[1]) + 1
        elif "undone" in msg:
            step = int(msg.split()[1])
        if "recovering" in msg:
            recovering = True
        if msg.startswith("recovered"):
            recovering = False
        if "start state ok" not in msg:
            last = msg
    instr = "recovery pose" if recovering else h["plan"][min(step, len(h["plan"]) - 1)]
    return instr, last


def panel(frame, title, colour, h, t, n_frames, success):
    img = Image.new("RGB", (W, W + BAR), (20, 22, 26))
    img.paste(Image.fromarray(frame), (0, BAR))
    d = ImageDraw.Draw(img)
    d.text((8, 6), title, fill=colour, font=F_TITLE)
    instr, last = state_at(h, t)
    d.text((8, 28), f"> {instr}", fill=(235, 235, 235), font=F_TEXT)
    d.text((8, 46), last[:58], fill=(160, 170, 180), font=F_TEXT)
    if t >= n_frames - 1:
        msg, c = ("SUCCESS", (90, 200, 120)) if success else ("FAILED (out of time)", (230, 110, 90))
        d.rectangle([0, BAR + W - 30, W, BAR + W], fill=(0, 0, 0))
        d.text((8, BAR + W - 24), msg, fill=c, font=F_TITLE)
    return np.asarray(img)


def main(out):
    frames = []
    for seed, task, ep in CLIPS:
        name = f"libero_10_{task}/eval_episode_{ep}.mp4"
        a = [f for f in imageio.get_reader(f"{RUNS}/m4c/s{seed}/videos/{name}")]
        b = [f for f in imageio.get_reader(f"{RUNS}/m3/s{seed}/videos/{name}")]
        ha, hb = harness_log("m4c", seed, task, ep), harness_log("m3", seed, task, ep)
        n = max(len(a), len(b))
        for t in range(n + FPS):  # hold the last frame ~1 s
            pa = panel(a[min(t, len(a) - 1)], "Recovery only", (230, 140, 90), ha, t, len(a), ha["success"])
            pb = panel(b[min(t, len(b) - 1)], "Recovery + retry", (110, 190, 255), hb, t, len(b), hb["success"])
            frames.append(np.concatenate([pa, np.full((W + BAR, 6, 3), 20, np.uint8), pb], axis=1))
    imageio.mimwrite(out, frames, fps=FPS, quality=7)
    print(f"wrote {out}: {len(frames)} frames")


if __name__ == "__main__":
    main(sys.argv[1])
