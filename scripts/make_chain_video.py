"""Render a fixed skill chain run through the Runner (camera perception).

  pick(cube) -> place(cube, target) -> home() -> push(cube, target) -> home()

Each episode uses a held-out cube position; the current skill and its result
are drawn on the frame.

  python make_chain_video.py --episodes 3 --out ../videos/skill_chain.mp4
"""

import argparse
import os
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scenarios as sc  # noqa: E402
from benchmark import heldout_positions  # noqa: E402
from env import FrankaCubeLift  # noqa: E402
from skills import LIBRARY, Runner  # noqa: E402
from skills.world import WorldModel  # noqa: E402

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
FPS = 50
EVERY = 2  # render every 2nd control step, played at 25 fps = real time


def label(frame, lines):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    for i, s in enumerate(lines):
        d.text((10, 10 + 16 * i), s, fill=(255, 255, 255))
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="../videos/skill_chain.mp4")
    ap.add_argument("--camera", default="grasp_cam")
    args = ap.parse_args()

    env = FrankaCubeLift()
    world = WorldModel(env)
    lib = {
        "pick": LIBRARY["pick"](),
        "place": LIBRARY["place"](os.path.join(MODELS, "bc_place3000_s0.pt")),
        "push": LIBRARY["push"](os.path.join(MODELS, "bc_push3000_s0.pt")),
        "home": LIBRARY["home"](),
    }
    frames, status = [], []
    rng = np.random.default_rng(7)

    def on_step(skill, t, check):
        if t % EVERY == 0:
            frames.append(label(env.render(args.camera), status + [f"> {skill.name}  step {t}"]))

    runner = Runner(env, world, on_step=on_step)
    for ep, xy in enumerate(heldout_positions(args.episodes)):
        env.reset(cube_xy=xy)
        world.reset()
        place_t = sc.place_target(rng)
        push_t = sc.push_target(rng, place_t)
        chain = [("pick", {"object": "cube_red"}),
                 ("place", {"object": "cube_red", "target": f"{place_t[0]:.3f},{place_t[1]:.3f}"}),
                 ("home", {}),
                 ("push", {"object": "cube_red", "target": f"{push_t[0]:.3f},{push_t[1]:.3f}"}),
                 ("home", {})]
        status = [f"episode {ep + 1}"]
        for name, kw in chain:
            res = runner.execute(lib[name], **kw)
            msg = f"{name}{tuple(kw.values()) if kw else '()'}: {res.status.value}"
            if not res.ok:
                msg += f" [{res.reason.value}]"
            status.append(msg)
            print(msg, flush=True)
            if not res.ok:
                break
        for _ in range(25):  # hold the final frame ~1 s
            frames.append(label(env.render(args.camera), status))

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimwrite(out, frames, fps=FPS // EVERY, quality=7)
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
