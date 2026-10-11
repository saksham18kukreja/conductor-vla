"""Render a fixed skill chain run through the Runner (camera perception).

  pick(cube) -> place(cube, target) -> home() -> push(cube, target) -> home()

Each episode uses a held-out cube position; the current skill and its result
are drawn on the frame.

--chain push_pick runs the handoff that needs recovery instead, and
--chain loop the whole cycle with no home() steps (with --recover the
recovery pose takes their place where a skill needs it):

  push_pick  push(cube, target) -> pick(cube)
  loop       pick(cube) -> place(cube, target) -> push(cube, target) -> pick(cube)

--recover runs every call through skills/recovery.py: a refusal to start
(target not visible, wrong start state) sends the arm to the recovery pose
(home, fingers open) and the skill is tried once more.

  python make_chain_video.py --episodes 3 --out ../videos/skill_chain.mp4
  python make_chain_video.py --chain push_pick --recover --out ../videos/push_pick_recover.mp4
  python make_chain_video.py --chain loop --recover --out ../videos/loop_recover.mp4
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
from skills.recovery import execute_with_recovery  # noqa: E402
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
    ap.add_argument("--chain", choices=["full", "push_pick", "loop"], default="full")
    ap.add_argument("--recover", action="store_true",
                    help="recover from refusals to start (skills/recovery.py)")
    args = ap.parse_args()

    env = FrankaCubeLift()
    world = WorldModel(env)
    lib = {
        "pick": LIBRARY["pick"](),
        "place": LIBRARY["place"](os.path.join(MODELS, "bc_place3000_s0.pt")),
        "push": LIBRARY["push"](os.path.join(MODELS, "bc_push3000_s0.pt")),
        "home": LIBRARY["home"](),
        "recover": LIBRARY["recover"](),
    }
    frames, status = [], []
    rng = np.random.default_rng(7)

    def on_step(skill, t, check):
        if t % EVERY == 0:
            frames.append(label(env.render(args.camera), status + [f"> {skill.name}  step {t}"]))

    class LoggingRunner(Runner):
        """Writes each call's result to the overlay as soon as it returns."""

        def execute(self, skill, **kw):
            r = super().execute(skill, **kw)
            msg = f"{r.skill}{tuple(r.args.values()) if r.args else '()'}: {r.status.value}"
            if not r.ok:
                msg += f" [{r.reason.value}]"
            status.append(msg)
            print(msg, flush=True)
            if not r.ok and r.steps == 0:  # refused before moving: show it for ~1 s
                for _ in range(25):
                    frames.append(label(env.render(args.camera), status))
            return r

    runner = LoggingRunner(env, world, on_step=on_step)
    for ep, xy in enumerate(heldout_positions(args.episodes)):
        env.reset(cube_xy=xy)
        world.reset()
        place_t = sc.place_target(rng)
        push_t = sc.push_target(rng, place_t)
        if args.chain == "push_pick":
            push_t = sc.push_target(rng, np.asarray(xy))
            chain = [("push", {"object": "cube_red", "target": f"{push_t[0]:.3f},{push_t[1]:.3f}"}),
                     ("pick", {"object": "cube_red"})]
        elif args.chain == "loop":
            chain = [("pick", {"object": "cube_red"}),
                     ("place", {"object": "cube_red",
                                "target": f"{place_t[0]:.3f},{place_t[1]:.3f}"}),
                     ("push", {"object": "cube_red", "target": f"{push_t[0]:.3f},{push_t[1]:.3f}"}),
                     ("pick", {"object": "cube_red"})]
        else:
            chain = [("pick", {"object": "cube_red"}),
                     ("place", {"object": "cube_red",
                                "target": f"{place_t[0]:.3f},{place_t[1]:.3f}"}),
                     ("home", {}),
                     ("push", {"object": "cube_red", "target": f"{push_t[0]:.3f},{push_t[1]:.3f}"}),
                     ("home", {})]
        status = [f"episode {ep + 1}" + ("  (recovery on)" if args.recover else "")]
        for name, kw in chain:
            if args.recover:
                res, _ = execute_with_recovery(runner, lib[name], lib["recover"], **kw)
            else:
                res = runner.execute(lib[name], **kw)
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
