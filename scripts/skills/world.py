"""World model: what the robot believes about the scene, from perception only.

Objects are tracked by id with a position estimate from table_cam and the
color detector. Detections run at `hz`; a rejected detection (object too
hidden) keeps the last good estimate. An object never seen has no position.
Robot state (joints, gripper, TCP) is read from the robot directly, which is
allowed: it is the robot's own proprioception, not object ground truth.
"""

from dataclasses import dataclass

import numpy as np

CONTROL_HZ = 50


@dataclass
class ObjectEstimate:
    id: str
    color: str
    position: np.ndarray = None  # None until first good detection
    visible: float = 0.0         # visible fraction at the last detection attempt
    last_seen: int = None        # world step of the last accepted detection

    @property
    def detected(self):
        return self.position is not None


class WorldModel:
    def __init__(self, env, hz=10, min_visible=0.6):
        from perception.camera import Camera
        from perception.detectors import ColorDetector

        self.env = env  # for proprioception only
        self.every = max(1, round(CONTROL_HZ / hz))
        self.cam = Camera(env)
        # One object for now: the red cube. More objects = more detectors here.
        self.detectors = {"cube_red": ColorDetector(self.cam, min_visible=min_visible)}
        self.reset()

    def reset(self):
        """Forget everything (new scene)."""
        self.objects = {oid: ObjectEstimate(oid, oid.split("_")[-1]) for oid in self.detectors}
        self.holding = None
        self.gripper_cmd_closed = False
        self.t = 0
        self.detect_calls = self.detect_rejects = 0

    def observe(self):
        """Run every detector now and fold accepted detections into memory."""
        for oid, det in self.detectors.items():
            p, info = det.detect()
            est = self.objects[oid]
            est.visible = info.get("visible", 0.0)
            self.detect_calls += 1
            if p is None:
                self.detect_rejects += 1
            else:
                est.position, est.last_seen = p, self.t

    def update(self):
        """Called by the Runner before every control step."""
        if self.t % self.every == 0:
            self.observe()
        self.t += 1

    def after_action(self, gripper_ctrl):
        self.gripper_cmd_closed = gripper_ctrl < 0.5 * 255.0

    # --------------------------------------------------------- robot state
    @property
    def gripper_width(self):
        return self.env.gripper_width

    @property
    def tcp_pos(self):
        return self.env.tcp_pos

    def snapshot(self):
        """Plain-data view, for Results and (later) a planner prompt."""
        return {
            "objects": {
                oid: {
                    "color": o.color,
                    "position": None if o.position is None else [round(float(v), 4) for v in o.position],
                    "visible": round(float(o.visible), 2),
                    "last_seen_step": o.last_seen,
                }
                for oid, o in self.objects.items()
            },
            "gripper": {
                "width_mm": round(1000 * float(self.gripper_width), 1),
                "commanded": "closed" if self.gripper_cmd_closed else "open",
                "holding": self.holding,
            },
            "step": self.t,
        }
