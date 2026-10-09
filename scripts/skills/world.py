"""World model: what the robot believes about the scene, from perception only.

Objects are tracked by id with a position estimate from table_cam and the
color detector. Detections run at `hz`; a rejected detection (object too
hidden) keeps the last good estimate. An object never seen has no position.
Robot state (joints, gripper, TCP) is read from the robot directly, which is
allowed: it is the robot's own proprioception, not object ground truth.

Held objects: while `holding` is set and the gripper is commanded closed, the
held object moves with the hand, so between detections its estimate follows
the gripper at the control rate (TCP pose + the object's offset in the
gripper frame, re-measured at every accepted detection: the object does slide
along the finger pads, up to ~1 cm while the arm tucks). A 10 Hz estimate of
a carried object lags by up to ~12 mm, and the place policy, trained on the
exact position, failed every episode with it (0/30 vs 27/30 at 50 Hz).

Pushed objects: when an object's last detection was rejected (hidden, e.g. by
the hand pushing it toward the robot, between it and the camera) and the TCP
is against it at table height, the object can't overlap the pusher: its
estimate is moved out to PUSH_CONTACT from the TCP along the line between
them. Without this a push toward the robot runs on a frozen estimate and
times out (all 9 such pushes in 25 episodes).
"""

from dataclasses import dataclass

import numpy as np

CONTROL_HZ = 50
PUSH_CONTACT = 0.032  # m, pusher centre to cube centre when touching (ScriptedPush)


@dataclass
class ObjectEstimate:
    id: str
    color: str
    position: np.ndarray = None  # None until first good detection
    visible: float = 0.0         # visible fraction at the last detection attempt
    last_seen: int = None        # world step of the last accepted detection
    fresh: bool = False          # the last detection attempt was accepted

    @property
    def detected(self):
        return self.position is not None


class WorldModel:
    def __init__(self, env, hz=10, min_visible=0.6, oracle=False):
        from perception.camera import Camera
        from perception.detectors import ColorDetector

        self.env = env  # for proprioception only
        self.every = max(1, round(CONTROL_HZ / hz))
        # oracle: object positions are the simulator's true ones (at the same
        # rate). Only for evaluation, to separate perception failures from
        # skill failures (benchmark.py --skill ... --perception gt).
        self.oracle = oracle
        self.cam = Camera(env)
        # One object for now: the red cube. More objects = more detectors here.
        self.detectors = {"cube_red": ColorDetector(self.cam, min_visible=min_visible)}
        self.reset()

    def reset(self):
        """Forget everything (new scene)."""
        self.objects = {oid: ObjectEstimate(oid, oid.split("_")[-1]) for oid in self.detectors}
        self.holding = None
        self.gripper_cmd_closed = False
        self.grip_offset = None  # held object's offset in the gripper frame
        self.t = 0
        self.detect_calls = self.detect_rejects = 0

    def observe(self):
        """Run every detector now and fold accepted detections into memory."""
        for oid, det in self.detectors.items():
            if self.oracle:
                p, info = self.env.cube_pos, {"visible": 1.0}
            else:
                p, info = det.detect()
            est = self.objects[oid]
            est.visible = info.get("visible", 0.0)
            self.detect_calls += 1
            est.fresh = p is not None
            if p is None:
                self.detect_rejects += 1
            else:
                est.position, est.last_seen = p, self.t

    def update(self):
        """Called by the Runner before every control step."""
        if self.t % self.every == 0:
            self.observe()
            self._latch_grip_offset()
        self._follow_gripper()
        self._follow_pusher()
        self.t += 1

    def _follow_pusher(self):
        tcp = self.env.tcp_pos
        for oid, est in self.objects.items():
            if est.fresh or not est.detected or oid == self.holding:
                continue
            if abs(tcp[2] - est.position[2]) > 0.02:
                continue
            d = est.position[:2] - tcp[:2]
            dist = float(np.linalg.norm(d))
            if 1e-6 < dist < PUSH_CONTACT:
                est.position = est.position.copy()
                est.position[:2] = tcp[:2] + d / dist * PUSH_CONTACT

    def _carrying(self):
        est = self.objects.get(self.holding) if self.holding else None
        return est if est is not None and est.detected and self.gripper_cmd_closed else None

    def _latch_grip_offset(self):
        est = self._carrying()
        if est is None:
            self.grip_offset = None
            return
        if est.last_seen == self.t:  # accepted just now
            self.grip_offset = self.env.tcp_mat.T @ (est.position - self.env.tcp_pos)

    def _follow_gripper(self):
        est = self._carrying()
        if est is not None and self.grip_offset is not None:
            est.position = self.env.tcp_pos + self.env.tcp_mat @ self.grip_offset

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
