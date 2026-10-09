"""Scripted experts for the skills after pick: place, push, home.

Like ScriptedGrasp they read the simulator's true cube pose (they are
demonstrators, not skills), ramp a setpoint toward each phase target under a
speed limit and leash it to the arm.

Unlike ScriptedGrasp they have **no hidden timers**: every phase change is
triggered by something a policy can observe (TCP position, gripper width,
where the cube is relative to its target). The pick expert's fixed 120-step
hold had to be patched into the policy's inputs (v3.0); these experts avoid
the problem by construction.

  ScriptedPlace  tuck into a carry posture, carry a held cube over a target,
                 lower it, open, retreat
  ScriptedPush   push the cube to a target with the closed fingers: an x leg,
                 then a y leg, each with line following; rise clear
  ScriptedHome   rise clear of the table, then move the joints home
"""

import numpy as np

from controller import TOP_DOWN, ScriptedGrasp
from env import CUBE_HALF, GRIPPER_CLOSE, GRIPPER_OPEN, TABLE_HEIGHT

REST_Z = TABLE_HEIGHT + CUBE_HALF


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class PhasedExpert(ScriptedGrasp):
    """ScriptedGrasp's ramp/leash/IK loop with observable phase transitions.

    Subclasses define PHASES as (name, speed [m/s], timeout), or override
    build_phases() when the sequence depends on the start state, plus
    target(), gripper() and ready() (-> bool: may this phase end now?).
    """

    PHASES = []

    def build_phases(self):
        return self.PHASES

    def reset(self):
        super().reset()
        # ScriptedGrasp.act reads phases[i][3] as the setpoint speed.
        self.phases = [(n, 0, None, speed, timeout) for n, speed, timeout in self.build_phases()]

    def on_enter(self, name):
        """Latch whatever the new phase needs."""

    def ready(self):
        raise NotImplementedError

    def _advance(self, target_pos):
        name, _, _, _, timeout = self.phases[self.phase_idx]
        self.phase_steps += 1
        ready = self.ready()
        if self.phase_steps >= timeout:
            self.timed_out = True
            ready = True
        if not ready:
            return
        if self.phase_idx == len(self.phases) - 1:
            self.done = True
        else:
            self.phase_idx += 1
            self.phase_steps = 0
            self.on_enter(self.phase)


# Carry posture: the home configuration with the wrist turned so the hand
# points straight down (TOP_DOWN). Pick reaches the whole workspace from here.
Q_CARRY = np.array([0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, -3 * np.pi / 4])


def joint_ramp(q_set, q_goal, max_step):
    return q_set + np.clip(q_goal - q_set, -max_step, max_step)


class ScriptedPlace(PhasedExpert):
    """Tuck, carry a held cube over `target_xy`, set it down, release, retreat.

    tuck: pick leaves the arm stretched up (elbow nearly straight, joint 4
    near its limit). Lowering from there with Cartesian IK pitches the shoulder
    forward and drags the forearm on the table: 17/100 stalled, and neither a
    posture term nor limit-aware IK fixed it (7-8/100). So the place first
    moves the joints to Q_CARRY, a posture pick works from, then works in
    Cartesian space. Ends when the joints are within 0.02 rad (observable).

    Posture: the arm is redundant (7 joints, 6-DoF hand pose), and a BC policy
    on joint deltas drifts along the redundant direction (joint 5 went 11 std
    outside the demos before the arm whipped). The IK's nullspace term pulls
    toward Q_CARRY so demos (with --joint-noise) show how to come back.
    """

    PHASES = [
        ("tuck", 0.0, 300),
        ("carry", 0.15, 400),
        ("lower", 0.08, 300),
        ("release", 0.05, 100),
        ("retreat", 0.15, 200),
    ]

    def __init__(self, env, target_xy, carry_height=0.12, set_clearance=0.003,
                 release_width=0.075, retreat_height=0.10, max_joint_speed=0.8,
                 nullspace_scale=1.0, **kwargs):
        kwargs["nullspace_scale"] = nullspace_scale
        self.target_xy = np.asarray(target_xy, dtype=float)
        self.carry_height = carry_height
        self.set_clearance = set_clearance
        self.release_width = release_width
        self.retreat_height = retreat_height
        self.max_joint_step = max_joint_speed * kwargs.get("control_dt", 0.02)
        super().__init__(env, **kwargs)
        self.ik.q_home = Q_CARRY.copy()  # posture the nullspace term pulls toward

    def reset(self):
        super().reset()
        self.q_set = self.env.arm_qpos
        self.grip_offset = None
        self.release_pos = None

    def _tcp_for_cube(self, cube_target):
        return cube_target + self.grip_offset

    def act(self):
        if self.phase != "tuck":
            return super().act()
        self.q_set = joint_ramp(self.q_set, Q_CARRY, self.max_joint_step)
        self._advance(None)
        return self.q_set.copy(), GRIPPER_CLOSE

    def target(self):
        tx, ty = self.target_xy
        if self.phase == "tuck":
            return self.env.tcp_pos
        if self.phase == "carry":
            return self._tcp_for_cube(np.array([tx, ty, REST_Z + self.carry_height]))
        if self.phase == "lower":
            return self._tcp_for_cube(np.array([tx, ty, REST_Z + self.set_clearance]))
        if self.phase == "release":
            return self.release_pos.copy()
        return self.release_pos + np.array([0.0, 0.0, self.retreat_height])

    def gripper(self):
        return GRIPPER_CLOSE if self.phase in ("tuck", "carry", "lower") else GRIPPER_OPEN

    def on_enter(self, name):
        if name == "carry":
            # Where the cube sits in the hand; the targets place the cube, not
            # the TCP. Cartesian ramp restarts from where the tuck ended.
            self.grip_offset = self.env.tcp_pos - self.env.cube_pos
            self.setpoint = self.env.tcp_pos.copy()
        elif name == "release":
            self.release_pos = self.env.tcp_pos.copy()

    def ready(self):
        if self.phase == "tuck":
            return np.abs(Q_CARRY - self.env.arm_qpos).max() < 0.02
        err = np.linalg.norm(self.target() - self.env.tcp_pos)
        if self.phase == "carry":
            return err < 0.010
        if self.phase == "lower":
            # 6 mm: the cube can slip a few mm in the fingers when it touches
            return err < 0.006
        if self.phase == "release":
            return self.env.gripper_width > self.release_width
        return err < 0.010


class ScriptedPush(PhasedExpert):
    """Push the cube along the table to `target_xy` with the closed fingers.

    The closed fingers are a flat pusher with axis-aligned faces, and a flat
    face slides the cube along its own normal: diagonal pushes missed by
    ~37 mm, axis-aligned ones by 4-7 mm. So the push is split into an x leg
    then a y leg (fixed order: which leg is active follows from the cube's
    offset to the target, no memory needed). This also keeps the cube
    axis-aligned, which pick and the detector assume.

    Each leg: approach above the standoff point behind the cube, descend,
    push with line following (heading turned back toward the leg's line in
    proportion to drift, capped), rise clear. A leg ends when the cube reaches
    the target along it, so a sideways miss never turns into chasing the cube.
    """

    STANDOFF = CUBE_HALF + 0.015 + 0.015  # cube half + pusher half + gap
    CONTACT = CUBE_HALF + 0.012           # pusher centre to cube centre in contact
    LEAD = 0.02                           # aim this far into the cube
    STEER_GAIN = 10.0                     # rad of heading per m of drift
    STEER_MAX = np.radians(30)
    MIN_LEG = 0.006                       # skip legs shorter than this

    def __init__(self, env, target_xy, hover_height=0.08, push_z=REST_Z,
                 tolerance=0.002, retreat_height=0.08, nullspace_scale=1.0, **kwargs):
        self.target_xy = np.asarray(target_xy, dtype=float)
        self.push_z = push_z
        self.tolerance = tolerance
        self.retreat_height = retreat_height
        super().__init__(env, hover_height=hover_height, nullspace_scale=nullspace_scale,
                         **kwargs)
        self.ik.q_home = Q_CARRY.copy()  # posture term, as in ScriptedPlace

    def build_phases(self):
        off = self.target_xy - self.env.cube_pos[:2]
        self.legs = [a for a in (0, 1) if abs(off[a]) >= self.MIN_LEG]
        phases = []
        for _ in self.legs:
            phases += [("approach", 0.35, 300), ("descend", 0.10, 250),
                       ("push", 0.06, 500), ("rise", 0.15, 200)]
        return phases or [("rise", 0.15, 200)]

    def reset(self):
        self.leg = 0
        self.rise_from = None
        super().reset()
        self.start_leg()

    def start_leg(self):
        if not self.legs:
            self.d0 = np.zeros(2)
            return
        axis = self.legs[self.leg]
        cube = self.env.cube_pos[:2]
        self.d0 = np.zeros(2)
        self.d0[axis] = np.sign(self.target_xy[axis] - cube[axis])
        self.n0 = np.array([-self.d0[1], self.d0[0]])
        # The leg's line: through the cube now, along the axis.
        self.line_point = cube.copy()

    def along(self):
        """Distance still to go along the current leg (negative = past it)."""
        return float((self.target_xy - self.env.cube_pos[:2]) @ self.d0)

    def heading(self):
        drift = float((self.env.cube_pos[:2] - self.line_point) @ self.n0)
        a = -np.clip(self.STEER_GAIN * drift, -self.STEER_MAX, self.STEER_MAX)
        c, s = np.cos(a), np.sin(a)
        return np.array([c * self.d0[0] - s * self.d0[1], s * self.d0[0] + c * self.d0[1]])

    def target(self):
        cube = self.env.cube_pos[:2]
        if self.phase in ("approach", "descend"):
            xy = cube - self.d0 * self.STANDOFF
            z = self.push_z + (self.hover_height if self.phase == "approach" else 0.0)
            return np.array([xy[0], xy[1], z])
        if self.phase == "push":
            h = self.heading()
            xy = cube - h * (self.CONTACT - np.clip(self.along(), 0.0, self.LEAD))
            return np.array([xy[0], xy[1], self.push_z])
        last = self.phase_idx == len(self.phases) - 1
        up = self.retreat_height if last else self.hover_height
        return self.rise_from + np.array([0.0, 0.0, up])

    def gripper(self):
        return GRIPPER_CLOSE

    def on_enter(self, name):
        if name == "rise":
            self.rise_from = self.env.tcp_pos.copy()
        elif name == "approach":
            self.leg += 1
            self.start_leg()

    def ready(self):
        err = np.linalg.norm(self.target() - self.env.tcp_pos)
        if self.phase == "approach":
            return err < 0.010
        if self.phase == "descend":
            return err < 0.006
        if self.phase == "push":
            return self.along() < self.tolerance
        return err < 0.010


class ScriptedHome:
    """Rise clear of the table in Cartesian space, then move the joints home.

    The gripper keeps whatever command it is given (home while holding keeps
    holding). Done when every joint is within `tol` rad of home.
    """

    def __init__(self, env, gripper_ctrl=GRIPPER_OPEN, clear_z=REST_Z + 0.12,
                 max_joint_speed=0.8, control_dt=0.02, tol=0.01, timeout=400):
        from controller import DiffIK
        self.env = env
        self.ik = DiffIK(env)
        self.q_home = self.ik.q_home
        self.grip = gripper_ctrl
        self.clear_z = clear_z
        self.max_step = max_joint_speed * control_dt
        self.tol = tol
        self.timeout = timeout
        self.reset()

    def reset(self):
        self.phase = "rise" if self.env.tcp_pos[2] < self.clear_z - 0.01 else "joints"
        self.rise_target = self.env.tcp_pos.copy()
        self.rise_target[2] = self.clear_z
        self.setpoint = self.env.tcp_pos.copy()
        self.q_set = None
        self.steps = 0
        self.done = self.timed_out = False

    def act(self):
        env = self.env
        self.steps += 1
        if self.phase == "rise":
            delta = self.rise_target - self.setpoint
            dist = np.linalg.norm(delta)
            if dist > 0.15 * 0.02:
                delta *= 0.15 * 0.02 / dist
            self.setpoint = self.setpoint + delta
            arm = self.ik.solve(self.setpoint, TOP_DOWN)
            if np.linalg.norm(self.rise_target - env.tcp_pos) < 0.01:
                self.phase = "joints"
        else:
            # Ramp a joint setpoint rather than stepping from the measured
            # angles: current + small delta leaves the servo too little error
            # to overcome its damping, and the wrist creeps.
            if self.q_set is None:
                self.q_set = env.arm_qpos
            self.q_set = joint_ramp(self.q_set, self.q_home, self.max_step)
            arm = self.q_set
            if np.abs(self.q_home - env.arm_qpos).max() < self.tol:
                self.done = True
        if self.steps >= self.timeout:
            self.timed_out = self.done = True
        return arm, self.grip
