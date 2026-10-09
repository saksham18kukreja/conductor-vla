"""A BC checkpoint trained on goal_obs (place, push) as a callable policy."""

import numpy as np
import torch

from collect_demos import GRIPPER_SCALE
from skills.features import goal_obs
from train_bc import GRIP_THRESHOLD, MLPPolicy


class GoalPolicy:
    def __init__(self, path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location=device, weights_only=False)
        self.policy = MLPPolicy(ck["obs_dim"], ck["act_dim"], hidden=ck["hidden"]).to(device)
        self.policy.load_state_dict(ck["state_dict"])
        self.policy.eval()
        self.norm = ck["norm"]
        self.skill = ck.get("skill")
        self.frame = ck.get("frame", "stored")
        self.grip_loss = ck.get("grip_loss", "bce")
        self.time_feature = False
        self.device = device

    def reset(self):
        pass

    def act(self, env, obj_pos, goal_xy):
        om, osd, am, asd = self.norm
        o = goal_obs(env, obj_pos, goal_xy)
        with torch.no_grad():
            x = torch.as_tensor((o - om) / osd, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
            a = self.policy(x).squeeze(0).cpu().numpy() * asd + am
        arm = np.clip(env.arm_qpos + a[:7], env.arm_limits[:, 0], env.arm_limits[:, 1])
        grip = float(a[7] > GRIP_THRESHOLD[self.grip_loss]) * GRIPPER_SCALE
        return arm, grip
