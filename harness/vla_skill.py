"""A VLA (SmolVLA via LeRobot) behind a skill interface.

The skill is "do this instruction": every control step it maps the LIBERO
observation to an action with the policy, using the instruction it was given
as the task text, through exactly the processing chain lerobot-eval uses:

  preprocess_observation -> task text -> env_preprocessor -> preprocessor
  -> policy.select_action -> postprocessor -> env_postprocessor

A VLA never says it is done, so termination is not part of the skill: the
Runner decides from the verifier, the step budget, or the episode ending.

The policy keeps an action queue (SmolVLA runs 50 actions per prediction) for
the whole batch, so reset() clears it for every env at once. Switching one
env's instruction while others keep theirs needs batch size 1.
"""

import numpy as np
import torch

from lerobot.envs import preprocess_observation
from lerobot.utils.constants import ACTION


class VLASkill:
    name = "vla"

    def __init__(self, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor):
        self.policy = policy
        self.env_preprocessor = env_preprocessor
        self.env_postprocessor = env_postprocessor
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def reset(self):
        """Start a new instruction: drop queued actions and processor state."""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def act(self, observation, instructions):
        """observation: raw vector-env observation; instructions: one str per env.
        -> actions, np.ndarray (n_envs, action_dim)."""
        obs = preprocess_observation(observation)
        obs["task"] = list(instructions)
        obs = self.env_preprocessor(obs)
        obs = self.preprocessor(obs)
        with torch.inference_mode():
            action = self.policy.select_action(obs)
        action = self.postprocessor(action)
        action = self.env_postprocessor({ACTION: action})[ACTION]
        action = action.to("cpu").numpy()
        assert action.ndim == 2, "actions should be (n_envs, action_dim)"
        return action

    @staticmethod
    def zero_action(n_envs, dim=7):
        return np.zeros((n_envs, dim), dtype=np.float32)
