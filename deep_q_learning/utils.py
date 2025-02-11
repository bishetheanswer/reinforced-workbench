from collections import deque
from dataclasses import dataclass

import configs
import gymnasium as gym
import numpy as np
import torch
from configs import Config
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
)

BatchTensors = tuple[
    torch.Tensor,  # current state
    torch.Tensor,  # actions
    torch.Tensor,  # rewards
    torch.Tensor,  # done || trunc
    torch.Tensor,  # next state
]


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    done_trunc: bool
    new_state: np.ndarray


class ExperienceBuffer:
    """Simple buffer to append and sample experience."""

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> list[Experience]:
        indices = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]


def batch_to_tensors(batch: list[Experience], device: torch.device) -> BatchTensors:
    """Transform a list of Experience into tensors."""
    states, actions, rewards, dones, new_state = [], [], [], [], []
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_trunc)
        new_state.append(e.new_state)
    states_t = torch.as_tensor(np.asarray(states))
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)
    dones_t = torch.BoolTensor(dones)
    new_states_t = torch.as_tensor(np.asarray(new_state))
    return (
        states_t.to(device),
        actions_t.to(device),
        rewards_t.to(device),
        dones_t.to(device),
        new_states_t.to(device),
    )


def get_env_and_config(env_name: str) -> tuple[gym.Env, Config]:
    """Instantiate an environment and return it along its config based on env name."""
    match env_name:
        case "lunar_lander":
            env = gym.make(
                "LunarLander-v3",
                continuous=False,
                enable_wind=False,
                render_mode="rgb_array",
            )
            return env, configs.lunar_lander_config

        case "breakout":
            env = gym.make(
                "ALE/Breakout-v5",
                frameskip=1,
                repeat_action_probability=0,
                render_mode="rgb_array",
            )
            env = AtariPreprocessing(
                env,
                noop_max=30,  # don't do anything for a random number of frames between 1 and 30
                frame_skip=4,  # play every 4th frame
                screen_size=84,  # resize the screen to 84x84
                terminal_on_life_loss=True,  # end the episode when a life is lost
                grayscale_obs=True,  # use grayscale observation
                grayscale_newaxis=False,  # add a new axis to the observation: (84, 84) -> (84, 84, 1)
                scale_obs=True,  # scale observation to (0, 1)
            )
            env = FrameStackObservation(env, stack_size=4)
            return env, configs.breakout
        case _:
            raise ValueError(f"Environment {env_name} not supported")
