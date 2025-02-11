from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import utils
from configs import Config
from torch import Tensor
from utils import Experience, ExperienceBuffer


class DQNNetwork(nn.Module):
    """Neural network architecture for DQN with fully connected layers.

    Args:
        input_size: Size of the input layer (shape of the observation space)
        output_size: Size of the output layer (number of possible actions)
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQNNetworkConv(nn.Module):
    """Convolutional neural network architecture for DQN.

    Args:
        input_shape: Shape of the input tensor (channels, height, width)
        output_size: Size of the output layer (number of possible actions)
    """

    def __init__(self, input_shape: tuple, output_size: int) -> None:
        super(DQNNetworkConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # calculate the size of the output of the convolutional layers
        # to pass it to the fully connected layers
        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        self.fc = nn.Sequential(
            nn.Linear(size, 512), nn.ReLU(), nn.Linear(512, output_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network."""
        x = self.conv(x)
        x = self.fc(x)
        return x


class DQNAgent:
    """Base class for Deep Q-Learning agents.

    Args:
        env: Gymnasium environment
        learning_rate: Learning rate for the optimizer
        initial_epsilon: Initial value for epsilon in epsilon-greedy policy
        final_epsilon: Final value for epsilon in epsilon-greedy policy
        last_epsilon_episode: Episode number at which epsilon reaches its final value
        discount_factor: Discount factor for future rewards
        batch_size: Size of batches used for training
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        final_epsilon: float,
        last_epsilon_episode: float,
        discount_factor: float,
        batch_size: int,
    ) -> None:
        self.device = self._select_torch_device()
        self.dqn_network = DQNNetwork(
            env.observation_space.shape[0], env.action_space.n
        ).to(self.device)
        self.target_network = DQNNetwork(
            env.observation_space.shape[0], env.action_space.n
        ).to(self.device)

        self.env = env
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = (initial_epsilon - final_epsilon) / last_epsilon_episode

        self.discount_factor = discount_factor
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(
            self.dqn_network.parameters(), lr=learning_rate
        )

    def _select_torch_device(self) -> torch.device:
        """Select the appropriate device (MPS, CUDA, or CPU) for torch operations."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def get_action(self, state: np.ndarray) -> int:
        """Epsilon greedy policy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(state)

    def get_best_action(self, state: np.ndarray) -> int:
        """Select the best action for the given state based on current Q-values."""
        state_tensor = torch.tensor(state).to(self.device)
        # reshape the tensor from [n] to [1, n]
        state_tensor.unsqueeze_(0)
        q_values = self.dqn_network(state_tensor)
        return torch.argmax(q_values).item()

    def update(
        self,
        buffer: ExperienceBuffer,
    ) -> None:
        """Update the agent's network based on a batch of experiences."""
        self.optimizer.zero_grad()
        batch = buffer.sample(self.batch_size)
        loss_t = self.calc_loss(batch)
        loss_t.backward()
        self.optimizer.step()

    def sync_target_network(self) -> None:
        """Synchronize target network weights with the main network."""
        self.target_network.load_state_dict(self.dqn_network.state_dict())

    def decay_epsilon(self) -> None:
        """Decay epsilon value for epsilon-greedy policy."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def has_converged(self, rewards: list[float]) -> bool:
        """Check if the agent has converged to an optimal policy.

        By default this method always returns False, but it can be overridden.
        """
        return False

    def calc_loss(self, batch: list[Experience]) -> Tensor:
        """Calculate the loss for the given batch of experiences."""
        states_t, actions_t, rewards_t, dones_t, new_states_t = utils.batch_to_tensors(
            batch, self.device
        )

        state_action_values = (
            self.dqn_network(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        )

        with torch.no_grad():
            # get the max q value of each next state
            # we select [0] because max returns a tuple with the max values and
            # the indices, but we are only interested in the value
            next_state_values = self.target_network(new_states_t).max(1)[0]
            next_state_values[dones_t] = 0.0
            # avoid tracking operations for gradient computation
            next_state_values = next_state_values.detach()

        expected_state_action_values = (
            rewards_t + self.discount_factor * next_state_values
        )
        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def save(self, path: str) -> None:
        """Save the agent's network to a file."""
        torch.save(self.dqn_network.state_dict(), path)


class LunarLanderDQNAgent(DQNAgent):
    """DQN Agent specifically for the LunarLander environment."""

    def has_converged(self, rewards: list[float]) -> bool:
        """Check if the agent has converged to an optimal policy.

        Based on gymnasium's docs (https://gymnasium.farama.org/environments/box2d/lunar_lander/):
            An episode is considered a solution if it scores at least 200 points.
        """
        return len(rewards) >= 100 and np.mean(rewards[-100:]) >= 200


class AtariDQNAgent(DQNAgent):
    """DQN Agent specifically for Atari games, using convolutional networks."""

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        final_epsilon: float,
        last_epsilon_episode: float,
        discount_factor: float,
        batch_size: int,
    ) -> None:
        super().__init__(
            env,
            learning_rate,
            initial_epsilon,
            final_epsilon,
            last_epsilon_episode,
            discount_factor,
            batch_size,
        )

        # override the dqn_network with a convolutional network
        self.dqn_network = DQNNetworkConv(
            env.observation_space.shape, env.action_space.n
        ).to(self.device)

        self.target_network = DQNNetworkConv(
            env.observation_space.shape, env.action_space.n
        ).to(self.device)

        # reinitialize the optimizer with the new network parameters
        self.optimizer = torch.optim.Adam(
            self.dqn_network.parameters(), lr=learning_rate
        )


def get_agent(env: gym.Env, env_name: str, cfg: Config) -> DQNAgent:
    """Factory function to create appropriate DQN agent based on environment.

    Args:
        env: Gymnasium environment
        env_name: Name of the environment
        cfg: Configuration object with hyperparameters

    Returns:
        DQNAgent: Agent for the given environment
    """
    match env_name:
        case "lunar_lander":
            return LunarLanderDQNAgent(
                env=env,
                learning_rate=cfg.learning_rate,
                initial_epsilon=cfg.initial_epsilon,
                final_epsilon=cfg.final_epsilon,
                last_epsilon_episode=cfg.last_epsilon_episode,
                discount_factor=cfg.discount_factor,
                batch_size=cfg.batch_size,
            )
        case "breakout":
            return AtariDQNAgent(
                env=env,
                learning_rate=cfg.learning_rate,
                initial_epsilon=cfg.initial_epsilon,
                final_epsilon=cfg.final_epsilon,
                last_epsilon_episode=cfg.last_epsilon_episode,
                discount_factor=cfg.discount_factor,
                batch_size=cfg.batch_size,
            )
        case _:
            return DQNAgent(
                env=env,
                learning_rate=cfg.learning_rate,
                initial_epsilon=cfg.initial_epsilon,
                final_epsilon=cfg.final_epsilon,
                last_epsilon_episode=cfg.last_epsilon_episode,
                discount_factor=cfg.discount_factor,
                batch_size=cfg.batch_size,
            )
