import json
import os
import pickle
from collections import defaultdict

import gymnasium as gym
import numpy as np


class Qlearning:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        final_epsilon: float,
        last_epsilon_episode: float,
        discount_factor: float,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = (initial_epsilon - final_epsilon) / last_epsilon_episode
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, state: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[state]))

    def update(
        self,
        state: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_state: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_state])
        temporal_difference = (
            reward
            + self.discount_factor * future_q_value
            - self.q_values[state][action]
        )

        self.q_values[state][action] = (
            self.q_values[state][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, path: str, as_json: bool = False) -> None:
        """Save the Q-values to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        q_values_dict = {k: v.tolist() for k, v in self.q_values.items()}
        if as_json:
            with open(path, "w") as f:
                json.dump(q_values_dict, f)
        else:
            with open(path, "wb") as f:
                pickle.dump(q_values_dict, f)
