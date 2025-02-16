from dataclasses import dataclass


@dataclass
class Config:
    learning_rate: float
    n_episodes: int
    start_epsilon: float
    final_epsilon: float
    last_epsilon_episode: int
    discount_factor: float


cliff_walking = Config(
    learning_rate=0.01,
    n_episodes=5000,
    start_epsilon=1.0,
    final_epsilon=0.01,
    last_epsilon_episode=2500,
    discount_factor=0.95,
)

frozen_lake = Config(
    learning_rate=0.1,
    n_episodes=3000,
    start_epsilon=1.0,
    final_epsilon=0.01,
    last_epsilon_episode=500,
    discount_factor=0.95,
)
