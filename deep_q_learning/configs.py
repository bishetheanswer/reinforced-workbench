from dataclasses import dataclass


@dataclass
class Config:
    env_name: str
    learning_rate: float
    discount_factor: float
    initial_epsilon: float
    final_epsilon: float
    last_epsilon_episode: int
    batch_size: int
    experience_buffer_size: int
    experience_buffer_start_size: int
    target_net_update_freq: int
    n_episodes: int


lunar_lander_config = Config(
    env_name="LunarLander-v3",
    learning_rate=1e-4,
    discount_factor=0.99,
    initial_epsilon=1.0,
    final_epsilon=0.01,
    last_epsilon_episode=500,
    batch_size=32,
    experience_buffer_size=10000,
    experience_buffer_start_size=2500,
    target_net_update_freq=1000,
    n_episodes=1000,
)

breakout = Config(
    env_name="ALE/Breakout-v5",
    learning_rate=1e-4,
    discount_factor=0.99,
    initial_epsilon=1.0,
    final_epsilon=0.01,
    last_epsilon_episode=25000,
    batch_size=32,
    experience_buffer_size=10000,
    experience_buffer_start_size=10000,
    target_net_update_freq=1000,
    n_episodes=100000,
)
