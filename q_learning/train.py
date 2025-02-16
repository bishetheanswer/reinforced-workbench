import argparse
from datetime import datetime

import configs
import gymnasium as gym
from configs import Config
from gymnasium.wrappers import RecordVideo
from logger import setup_logger
from qlearning import Qlearning

RESULTS_BASE_PATH = "results"


def get_results_path(env_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{RESULTS_BASE_PATH}/{env_name}/{timestamp}"


def get_env_and_config(env_name: str) -> tuple[gym.Env, Config]:
    """Instantiate an environment and return it along its config based on env name."""
    match env_name:
        case "cliff_walking":
            return gym.make(
                "CliffWalking-v0", render_mode="rgb_array"
            ), configs.cliff_walking
        case "frozen_lake":
            return gym.make(
                "FrozenLake-v1", is_slippery=False, render_mode="rgb_array"
            ), configs.frozen_lake
        case _:
            raise ValueError(f"Unknown environment: {env_name}")


def train(env_name: str, recording_freq: int) -> None:
    """Train a Q-learning agent in the specified environment."""
    logger = setup_logger(f"q_{env_name}")
    logger.info(f"Training Q-learning agent on {env_name} environment")

    results_path = get_results_path(env_name)

    env, cfg = get_env_and_config(env_name)
    env = RecordVideo(
        env,
        video_folder=results_path,
        name_prefix="training",
        episode_trigger=lambda x: x % recording_freq == 0 or x == cfg.n_episodes - 1,
    )

    agent = Qlearning(
        env=env,
        learning_rate=cfg.learning_rate,
        initial_epsilon=cfg.start_epsilon,
        final_epsilon=cfg.final_epsilon,
        last_epsilon_episode=cfg.last_epsilon_episode,
        discount_factor=cfg.discount_factor,
    )

    for episode in range(cfg.n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.update(state, action, reward, terminated, next_state)

            done = terminated or truncated
            state = next_state
            episode_reward += reward

        logger.info(
            f"Episode: {episode}, Epsilon: {agent.epsilon}, Reward: {episode_reward}"
        )

        # decay epsilon after each episode
        agent.decay_epsilon()

    env.close()

    agent.save(f"{results_path}/agent.pkl")
    # save the q table as json for easier inspection
    agent.save(f"{results_path}/agent.json", as_json=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        choices=["cliff_walking", "frozen_lake"],
        default="cliff_walking",
        help="Name of the environment. Can be: cliff_walking, frozen_lake",
    )
    parser.add_argument(
        "--recording_freq",
        type=int,
        default=100,
        help="Frequency at which to record the environment",
    )
    args = parser.parse_args()
    train(args.env_name, args.recording_freq)
