import argparse
import os
from dataclasses import asdict
from datetime import datetime
from random import randint

import agents
import ale_py
import gymnasium as gym
import utils
import wandb
from configs import Config
from gymnasium.wrappers import RecordVideo
from logger import setup_logger
from utils import Experience, ExperienceBuffer

gym.register_envs(ale_py)

RESULTS_BASE_PATH = "results"


def setup_wandb(
    project: str, name: str, tags: list[str], cfg: Config, notes: str = ""
) -> None:
    """Setup Weights & Biases for tracking the experiment."""
    api_key = os.getenv("WANDB_KEY")
    wandb.login(key=api_key)
    wandb.init(project=project, name=name, tags=tags, notes=notes, config=asdict(cfg))


def get_results_path(env_name: str) -> str:
    """Get the path where to store the results of the experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{RESULTS_BASE_PATH}/{env_name}/{timestamp}"


def train(env_name: str, recording_freq: int, track: bool) -> None:
    """Train a DQN agent in the specified environment."""
    logger = setup_logger(f"dqn_{env_name}")
    logger.info(f"Training DQN agent on {env_name} environment")

    results_path = get_results_path(env_name)

    env, cfg = utils.get_env_and_config(env_name)
    env = RecordVideo(
        env,
        video_folder=results_path,
        name_prefix="training",
        episode_trigger=lambda x: x % recording_freq == 0 or x == cfg.n_episodes - 1,
    )

    agent = agents.get_agent(env, env_name, cfg)
    buffer = ExperienceBuffer(capacity=cfg.experience_buffer_size)

    if track:
        setup_wandb(
            project=env_name,
            name=f"dqn-experience_buffer-{randint(0, 10000)}",
            tags=["dqn", "experience_buffer"],
            cfg=cfg,
        )
        wandb.watch(agent.dqn_network, log="all", log_freq=100, log_graph=True)

    episodes_rewards = []
    i = 1
    for episode in range(cfg.n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            exp = Experience(state, action, reward, terminated or truncated, next_state)
            buffer.append(exp)
            # wait until buffer has enough experience
            if len(buffer) < cfg.experience_buffer_start_size:
                continue

            if i % cfg.target_net_update_freq == 0:
                agent.sync_target_network()

            agent.update(buffer)

            done = terminated or truncated
            state = next_state
            i += 1
            episode_reward += reward
            episode_steps += 1

        logger.info(
            f"Episode: {episode}, Epsilon: {agent.epsilon}, Reward: {episode_reward}"
        )
        episodes_rewards.append(episode_reward)

        if track:
            wandb.log(
                {
                    "reward": episode_reward,
                    "episode_durantion_steps": episode_steps,
                    "epsilon": agent.epsilon,
                }
            )

        if agent.has_converged(episodes_rewards):
            logger.info(f"Converged after {episode} episodes")
            break

        # decay epsilon after each episode
        agent.decay_epsilon()

    env.close()

    agent.save(f"{results_path}/agent.pth")
    wandb.save(f"{results_path}/agent.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        choices=["lunar_lander", "breakout"],
        default="lunar_lander",
        help="Name of the environment. Can be: lunar_lander, breakout",
    )
    parser.add_argument(
        "--recording_freq",
        type=int,
        default=5000,
        help="Frequency at which to record the environment",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Track the experiment with Weights & Biases",
        default=False,
    )
    args = parser.parse_args()
    train(args.env_name, args.recording_freq, args.track)
