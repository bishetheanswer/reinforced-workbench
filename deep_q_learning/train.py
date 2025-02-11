import argparse

import agents
import ale_py
import gymnasium as gym
import torch
import utils
from gymnasium.wrappers import RecordVideo
from utils import Experience, ExperienceBuffer

from logger import setup_logger

gym.register_envs(ale_py)


def train(env_name: str) -> None:
    """Train a DQN agent in the specified environment."""
    logger = setup_logger(f"dqn_{env_name}")
    logger.info(f"Training DQN agent on {env_name} environment")

    env, cfg = utils.get_env_and_config(env_name)
    env = RecordVideo(
        env,
        video_folder=f"videos/{env_name}",
        name_prefix="training",
        episode_trigger=lambda x: x % 100 == 0 or x == cfg.n_episodes - 1,
    )
    agent = agents.get_agent(env, env_name, cfg)
    buffer = ExperienceBuffer(capacity=cfg.experience_buffer_size)

    episodes_rewards = []
    i = 1
    for episode in range(cfg.n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

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

        logger.info(
            f"Episode: {episode}, Epsilon: {agent.epsilon}, Reward: {episode_reward}"
        )
        episodes_rewards.append(episode_reward)

        if agent.has_converged(episodes_rewards):
            logger.info(f"Converged after {episode} episodes")
            break

        # decay epsilon after each episode
        agent.decay_epsilon()

    torch.save(agent.dqn_network.state_dict(), f"{env_name}.pth")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        choices=["lunar_lander", "breakout"],
        default="lunar_lander",
        help="Name of the environment. Can be: lunar_lander, breakout",
    )
    args = parser.parse_args()
    train(args.env_name)
