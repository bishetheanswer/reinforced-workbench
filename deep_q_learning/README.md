# Deep Q-Learning Implementation

A PyTorch implementation of Deep Q-Learning (DQN) that can be used to train agents for
various Gymnasium environments.

## Training
To train an agent, use the `train.py` script with the desired environment:

```bash
# Train on LunarLander
python train.py --env_name lunar_lander --recording_freq 100 --track

# Train on Breakout
python train.py --env_name breakout
```

## Core Components

- `agents.py`: Contains DQN agent implementations and neural network architectures
- `configs.py`: Environment-specific configurations and hyperparameters
- `train.py`: Training loop implementation
- `utils.py`: Helper functions and experience replay buffer implementation

## Agent Types

1. `DQNAgent`: Base agent with fully connected network
2. `LunarLanderDQNAgent`: Specialized agent for LunarLander environment
3. `AtariDQNAgent`: Agent with convolutional network for Atari games

## Adding New Environments
### 1. Create Environment Configuration
Add a new configuration in `configs.py` with appropriate hyperparameters.

### 2. Add Environment Setup Logic
Extend the `get_env_and_config()` function in `utils.py`.

### 3. Create a Specialized Agent (Optional)
If your environment needs specific handling (e.g., custom convergence criteria or
network architecture), create a new agent class in `agents.py`.

### 4. Update Agent Factory
Add your new environment to the agent factory in `agents.py` (get_agent()).

### 5. Add the environment to the choices in the train script
`choices=["lunar_lander", "breakout", "new_environment"]`,
