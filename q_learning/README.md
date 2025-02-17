# Q-Learning Implementation

An implementation of Q-Learning that can be used to train agents for various Gymnasium
environments.

## Training
To train an agent, use the `train.py` script with the desired environment:

```bash
# Train on Cliff Walking
python train.py --env_name cliff_walking --recording_freq 100

# Train on Frozen Lake
python train.py --env_name frozen_lake
```

## Core Components
- `qlearning.py`: Contains Q-Learning agent implementation
- `configs.py`: Environment-specific configurations and hyperparameters
- `train.py`: Training loop implementation
