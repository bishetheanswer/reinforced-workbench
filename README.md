# Reinforced Workbench
This is a library whose goal is to implement various Reinforcement Learning algorithms
in a modular and extensible way. The library is designed to be used as a workbench for
experimenting with different algorithms and environments. The library is implemented in
Python and uses PyTorch for neural network implementations.

## Installation
To install the library, clone the repository and install the dependencies using pip:

```bash
pip install -r requirements.txt
```

If you get an error related to `swig` during the installation of the dependencies, you
may need to install `swig` using your package manager.

## Algorithms
The following algorithms are implemented in the library (the algorithms have their
respective README files with more detailed information in their directories):
- Q-Learning: A simple Q-Learning implementation with epsilon-greedy exploration.
- Deep Q-Learning (DQN): A PyTorch implementation of Deep Q-Learning with experience
replay and epsilon-greedy exploration. Furthermore DQN supports experiment tracking
with Weight & Biases.
