# CartPole Policy Gradient Training

A PyTorch implementation of a policy gradient (REINFORCE) algorithm for solving the CartPole-v1 environment from OpenAI's Gymnasium.

## Overview

This project trains a neural network-based policy to balance a pole on a moving cart. The agent learns through reinforcement learning using the policy gradient method with discounted rewards and reward normalization.

## Problem Description

**CartPole-v1** is a classic control problem where:
- **Goal**: Keep a pole balanced on top of a moving cart
- **Observation Space**: 4 continuous values (cart position, cart velocity, pole angle, pole angular velocity)
- **Action Space**: 2 discrete actions (push cart left or right)
- **Reward**: +1 for each timestep the pole remains upright
- **Success Criterion**: Achieve a mean reward of 195 over 100 consecutive episodes (target: 200)

## Solution Approach

### Algorithm: Policy Gradient (REINFORCE)

The implementation uses a neural network policy to directly learn the action probabilities:

1. **Policy Network**: A 3-layer neural network with:
   - Input: 4-dimensional observation space
   - Hidden layers: 64 neurons with ReLU activation
   - Output: 2 action probabilities (softmax)

2. **Training Loop**:
   - Sample actions from the learned policy
   - Collect rewards and log probabilities during episodes
   - Compute discounted rewards (γ = 0.99)
   - Normalize rewards for stable learning
   - Compute loss: -Σ(log_prob × discounted_reward)
   - Backpropagate and update weights using Adam optimizer

3. **Key Features**:
   - **Discounted Rewards**: Future rewards weighted by γ^t
   - **Reward Normalization**: Standardize returns to reduce variance
   - **Periodic Rendering**: Capture environment frames every 10 episodes
   - **Early Stopping**: Stop when mean reward ≥ 200 over 100 episodes

## Dependencies

- **Python**: 3.11
- **PyTorch**: 2.2.2
- **Gymnasium**: [classic-control]
- **NumPy**: < 2.0
- **Matplotlib**: For visualization
- **Plotly**: For interactive plots

See `environment.yml` for complete dependency list.

## Installation

### Using Conda

```bash
conda env create -f environment.yml
conda activate rl
```

### Manual Installation

```bash
pip install torch==2.2.2 torchvision==0.17.2
pip install 'numpy<2.0'
pip install 'gymnasium[classic-control]'
pip install matplotlib plotly
```

## Usage

Run the notebook to train the policy:

```bash
jupyter notebook cartpole.ipynb
```

Or execute cells individually to:
1. Import dependencies and initialize the environment
2. Inspect observation/action spaces
3. Define and create the policy network
4. Test policy output before training
5. Train the policy (500 episodes)
6. Visualize training metrics and episode recordings

## Project Structure

- `cartpole.ipynb`: Main training notebook with all code cells
- `environment.yml`: Conda environment specification
- `README.md`: This file

## Key Cells in the Notebook

| Cell | Purpose |
|------|---------|
| 1-2 | Import libraries and explore available environments |
| 3-8 | Initialize CartPole environment and inspect spaces |
| 9-11 | Define and create policy network architecture |
| 12-13 | Test policy network output on sampled observations |
| 14-17 | **Main training loop** (500 episodes) |
| 18 | Generate animation of a recorded episode |

## Results

The policy gradient algorithm successfully solves the CartPole-v1 environment by:
- Learning to balance the pole consistently
- Achieving target reward of 200+ over 100 episodes
- Demonstrating convergence within ~300-400 episodes
- Producing smooth, stable control behavior

## Visualization

The notebook includes:
- **Training Progress**: Loss and reward tracking over episodes
- **Interactive Plots**: Episode rewards and mean reward trends
- **Animation**: Matplotlib animation of pole balancing in the learned policy

## Mathematical Details

### Discounted Reward Calculation
```
G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
```

### Loss Function
```
Loss = -Σ log(π(a_t|s_t)) * G_t
```

### Gradient Update
```
θ ← θ - α ∇Loss(θ)
```

Where:
- `π(a|s)`: Policy probability for action `a` in state `s`
- `G_t`: Discounted cumulative reward
- `θ`: Network parameters
- `α`: Learning rate

## Future Improvements

- Baseline subtraction to reduce variance
- Actor-Critic methods
- Entropy regularization for exploration
- Multi-episode batch training
- Hyperparameter tuning (learning rate, hidden size, γ)

## References

- [OpenAI Gymnasium Docs](https://gymnasium.farama.org/)
- [Policy Gradient Methods](https://spinningup.openai.com/en/latest/algorithms/pg.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
