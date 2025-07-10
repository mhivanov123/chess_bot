# Chess RL Agent Framework

A basic framework for training reinforcement learning agents to play chess.

## Features

- Custom chess environment using `python-chess`
- Deep Q-Network (DQN) agent implementation
- Training infrastructure with logging and visualization
- Configurable hyperparameters
- Evaluation and testing utilities

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training a new agent

```bash
python train.py --config configs/dqn_config.yaml
```

### Evaluating a trained agent

```bash
python evaluate.py --model_path models/chess_agent.pth
```

### Playing against the agent

```bash
python play.py --model_path models/chess_agent.pth
```

## Project Structure

```
chess_bot/
├── configs/           # Configuration files
├── models/            # Trained model checkpoints
├── logs/              # Training logs and tensorboard files
├── src/
│   ├── environment/   # Chess environment implementation
│   ├── agents/        # RL agent implementations
│   ├── utils/         # Utility functions
│   └── training/      # Training loops and evaluation
├── train.py           # Main training script
├── evaluate.py        # Evaluation script
├── play.py           # Interactive play script
└── requirements.txt   # Dependencies
```

## Configuration

The framework uses YAML configuration files to manage hyperparameters. See `configs/dqn_config.yaml` for an example.

## Training

The training process includes:
- Experience replay buffer
- Target network updates
- Epsilon-greedy exploration
- Reward shaping for chess-specific objectives
- Tensorboard logging for monitoring

## Environment

The chess environment provides:
- Standard chess rules enforcement
- State representation as board features
- Action space of legal moves
- Reward function based on game outcome and material advantage 