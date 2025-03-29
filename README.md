# MADDPG with PettingZoo

This is a modernized implementation of the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm, adapted to work with PettingZoo environments.

## Setup

This project uses Poetry for dependency management. Make sure you have Poetry installed (see [Poetry installation](https://python-poetry.org/docs/#installation)).

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## Running the Training

The simplest way to run the training is:

```bash
# Run using the default configuration in config.py
poetry run train
```

You can also run with custom arguments:

```bash
# Example: Run with a different environment
poetry run python train_maddpg.py --env simple_spread --render_freq 100

# Example: Use SAC algorithm instead of MADDPG
poetry run python train_maddpg.py --use_sac --sac_alpha 0.05
```

## Available Environments

The following PettingZoo environments are supported:

- `simple` - Simple reference environment
- `simple_speaker_listener` - Cooperative communication task
- `simple_spread` - Cooperative navigation task
- `knights_archers_zombies` - Competitive team-based game
- `rps` - Rock, Paper, Scissors game

## Configuration

You can modify the default configuration in `config.py` to change the default behavior without having to pass command-line arguments.
