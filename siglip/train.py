"""Training loop and checkpoint utilities."""

from pathlib import Path

import torch.optim as optim

from siglip.config import Config, TrainConfig
from siglip.model import ImageClassifier


def train(config: Config) -> int:
    """Train an image classifier.

    Args:
        config: Validated application configuration.

    Returns:
        Process exit code.
    """
    raise NotImplementedError("Image classifier training is not implemented yet")


def get_optimizer(model: ImageClassifier, config: TrainConfig) -> optim.Optimizer:
    """Create the configured optimizer.

    Args:
        model: Image classifier to optimize.
        config: Training configuration.

    Returns:
        Optimizer for the model parameters.
    """
    raise NotImplementedError


def save_checkpoint(
    model: ImageClassifier,
    optimizer: optim.Optimizer,
    epoch: int,
    checkpoint_dir: Path,
) -> None:
    """Save a training checkpoint."""
    raise NotImplementedError


def load_checkpoint(
    model: ImageClassifier,
    optimizer: optim.Optimizer,
    checkpoint_path: Path,
) -> int:
    """Load a training checkpoint and return its epoch."""
    raise NotImplementedError
