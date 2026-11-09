"""Evaluation utilities for image classification."""

import torch
from torch.utils.data import DataLoader

from siglip.model import ImageClassifier


def evaluate(
    model: ImageClassifier,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a classifier on labeled images.

    Args:
        model: Image classifier.
        data_loader: Batches of image tensors and integer class labels.
        device: Device used for evaluation.

    Returns:
        Classification metrics.
    """
    raise NotImplementedError


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Compute classification metrics from predictions and targets.

    Args:
        predictions: Class logits with shape ``(batch_size, num_classes)``.
        targets: Integer class labels with shape ``(batch_size,)``.

    Returns:
        Classification metrics.
    """
    raise NotImplementedError
