"""Image classification inference interface."""

import logging
from pathlib import Path

import torch

from siglip.model import ImageClassifier

logger = logging.getLogger(__name__)


class Inferencer:
    """Run image classification with a loaded model."""

    def __init__(self, model_path: Path, device: str) -> None:
        """Initialize an inferencer.

        Args:
            model_path: Path to model checkpoint.
            device: Explicit device name, such as ``cpu`` or ``cuda``.
        """
        self.model_path = model_path
        self.device = torch.device(device)
        self.model: ImageClassifier | None = None

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for an image batch.

        Args:
            images: RGB image batch in NCHW layout.

        Returns:
            Class logits with shape ``(batch_size, num_classes)``.
        """
        raise NotImplementedError
