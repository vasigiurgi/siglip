"""Image-classification model interfaces."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class ImageClassifier(nn.Module, ABC):
    """Base interface for image classifiers."""

    def __init__(self, num_classes: int) -> None:
        """Initialize classifier metadata.

        Args:
            num_classes: Number of output classes.

        Raises:
            ValueError: If ``num_classes`` is not positive.
        """
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for an image batch.

        Args:
            images: RGB image batch in NCHW layout, typically float32 in [0, 1].

        Returns:
            Class logits with shape ``(batch_size, num_classes)``.
        """
