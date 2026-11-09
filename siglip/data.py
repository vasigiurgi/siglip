"""Image classification data interfaces."""

import logging
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader, Dataset

from siglip.config import DataConfig

logger = logging.getLogger(__name__)


class ImageClassificationDataset(Dataset[tuple[torch.Tensor, int]], ABC):
    """Dataset returning an image tensor and its integer class label."""

    def __init__(self, config: DataConfig) -> None:
        """Initialize dataset.

        Args:
            config: Data configuration object.
        """
        self.config = config

    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get one image and its class label.

        Args:
            idx: Index of item to retrieve.

        Returns:
            An RGB image tensor and its integer class label.
        """
        raise NotImplementedError


def get_data_loaders(
    config: DataConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create training and test data loaders.

    Args:
        config: Data configuration object.

    Returns:
        Training and test data loaders.
    """
    # TODO: Implement data loader creation
    raise NotImplementedError
