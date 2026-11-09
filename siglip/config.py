"""Typed configuration for image classification."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """Classifier output configuration."""

    model_config = ConfigDict(extra="forbid")

    num_classes: int = Field(gt=0)


class DataConfig(BaseModel):
    """CIFAR-10 data configuration."""

    model_config = ConfigDict(extra="forbid")

    dataset_url: str
    data_dir: Path
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    seed: int


class TrainConfig(BaseModel):
    """Training runtime configuration."""

    model_config = ConfigDict(extra="forbid")

    device: str
    checkpoint_dir: Path


class Config(BaseModel):
    """Application configuration."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    data: DataConfig
    train: TrainConfig

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load and validate configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration.

        Returns:
            The validated application configuration.

        Raises:
            FileNotFoundError: If the configuration file is missing.
            ValueError: If the YAML document is not a mapping.
        """
        with config_path.open(encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
        if not isinstance(raw_config, dict):
            raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
        return cls.model_validate(raw_config)
