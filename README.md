# Image Classification

A PyTorch image-classification project scaffold using CIFAR-10.

## Setup

Install the project and development tools with `uv sync`.

On Linux x86_64, PyTorch and torchvision are resolved from the CUDA 12.8 wheel index. On other platforms, they are resolved from PyPI.

The starter configuration is in `configs/default.yaml`. It points to the CIFAR-10 Python archive and uses the dataset's ten class labels. The archive is not downloaded during setup. Training and the concrete model architecture are not implemented yet.
