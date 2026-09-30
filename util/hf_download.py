# MIT License
#
# Copyright (c) 2025 DMPO Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Hugging Face download utilities for DMPO checkpoints and datasets.

Usage:
    In config files, specify paths with the `hf://` prefix:

    # For checkpoints (fine-tuning)
    base_policy_path: hf://pretrained_checkpoints/DMPO_pretrained_gym_checkpoints/gym_improved_meanflow/hopper-medium-v2_best.pt

    # For datasets (pre-training)
    train_dataset_path: hf://gym/hopper-medium-v2/train.npz

    Files will be automatically downloaded from the Hugging Face repositories.
"""

import os
import logging
from huggingface_hub import hf_hub_download

log = logging.getLogger(__name__)

# Default Hugging Face repositories for DMPO
DMPO_CHECKPOINT_REPO = "Guowei-Zou/OGPO-checkpoints"
DMPO_DATASET_REPO = "Guowei-Zou/OGPO-datasets"

# Legacy alias for backward compatibility
DMPO_HF_REPO = DMPO_CHECKPOINT_REPO

# Prefix used to identify Hugging Face paths in config files
HF_PREFIX = "hf://"


def is_hf_path(path: str) -> bool:
    """Check if a path is a Hugging Face path (starts with hf://)."""
    return path is not None and path.startswith(HF_PREFIX)


def parse_hf_path(path: str) -> str:
    """
    Parse the Hugging Face path and return the relative file path.

    Args:
        path: Path starting with hf://, e.g., "hf://pretrained_checkpoints/xxx.pt"

    Returns:
        Relative file path in the HF repository, e.g., "pretrained_checkpoints/xxx.pt"
    """
    if not is_hf_path(path):
        raise ValueError(f"Path does not start with {HF_PREFIX}: {path}")
    return path[len(HF_PREFIX):]


def download_from_hf(
    path: str,
    repo_id: str = DMPO_HF_REPO,
    cache_dir: str = None,
) -> str:
    """
    Download a file from Hugging Face Hub.

    Args:
        path: Path starting with hf://, e.g., "hf://pretrained_checkpoints/xxx.pt"
        repo_id: Hugging Face repository ID (default: DMPO_HF_REPO)
        cache_dir: Optional custom cache directory

    Returns:
        Local path to the downloaded file
    """
    if not is_hf_path(path):
        raise ValueError(f"Path does not start with {HF_PREFIX}: {path}")

    # Parse the relative path in the HF repository
    filename = parse_hf_path(path)

    log.info(f"Downloading checkpoint from Hugging Face: {repo_id}/{filename}")

    # Download the file (uses cache automatically)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        repo_type="model",
    )

    log.info(f"Checkpoint downloaded to: {local_path}")
    return local_path


def resolve_checkpoint_path(path: str, repo_id: str = DMPO_CHECKPOINT_REPO) -> str:
    """
    Resolve a checkpoint path - download from HF if needed.

    Args:
        path: Either a local path or a hf:// path
        repo_id: Hugging Face repository ID for hf:// paths

    Returns:
        Local path to the checkpoint file
    """
    if path is None:
        return None

    if is_hf_path(path):
        return download_from_hf(path, repo_id=repo_id)

    # Return as-is for local paths
    return path


def download_dataset_from_hf(
    path: str,
    repo_id: str = DMPO_DATASET_REPO,
    cache_dir: str = None,
) -> str:
    """
    Download a dataset file from Hugging Face Hub.

    Args:
        path: Path starting with hf://, e.g., "hf://gym/hopper-medium-v2/train.npz"
        repo_id: Hugging Face dataset repository ID (default: DMPO_DATASET_REPO)
        cache_dir: Optional custom cache directory

    Returns:
        Local path to the downloaded file
    """
    if not is_hf_path(path):
        raise ValueError(f"Path does not start with {HF_PREFIX}: {path}")

    # Parse the relative path in the HF repository
    filename = parse_hf_path(path)

    log.info(f"Downloading dataset from Hugging Face: {repo_id}/{filename}")

    # Download the file (uses cache automatically)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        repo_type="dataset",
    )

    log.info(f"Dataset downloaded to: {local_path}")
    return local_path


def resolve_dataset_path(path: str, repo_id: str = DMPO_DATASET_REPO) -> str:
    """
    Resolve a dataset path - download from HF if needed.

    Args:
        path: Either a local path or a hf:// path
        repo_id: Hugging Face dataset repository ID for hf:// paths

    Returns:
        Local path to the dataset file
    """
    if path is None:
        return None

    if is_hf_path(path):
        return download_dataset_from_hf(path, repo_id=repo_id)

    # Return as-is for local paths
    return path
