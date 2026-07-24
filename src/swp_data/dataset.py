"""Gold consumer: PyTorch Dataset for the GL23x45 windowed dTEC dataset.

Expected directory layout:
    data/gold/training_windows/
      train_tec_input.npy
      train_omni_input.npy
      train_target.npy
      train_window_start_times.npy
      val_*.npy
      test_*.npy
      metadata.json

Each item returns:
    tec_input   float32 tensor [6, 23, 45]
    omni_input  float32 tensor [6, n_driver_features]
    target      float32 tensor [3, 23, 45]
    timestamp   int64 epoch seconds for the window start
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .settings import DataLayout

Split = Literal["train", "val", "test"]

_DEFAULT_ROOT = DataLayout("data").training_windows


class DTECWindowDataset(Dataset):
    """Memory-mapped PyTorch Dataset for normalized dTEC forecast windows."""

    def __init__(self, root: str | Path = _DEFAULT_ROOT,
                 split: Split = "train") -> None:
        self.root = Path(root)
        self.split = split
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val, or test; got {split!r}")

        self.tec_input = np.load(self.root / f"{split}_tec_input.npy", mmap_mode="r")
        self.omni_input = np.load(self.root / f"{split}_omni_input.npy", mmap_mode="r")
        self.target = np.load(self.root / f"{split}_target.npy", mmap_mode="r")
        self.window_start_times = np.load(
            self.root / f"{split}_window_start_times.npy",
            mmap_mode="r",
        )

        with (self.root / "metadata.json").open() as f:
            self.metadata = json.load(f)

        self._validate_shapes()

    def _validate_shapes(self) -> None:
        n = self.tec_input.shape[0]
        if self.omni_input.shape[0] != n or self.target.shape[0] != n:
            raise ValueError(
                f"{self.split}: sample count mismatch: "
                f"tec={self.tec_input.shape}, omni={self.omni_input.shape}, "
                f"target={self.target.shape}"
            )
        if self.window_start_times.shape[0] != n:
            raise ValueError(
                f"{self.split}: timestamp count mismatch: "
                f"{self.window_start_times.shape[0]} vs {n}"
            )
        if self.tec_input.shape[1:] != (6, 23, 45):
            raise ValueError(f"{self.split}: bad tec_input shape {self.tec_input.shape}")
        n_driver_features = len(self.metadata["omni_features"])
        if self.omni_input.shape[1:] != (6, n_driver_features):
            raise ValueError(f"{self.split}: bad omni_input shape {self.omni_input.shape}")
        if self.target.shape[1:] != (3, 23, 45):
            raise ValueError(f"{self.split}: bad target shape {self.target.shape}")

    def __len__(self) -> int:
        return int(self.tec_input.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "tec_input": torch.from_numpy(np.array(self.tec_input[idx], dtype=np.float32, copy=True)),
            "omni_input": torch.from_numpy(np.array(self.omni_input[idx], dtype=np.float32, copy=True)),
            "target": torch.from_numpy(np.array(self.target[idx], dtype=np.float32, copy=True)),
            "timestamp": torch.tensor(int(self.window_start_times[idx]), dtype=torch.int64),
        }


def make_dataloader(root: str | Path = _DEFAULT_ROOT,
                    split: Split = "train",
                    batch_size: int = 16,
                    shuffle: bool | None = None,
                    num_workers: int = 0,
                    pin_memory: bool = False) -> DataLoader:
    """Create a DataLoader with train shuffling enabled by default."""
    dataset = DTECWindowDataset(root=root, split=split)
    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
