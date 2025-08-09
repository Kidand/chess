#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay buffer dataset for AlphaZero training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    planes: np.ndarray  # (15,10,9)
    policy: np.ndarray  # (8100,)
    value: float        # scalar


class ReplayDataset(Dataset):
    def __init__(self, samples: Sequence[Sample]):
        # copy to list to avoid external mutation during training
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = torch.from_numpy(s.planes).float()
        p = torch.from_numpy(s.policy).float()
        v = torch.tensor([s.value], dtype=torch.float32)
        return x, p, v


