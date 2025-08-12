#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay buffer for AlphaZero Xiangqi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Deque
from collections import deque
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    planes: np.ndarray  # (15,10,9)
    policy: np.ndarray  # (8100,)
    value: float


class ReplayBuffer(Dataset):
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buf: Deque[Sample] = deque(maxlen=capacity)

    def push_game(self, game_data: List[Sample]):
        for s in game_data:
            self.buf.append(s)

    def __len__(self) -> int:
        return len(self.buf)

    def __getitem__(self, idx: int):
        s = self.buf[idx]
        x = torch.from_numpy(s.planes).float()
        p = torch.from_numpy(s.policy).float()
        v = torch.tensor([s.value], dtype=torch.float32)
        return x, p, v


