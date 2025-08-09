#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaZero-style network for Xiangqi.

- Input: 15 planes (7 red + 7 black + side)
- Policy: 8100 logits (from-square to to-square)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class XQAlphaZeroNet(nn.Module):
    def __init__(self, channels: int = 256, num_blocks: int = 12):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(15, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        # policy head
        self.p_head = nn.Sequential(
            nn.Conv2d(channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.p_fc = nn.Linear(64 * 10 * 9, 8100)

        # value head
        self.v_head = nn.Sequential(
            nn.Conv2d(channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.v_fc1 = nn.Linear(64 * 10 * 9, 256)
        self.v_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.blocks(x)
        p = self.p_head(x)
        p = p.view(p.size(0), -1)
        p = self.p_fc(p)
        v = self.v_head(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        return p, v.squeeze(-1)


