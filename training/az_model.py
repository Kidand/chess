#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaZero-style policy-value network for Xiangqi.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class XQAZNet(nn.Module):
    def __init__(self, channels: int = 256, blocks: int = 12, policy_head: str = "flat"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(15, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        # policy
        self.policy_head_type = policy_head
        if policy_head == "flat":
            self.p_head = nn.Sequential(
                nn.Conv2d(channels, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.p_fc = nn.Linear(64 * 10 * 9, 8100)
        elif policy_head == "structured":
            from backend.policy_planes import num_policy_planes
            k = num_policy_planes()
            self.p_conv = nn.Conv2d(channels, k, 1)
        else:
            raise ValueError(f"unknown policy_head: {policy_head}")
        # value
        self.v_head = nn.Sequential(
            nn.Conv2d(channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.v_fc1 = nn.Linear(64 * 10 * 9, 256)
        self.v_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)
        if self.policy_head_type == "flat":
            p = self.p_head(x)
            p = p.view(p.size(0), -1)
            p = self.p_fc(p)
        else:
            # structured (B, K, H, W) -> flatten to (B, K*H*W)
            p = self.p_conv(x)
            p = p.view(p.size(0), -1)
        v = self.v_head(x)
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)
        return p, v


