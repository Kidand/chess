#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple left-right augmentation for Xiangqi AlphaZero samples.

English comments per user preference.
"""

from __future__ import annotations

import numpy as np
from backend.encoding import index_to_move, move_to_index


def flip_planes_lr(planes: np.ndarray) -> np.ndarray:
    # planes: (15,10,9) -> flip horizontally (width axis)
    return planes[:, :, ::-1].copy()


def flip_policy_lr(pi: np.ndarray) -> np.ndarray:
    # pi: (8100,) dense array over from-to squares. Map (fr,fc,tr,tc)->(fr,8-fc,tr,8-tc)
    out = np.zeros_like(pi)
    # iterate over non-trivial mass to avoid O(8100)
    idxs = np.where(pi > 1e-12)[0]
    for idx in idxs:
        fr, fc, tr, tc = index_to_move(int(idx))
        fc2 = 8 - fc
        tc2 = 8 - tc
        j = move_to_index(fr, fc2, tr, tc2)
        out[j] += pi[idx]
    # numerical stability
    s = out.sum()
    if s > 0:
        out /= s
    return out


