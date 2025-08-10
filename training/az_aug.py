#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data augmentation utilities for AlphaZero training.

English comments per user preference.
"""

from __future__ import annotations

import numpy as np

# Reuse encoding constants for shapes and indexing
from backend.encoding import BOARD_H, BOARD_W, NUM_SQUARES


def flip_planes_lr(planes: np.ndarray) -> np.ndarray:
    """Flip input planes (C,H,W) horizontally (left-right).

    The side-to-move plane is also flipped geometrically; semantics remain unchanged.
    """
    assert planes.ndim == 3 and planes.shape[1] == BOARD_H and planes.shape[2] == BOARD_W
    return planes[:, :, ::-1].copy()


def _square_lr_permutation() -> np.ndarray:
    """Return a permutation array M of length 90 where M[i] is the index of the
    horizontally mirrored square for square index i.
    """
    grid = np.arange(NUM_SQUARES, dtype=np.int32).reshape(BOARD_H, BOARD_W)
    mirrored = grid[:, ::-1].reshape(-1)
    return mirrored


def flip_policy_lr(policy: np.ndarray) -> np.ndarray:
    """Flip policy vector (8100,) horizontally.

    Mapping: move (fr,fc)->(tr,tc) becomes (fr, 8-fc)->(tr, 8-tc).
    """
    assert policy.ndim == 1 and policy.size == NUM_SQUARES * NUM_SQUARES
    M = _square_lr_permutation()
    idxs = np.arange(policy.size, dtype=np.int32)
    fr = idxs // NUM_SQUARES
    to = idxs % NUM_SQUARES
    new_idxs = (M[fr] * NUM_SQUARES + M[to]).astype(np.int32)
    out = np.zeros_like(policy)
    out[new_idxs] = policy
    return out

