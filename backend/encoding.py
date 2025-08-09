#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Input/Output encoding for Xiangqi AlphaZero-style model.

English comments per user preference.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


Board = List[List[str]]  # 10x9

PIECE_TYPES = ["K", "A", "E", "H", "R", "C", "S"]  # 7 types
NUM_CHANNELS = 14 + 1  # 7 per color + side-to-move plane
BOARD_H, BOARD_W = 10, 9
NUM_SQUARES = BOARD_H * BOARD_W  # 90
POLICY_SIZE = NUM_SQUARES * NUM_SQUARES  # from x to mapping, 8100 logits


def square_index(r: int, c: int) -> int:
    return r * BOARD_W + c


def move_to_index(fr: int, fc: int, tr: int, tc: int) -> int:
    return square_index(fr, fc) * NUM_SQUARES + square_index(tr, tc)


def index_to_move(idx: int) -> Tuple[int, int, int, int]:
    frfc, trtc = divmod(idx, NUM_SQUARES)
    fr, fc = divmod(frfc, BOARD_W)
    tr, tc = divmod(trtc, BOARD_W)
    return fr, fc, tr, tc


def board_to_planes(b: Board, side: str) -> np.ndarray:
    """Return (C,H,W) float32 planes. Channels: 7 red, 7 black, 1 side.
    Red pieces are uppercase, black lowercase.
    """
    planes = np.zeros((NUM_CHANNELS, BOARD_H, BOARD_W), dtype=np.float32)
    # 0..6 red K,A,E,H,R,C,S; 7..13 black k,a,e,h,r,c,s
    red_offset = 0
    black_offset = 7
    piece_to_channel = {t: i for i, t in enumerate(PIECE_TYPES)}
    for r in range(BOARD_H):
        for c in range(BOARD_W):
            p = b[r][c]
            if p == ".":
                continue
            if p.isupper():
                ch = piece_to_channel[p]
                planes[red_offset + ch, r, c] = 1.0
            else:
                ch = piece_to_channel[p.upper()]
                planes[black_offset + ch, r, c] = 1.0
    # side-to-move plane as ones for red, zeros for black
    if side == "r":
        planes[14, :, :] = 1.0
    else:
        planes[14, :, :] = 0.0
    return planes


