#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured policy head for Xiangqi: mapping moves to (plane, from-square).

Each plane encodes a semantic action type relative to the side to move.
The final policy head has K planes, each of size (10, 9). The logit for a
specific move is read at [plane_id, from_row, from_col].

Plane layout (K = 99):
- Rook slides (R):           4 dirs × 9 steps = 36   (offset 0)
- Cannon slides (C move):    4 dirs × 9 steps = 36   (offset 36)
- Cannon hits (C capture):   4 dirs × 1       = 4    (offset 72)
- Knight (H):                8 L-moves        = 8    (offset 76)
- Advisor (A):               4 diagonal steps = 4    (offset 84)
- Elephant (E):              4 big diagonals  = 4    (offset 88)
- King (K):                  4 orth steps     = 4    (offset 92)
- Soldier (S):               3 (forward/left/right by side) = 3 (offset 96)

Index mapping: global_index = plane_id * 90 + (from_row * 9 + from_col)

English comments per user preference.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .encoding import square_index


POLICY_PLANES_K = 99

# Offsets
OFF_R_SLIDE = 0
OFF_C_SLIDE = OFF_R_SLIDE + 36
OFF_C_HIT = OFF_C_SLIDE + 36
OFF_H_KNIGHT = OFF_C_HIT + 4
OFF_A_ADVISOR = OFF_H_KNIGHT + 8
OFF_E_ELEPHANT = OFF_A_ADVISOR + 4
OFF_K_KING = OFF_E_ELEPHANT + 4
OFF_S_SOLDIER = OFF_K_KING + 4


def num_policy_planes() -> int:
    return POLICY_PLANES_K


def _dir_and_steps(fr: int, fc: int, tr: int, tc: int) -> Optional[Tuple[int, int]]:
    dr = tr - fr
    dc = tc - fc
    if dc == 0 and dr != 0:
        return (0 if dr < 0 else 1), abs(dr)  # 0: up, 1: down
    if dr == 0 and dc != 0:
        return (2 if dc < 0 else 3), abs(dc)  # 2: left, 3: right
    return None


def _knight_move_index(dr: int, dc: int) -> Optional[int]:
    # 8 L moves
    mapping = {
        (-2, -1): 0, (-2, 1): 1,
        (-1, -2): 2, (-1, 2): 3,
        (1, -2): 4,  (1, 2): 5,
        (2, -1): 6,  (2, 1): 7,
    }
    return mapping.get((dr, dc))


def _advisor_move_index(dr: int, dc: int) -> Optional[int]:
    mapping = {(-1, -1): 0, (-1, 1): 1, (1, -1): 2, (1, 1): 3}
    return mapping.get((dr, dc))


def _elephant_move_index(dr: int, dc: int) -> Optional[int]:
    mapping = {(-2, -2): 0, (-2, 2): 1, (2, -2): 2, (2, 2): 3}
    return mapping.get((dr, dc))


def _king_move_index(dr: int, dc: int) -> Optional[int]:
    mapping = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}
    return mapping.get((dr, dc))


def _soldier_move_index(side: str, dr: int, dc: int) -> Optional[int]:
    # forward/left/right relative to side
    if side == 'r':
        mapping = {(-1, 0): 0, (0, -1): 1, (0, 1): 2}
    else:
        mapping = {(1, 0): 0, (0, 1): 1, (0, -1): 2}
    return mapping.get((dr, dc))


def map_move_to_plane_id(board, side: str, fr: int, fc: int, tr: int, tc: int) -> Optional[int]:
    """Return plane_id (0..K-1) for a legal move on the given board.

    This function assumes the move is legal; it decides semantic plane by piece type
    at (fr,fc) and relative displacement.
    """
    p = board[fr][fc]
    if not p or p == '.':
        return None
    is_red_piece = p.isupper()
    # Determine piece type (lowercase)
    t = p.lower()
    dr = tr - fr
    dc = tc - fc

    if t == 'r':  # Rook
        res = _dir_and_steps(fr, fc, tr, tc)
        if res is None:
            return None
        direction, steps = res
        if steps <= 0 or steps > 9:
            return None
        return OFF_R_SLIDE + direction * 9 + (steps - 1)

    if t == 'c':  # Cannon
        res = _dir_and_steps(fr, fc, tr, tc)
        if res is None:
            return None
        direction, steps = res
        # detect capture vs move by destination occupancy
        dest = board[tr][tc]
        if dest == '.':
            if steps <= 0 or steps > 9:
                return None
            return OFF_C_SLIDE + direction * 9 + (steps - 1)
        else:
            # capture: should be exactly one screen between
            return OFF_C_HIT + direction

    if t == 'h':  # Knight
        idx = _knight_move_index(dr, dc)
        if idx is None:
            return None
        return OFF_H_KNIGHT + idx

    if t == 'a':  # Advisor
        idx = _advisor_move_index(dr, dc)
        if idx is None:
            return None
        return OFF_A_ADVISOR + idx

    if t == 'e':  # Elephant
        idx = _elephant_move_index(dr, dc)
        if idx is None:
            return None
        return OFF_E_ELEPHANT + idx

    if t == 'k':  # King
        idx = _king_move_index(dr, dc)
        if idx is None:
            return None
        return OFF_K_KING + idx

    if t == 's':  # Soldier
        idx = _soldier_move_index(side, dr, dc)
        if idx is None:
            return None
        return OFF_S_SOLDIER + idx

    return None


def policy_index_from_move(plane_id: int, fr: int, fc: int) -> int:
    return plane_id * 90 + square_index(fr, fc)


