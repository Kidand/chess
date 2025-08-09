#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-play loop using AZMCTS.
"""

from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np
import torch

from backend.xiangqi import parse_fen, other, to_fen, generate_legal_moves, is_in_check
from backend.encoding import board_to_planes
from .az_mcts import AZMCTS, MCTSConfig
from .az_replay import Sample


START_FEN = "rheakaehr/9/1c5c1/s1s1s1s1s/9/9/S1S1S1S1S/1C5C1/9/RHEAKAEHR r"


def play_one_game(
    net,
    device: torch.device,
    cfg: MCTSConfig,
    temperature_moves: int,
    no_capture_draw_plies: int,
    resign_threshold: float = -0.95,
    resign_consec: int = 3,
    enable_repetition_draw: bool = True,
) -> Tuple[List[Sample], float, Dict[str, float]]:
    b, side = parse_fen(START_FEN)
    fen = START_FEN
    data: List[Sample] = []
    mcts = AZMCTS(net, device, cfg)
    no_cap = 0
    caps = 0
    rep_count: Dict[str, int] = {}
    red_bad = 0
    black_bad = 0

    for ply in range(512):
        planes = board_to_planes(b, side)
        temp = 1.0 if ply < temperature_moves else 0.0
        visits, action, v0 = mcts.run(fen, temperature=temp)
        pi = visits / (visits.sum() + 1e-12)
        data.append(Sample(planes=planes, policy=pi, value=0.0))
        # repetition tracking (position+side)
        if enable_repetition_draw:
            rep_count[fen] = rep_count.get(fen, 0) + 1
            if rep_count[fen] >= 3:
                return data, 0.0, {"plies": float(ply+1), "caps": float(caps), "result": 0.0, "reason": 4.0}
        # resign check (value from side-to-move perspective)
        if v0 <= resign_threshold:
            if side == 'r':
                red_bad += 1
                black_bad = 0
                if red_bad >= resign_consec:
                    return data, -1.0, {"plies": float(ply+1), "caps": float(caps), "result": -1.0, "reason": 5.0}
            else:
                black_bad += 1
                red_bad = 0
                if black_bad >= resign_consec:
                    return data, -1.0, {"plies": float(ply+1), "caps": float(caps), "result": -1.0, "reason": 5.0}
        else:
            red_bad = 0 if side == 'r' else red_bad
            black_bad = 0 if side == 'b' else black_bad
        frfc = action // 90; trtc = action % 90
        fr, fc = divmod(frfc, 9)
        tr, tc = divmod(trtc, 9)
        legals = generate_legal_moves(b, side)
        mv = None
        for m in legals:
            if m.from_row == fr and m.from_col == fc and m.to_row == tr and m.to_col == tc:
                mv = m; break
        if mv is None:
            if not legals:
                z = -1.0
                return data, z
            mv = legals[0]
            fr, fc, tr, tc = mv.from_row, mv.from_col, mv.to_row, mv.to_col
        cap = 1 if b[tr][tc] != '.' else 0
        b[tr][tc] = b[fr][fc]
        b[fr][fc] = '.'
        if cap:
            caps += 1
        side = other(side)
        fen = to_fen(b, side)
        if cap: no_cap = 0
        else:
            no_cap += 1
            if no_cap >= no_capture_draw_plies:
                return data, 0.0, {"plies": float(ply+1), "caps": float(caps), "result": 0.0, "reason": 1.0}
        # terminal check
        if not generate_legal_moves(b, side):
            return data, 1.0, {"plies": float(ply+1), "caps": float(caps), "result": 1.0, "reason": 2.0}
    return data, 0.0, {"plies": float(512), "caps": float(caps), "result": 0.0, "reason": 3.0}


