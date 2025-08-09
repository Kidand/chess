#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-play to generate training data using MCTS + current network.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch
from tqdm import tqdm

from backend.xiangqi import to_fen, other, parse_fen, generate_legal_moves
from backend.encoding import board_to_planes, move_to_index
from .mcts import MCTS, MCTSConfig
from .eval_pool import BatchedEvaluator


def play_one_game(
    net,
    device: torch.device,
    temperature: float = 1.0,
    max_moves: int = 512,
    mcts_sims: int = 800,
    mcts_batch: int = 64,
    evaluator: BatchedEvaluator | None = None,
    log_steps: bool = False,
    log_result: bool = True,
    log_fn = None,
    game_tag: str = "",
    progress: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Return list of (planes, pi, z) for each position.
    z is final outcome from the perspective of the side to move at that state.
    """
    START_FEN = "rheakaehr/9/1c5c1/s1s1s1s1s/9/9/S1S1S1S1S/1C5C1/9/RHEAKAEHR r"
    b, side = parse_fen(START_FEN)
    fen = START_FEN
    data: List[Tuple[np.ndarray, np.ndarray, float]] = []
    local_eval = evaluator is None
    evaluator = evaluator or BatchedEvaluator(net, device, max_batch=mcts_batch)
    mcts = MCTS(net, device, MCTSConfig(num_simulations=mcts_sims, batch_size=mcts_batch), evaluator=evaluator)
    history = []

    bar = tqdm(total=max_moves, desc='SP Game', leave=False, dynamic_ncols=True) if progress else None
    for step_idx in range(max_moves):
        if log_steps and log_fn is not None:
            mover = 'R' if side == 'r' else 'B'
            log_fn(f"[SP] {game_tag} step {step_idx+1}/{max_moves} mover={mover}")
        planes = board_to_planes(b, side)
        visits, action = mcts.run(fen, temperature=temperature)
        pi = visits / (visits.sum() + 1e-8)
        data.append((planes, pi, 0.0))
        # decode action to move
        frfc = action // 90; trtc = action % 90
        fr, fc = divmod(frfc, 9)
        tr, tc = divmod(trtc, 9)
        # check legality; if illegal due to exploration noise, pick first legal
        legal = generate_legal_moves(b, side)
        mv = None
        for m in legal:
            if m.from_row == fr and m.from_col == fc and m.to_row == tr and m.to_col == tc:
                mv = m
                break
        if mv is None:
            if not legal:
                # terminal: side has no moves -> side loses
                z = -1.0
                for i in range(len(data)):
                    # perspective flip: alternating sides
                    data[i] = (data[i][0], data[i][1], z if i % 2 == 0 else -z)
                if log_result and log_fn is not None:
                    winner = 'R' if other(side) == 'r' else 'B'
                    log_fn(f"[SP] {game_tag} result: winner={winner} by no-legal-move at step {step_idx+1}")
                if bar: bar.close()
                if local_eval:
                    evaluator.close()
                return data
            mv = legal[0]
            fr, fc, tr, tc = mv.from_row, mv.from_col, mv.to_row, mv.to_col
        # apply move
        b[tr][tc] = b[fr][fc]
        b[fr][fc] = '.'
        side = other(side)
        fen = to_fen(b, side)
        history.append((fr, fc, tr, tc))
        if bar: bar.update(1)
        # quick terminal check
        if not generate_legal_moves(b, side):
            # current side to move has no legal move => loses
            z = 1.0
            for i in range(len(data)):
                data[i] = (data[i][0], data[i][1], z if i % 2 == 0 else -z)
            if log_result and log_fn is not None:
                winner = 'R' if other(side) == 'r' else 'B'
                log_fn(f"[SP] {game_tag} result: winner={winner} by no-legal-move at step {step_idx+1}")
            if bar: bar.close()
            if local_eval:
                evaluator.close()
            return data

    if bar: bar.close()
    if local_eval:
        evaluator.close()
    # draw by move limit
    for i in range(len(data)):
        data[i] = (data[i][0], data[i][1], 0.0)
    if log_result and log_fn is not None:
        log_fn(f"[SP] {game_tag} result: draw by max-moves ({max_moves})")
    return data


