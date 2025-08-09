#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-play to generate training data using MCTS + current network.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch
from tqdm import tqdm

from backend.xiangqi import to_fen, other, parse_fen, generate_legal_moves, is_in_check
from backend.encoding import board_to_planes, move_to_index
from .mcts import MCTS, MCTSConfig
from .eval_pool import BatchedEvaluator


def play_one_game(
    net,
    device: torch.device,
    temperature: float = 1.0,
    temp_init: float = 1.0,
    temp_final: float = 0.0,
    temp_decay_moves: int = 16,
    max_moves: int = 512,
    mcts_sims: int = 800,
    mcts_batch: int = 64,
    evaluator: BatchedEvaluator | None = None,
    log_steps: bool = False,
    log_result: bool = True,
    log_fn = None,
    game_tag: str = "",
    progress: bool = False,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, float]], float]:
    """Return list of (planes, pi, z) for each position.
    z is final outcome from the perspective of the side to move at that state.
    """
    START_FEN = "rheakaehr/9/1c5c1/s1s1s1s1s/9/9/S1S1S1S1S/1C5C1/9/RHEAKAEHR r"
    b, side = parse_fen(START_FEN)
    fen = START_FEN
    data: List[Tuple[np.ndarray, np.ndarray, float, float]] = []  # (planes, pi, v0, reward)
    local_eval = evaluator is None
    evaluator = evaluator or BatchedEvaluator(net, device, max_batch=mcts_batch)
    mcts = MCTS(net, device, MCTSConfig(num_simulations=mcts_sims, batch_size=mcts_batch), evaluator=evaluator)
    history = []

    bar = tqdm(total=max_moves, desc='SP Game', leave=False, dynamic_ncols=True) if progress else None
    draw_patience = 8  # early-stop if no capture for many plies
    no_capture_steps = 0
    last_cap = False
    for step_idx in range(max_moves):
        if log_steps and log_fn is not None:
            mover = 'R' if side == 'r' else 'B'
            log_fn(f"[SP] {game_tag} step {step_idx+1}/{max_moves} mover={mover}")
        planes = board_to_planes(b, side)
        # temperature schedule
        if temp_decay_moves > 0:
            t = max(0.0, 1.0 - step_idx / max(1, temp_decay_moves))
            temp_now = temp_final + (temp_init - temp_final) * t
        else:
            temp_now = temperature
        visits, action, v0 = mcts.run(fen, temperature=temp_now)
        pi = visits / (visits.sum() + 1e-8)
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
                # finalize and return; we return z, and caller composes targets
                if log_result and log_fn is not None:
                    winner = 'R' if other(side) == 'r' else 'B'
                    log_fn(f"[SP] {game_tag} result: winner={winner} by no-legal-move at step {step_idx+1}")
                if bar: bar.close()
                if local_eval:
                    evaluator.close()
                return data, z
            mv = legal[0]
            fr, fc, tr, tc = mv.from_row, mv.from_col, mv.to_row, mv.to_col
        # apply move
        cap = 1.0 if b[tr][tc] != '.' else 0.0
        b[tr][tc] = b[fr][fc]
        b[fr][fc] = '.'
        # compute check reward after move (opponent to move is 'side' after flip)
        # delay side flip till after reward compute
        # we'll flip below
        # draw patience: if long no-capture, early draw to avoid endless games
        if cap > 0:
            no_capture_steps = 0
        else:
            no_capture_steps += 1
            if no_capture_steps >= draw_patience:
                if log_result and log_fn is not None:
                    log_fn(f"[SP] {game_tag} result: early-draw by no-capture ({no_capture_steps})")
                if bar: bar.close()
                if local_eval:
                    evaluator.close()
                z = 0.0
                return data, z
        # opponent side to move
        side = other(side)
        # check reward: is opponent in check?
        chk = 1.0 if is_in_check(b, side) else 0.0
        reward = cap + 0.5 * chk  # weights will be applied in caller
        # append record after getting reward and v0
        data.append((planes, pi, float(v0), float(reward)))
        fen = to_fen(b, side)
        history.append((fr, fc, tr, tc))
        if bar: bar.update(1)
        # quick terminal check
        if not generate_legal_moves(b, side):
            # current side to move has no legal move => loses
            z = 1.0
            if log_result and log_fn is not None:
                winner = 'R' if other(side) == 'r' else 'B'
                log_fn(f"[SP] {game_tag} result: winner={winner} by no-legal-move at step {step_idx+1}")
            if bar: bar.close()
            if local_eval:
                evaluator.close()
            return data, z

    if bar: bar.close()
    if local_eval:
        evaluator.close()
    # draw by move limit
    z = 0.0
    if log_result and log_fn is not None:
        log_fn(f"[SP] {game_tag} result: draw by max-moves ({max_moves})")
    return data, z


