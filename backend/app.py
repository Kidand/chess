#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flask backend for Chinese Chess AI.

English comments per user preference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
# Robust import of neural engine in both run modes
try:
    from backend.nn_engine import best_move_nn, load_model as nn_load_model  # absolute package import
except Exception as _e1:
    try:
        from .nn_engine import best_move_nn, load_model as nn_load_model      # relative when run as module
    except Exception as _e2:
        try:
            import nn_engine as _nne                                         # script run from backend/ folder
            best_move_nn = _nne.best_move_nn
            nn_load_model = _nne.load_model
        except Exception as _e3:
            print("nn_engine import error:", _e1, "|", _e2, "|", _e3)
            best_move_nn = None
            nn_load_model = None


app = Flask(__name__)
CORS(app)
# ======= Frontend static (serve index.html and app.js) =======
FRONTEND_DIR = (Path(__file__).resolve().parents[1] / 'frontend').resolve()


@app.route('/')
def index():
    # Serve the frontend so that fetch uses same-origin
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/app.js')
def frontend_js():
    return send_from_directory(FRONTEND_DIR, 'app.js')


@app.route('/load-model', methods=['POST'])
def load_model_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    model_path = data.get('model_path')
    # Try to load once and report
    try:
        if nn_load_model is None:
            raise RuntimeError("nn engine unavailable")
        net = nn_load_model(model_path)
        import torch
        dev = next(net.parameters()).device
        arch = type(net).__name__
        return jsonify({"ok": True, "device": str(dev), "arch": arch})
    except Exception as e:
        import traceback, os
        exists = bool(model_path) and os.path.isfile(model_path)
        print("/load-model error:", e, "exists=", exists, "path=", model_path)
        traceback.print_exc()
        return jsonify({"error": str(e), "exists": exists, "path": model_path}), 500


# ======= Data structures =======

Board = List[List[str]]  # 10x9 board, '.' for empty, letters for pieces


@dataclass
class Move:
    from_row: int
    from_col: int
    to_row: int
    to_col: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "fr": self.from_row,
            "fc": self.from_col,
            "tr": self.to_row,
            "tc": self.to_col,
        }


# ======= Utils =======

def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 10 and 0 <= c < 9


def is_red(p: str) -> bool:
    return bool(p) and p != "." and p.isupper()


def piece_color(p: str) -> Optional[str]:
    if p == "." or not p:
        return None
    return "r" if is_red(p) else "b"


def other(color: str) -> str:
    return "b" if color == "r" else "r"


TEXT = {
    "K": "帅",
    "A": "仕",
    "E": "相",
    "H": "马",
    "R": "车",
    "C": "炮",
    "S": "兵",
    "k": "将",
    "a": "士",
    "e": "象",
    "h": "马",
    "r": "车",
    "c": "炮",
    "s": "卒",
}


def parse_fen(fen: str) -> Tuple[Board, str]:
    rows, stm = fen.strip().split(" ")
    lines = rows.split("/")
    b: Board = [["."] * 9 for _ in range(10)]
    for r in range(10):
        c = 0
        for ch in lines[r]:
            if ch.isdigit():
                c += int(ch)
            else:
                b[r][c] = ch
                c += 1
    side = "b" if stm == "b" else "r"
    return b, side


def apply_move(b: Board, m: Move) -> str:
    cap = b[m.to_row][m.to_col]
    b[m.to_row][m.to_col] = b[m.from_row][m.from_col]
    b[m.from_row][m.from_col] = "."
    return cap


def undo_apply_move(b: Board, m: Move, cap: str) -> None:
    b[m.from_row][m.from_col] = b[m.to_row][m.to_col]
    b[m.to_row][m.to_col] = cap


def find_king(b: Board, color: str) -> Optional[Tuple[int, int]]:
    target = "K" if color == "r" else "k"
    for r in range(10):
        for c in range(9):
            if b[r][c] == target:
                return r, c
    return None


def is_in_check(b: Board, color: str) -> bool:
    king = find_king(b, color)
    if not king:
        return True
    tr, tc = king
    enemy = other(color)
    for r in range(10):
        for c in range(9):
            p = b[r][c]
            if p == ".":
                continue
            if piece_color(p) != enemy:
                continue
            for m in gen_piece_moves(b, r, c, p):
                if m.to_row == tr and m.to_col == tc:
                    return True
    return False


def generals_face(b: Board) -> bool:
    rk = find_king(b, "r")
    bk = find_king(b, "b")
    if not rk or not bk:
        return False
    if rk[1] != bk[1]:
        return False
    c = rk[1]
    r1, r2 = sorted((rk[0], bk[0]))
    for r in range(r1 + 1, r2):
        if b[r][c] != ".":
            return False
    return True


def generals_face_after(b: Board, m: Move) -> bool:
    cap = apply_move(b, m)
    face = generals_face(b)
    undo_apply_move(b, m, cap)
    return face


def mv(fr: int, fc: int, tr: int, tc: int) -> Move:
    return Move(fr, fc, tr, tc)


def gen_piece_moves(b: Board, r: int, c: int, p: str) -> List[Move]:
    color = piece_color(p)
    assert color is not None
    res: List[Move] = []
    forward = -1 if color == "r" else 1
    palace_rows = {"r": {7, 8, 9}, "b": {0, 1, 2}}[color]

    def in_palace(rr: int, cc: int) -> bool:
        return rr in palace_rows and 3 <= cc <= 5

    t = p.lower()
    if t == "k":
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if not in_bounds(rr, cc):
                continue
            if not in_palace(rr, cc):
                continue
            if piece_color(b[rr][cc]) != color:
                res.append(mv(r, c, rr, cc))
    elif t == "a":
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            rr, cc = r + dr, c + dc
            if not in_bounds(rr, cc):
                continue
            if not in_palace(rr, cc):
                continue
            if piece_color(b[rr][cc]) != color:
                res.append(mv(r, c, rr, cc))
    elif t == "e":
        for dr, dc in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
            rr, cc = r + dr, c + dc
            er, ec = r + dr // 2, c + dc // 2
            if not in_bounds(rr, cc):
                continue
            # river rule
            if color == "r" and rr < 5:
                continue
            if color == "b" and rr > 4:
                continue
            if b[er][ec] != ".":
                continue
            if piece_color(b[rr][cc]) != color:
                res.append(mv(r, c, rr, cc))
    elif t == "h":
        legs = (
            (0, 1, -1, 2),
            (0, 1, 1, 2),
            (1, 0, 2, 1),
            (-1, 0, -2, 1),
            (0, -1, -1, -2),
            (0, -1, 1, -2),
            (1, 0, 2, -1),
            (-1, 0, -2, -1),
        )
        for lr, lc, dr, dc in legs:
            br, bc = r + lr, c + lc
            rr, cc = r + dr, c + dc
            if not in_bounds(rr, cc) or not in_bounds(br, bc):
                continue
            if b[br][bc] != ".":
                continue
            if piece_color(b[rr][cc]) != color:
                res.append(mv(r, c, rr, cc))
    elif t == "r":
        def slide(sr: int, sc: int, dr: int, dc: int) -> None:
            rr, cc = sr + dr, sc + dc
            while in_bounds(rr, cc):
                if b[rr][cc] == ".":
                    res.append(mv(sr, sc, rr, cc))
                    rr += dr
                    cc += dc
                else:
                    if piece_color(b[rr][cc]) != color:
                        res.append(mv(sr, sc, rr, cc))
                    break

        slide(r, c, 1, 0)
        slide(r, c, -1, 0)
        slide(r, c, 0, 1)
        slide(r, c, 0, -1)
    elif t == "c":
        def slide_move(sr: int, sc: int, dr: int, dc: int) -> None:
            rr, cc = sr + dr, sc + dc
            while in_bounds(rr, cc) and b[rr][cc] == ".":
                res.append(mv(sr, sc, rr, cc))
                rr += dr
                cc += dc
            # find screen
            while in_bounds(rr, cc) and b[rr][cc] == ".":
                rr += dr
                cc += dc
            if in_bounds(rr, cc):
                rr += dr
                cc += dc
                while in_bounds(rr, cc):
                    if b[rr][cc] != ".":
                        if piece_color(b[rr][cc]) != color:
                            res.append(mv(r, c, rr, cc))
                        break
                    rr += dr
                    cc += dc

        slide_move(r, c, 1, 0)
        slide_move(r, c, -1, 0)
        slide_move(r, c, 0, 1)
        slide_move(r, c, 0, -1)
    elif t == "s":
        candidates = [(forward, 0)]
        if (color == "r" and r <= 4) or (color == "b" and r >= 5):
            candidates.extend([(0, 1), (0, -1)])
        for dr, dc in candidates:
            rr, cc = r + dr, c + dc
            if not in_bounds(rr, cc):
                continue
            if piece_color(b[rr][cc]) != color:
                res.append(mv(r, c, rr, cc))
    return res


def generate_legal_moves(b: Board, color: str) -> List[Move]:
    # pseudo-legal then filter by king safety and facing
    moves: List[Move] = []
    for r in range(10):
        for c in range(9):
            p = b[r][c]
            if p == ".":
                continue
            if piece_color(p) != color:
                continue
            moves.extend(gen_piece_moves(b, r, c, p))
    legal: List[Move] = []
    for m in moves:
        cap = apply_move(b, m)
        if not is_in_check(b, color):
            legal.append(m)
        undo_apply_move(b, m, cap)
    # generals cannot face
    return [m for m in legal if not generals_face_after(b, m)]


VAL: Dict[str, int] = {
    "K": 10000,
    "A": 120,
    "E": 120,
    "H": 270,
    "R": 600,
    "C": 350,
    "S": 70,
    "k": 10000,
    "a": 120,
    "e": 120,
    "h": 270,
    "r": 600,
    "c": 350,
    "s": 70,
}


def mobility_bonus(b: Board, r: int, c: int) -> int:
    cnt = 0
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        while in_bounds(rr, cc) and b[rr][cc] == ".":
            cnt += 1
            rr += dr
            cc += dc
    return cnt * 2


def eval_board(b: Board) -> int:
    score = 0
    for r in range(10):
        for c in range(9):
            p = b[r][c]
            if p == ".":
                continue
            v = VAL.get(p, 0)
            bonus = 0
            if p == "S" and r <= 4:
                bonus = 20
            if p == "s" and r >= 5:
                bonus = 20
            if p in ("R", "r"):
                bonus = mobility_bonus(b, r, c)
            if p in ("C", "c"):
                bonus += 5
            score += (v + bonus) if is_red(p) else -(v + bonus)
    if is_in_check(b, "r"):
        score -= 30
    if is_in_check(b, "b"):
        score += 30
    return score


def order_moves(b: Board, moves: List[Move]) -> List[Move]:
    # captures first with MVV-LVA style
    def key(m: Move) -> int:
        ca = b[m.to_row][m.to_col]
        attacker = b[m.from_row][m.from_col]
        return (VAL.get(ca, 0) - VAL.get(attacker, 0))

    # Stable sort for reproducibility
    return sorted(moves, key=key, reverse=True)


def is_terminal(b: Board, to_move: str) -> bool:
    return len(generate_legal_moves(b, to_move)) == 0


def ai_best_move(b: Board, side: str, max_depth: int, time_limit_ms: int) -> Tuple[Optional[Move], int, int, int]:
    # Iterative deepening alpha-beta with quiescence
    start = time.perf_counter()

    def time_up() -> bool:
        return (time.perf_counter() - start) * 1000.0 > time_limit_ms

    me = side
    nodes = 0
    best: Optional[Move] = None
    best_score = -10**18
    reached_depth = 0

    root_moves = order_moves(b, generate_legal_moves(b, me))

    def quiescence(alpha: int, beta: int, to_move: str) -> int:
        nonlocal nodes
        nodes += 1
        stand = eval_board(b) if to_move == me else -eval_board(b)
        if stand >= beta:
            return beta
        if alpha < stand:
            alpha = stand
        cap_moves = [m for m in generate_legal_moves(b, to_move) if b[m.to_row][m.to_col] != "."]
        cap_moves = order_moves(b, cap_moves)
        for m in cap_moves:
            cap = apply_move(b, m)
            score = -quiescence(-beta, -alpha, other(to_move))
            undo_apply_move(b, m, cap)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
            if time_up():
                break
        return alpha

    def search(depth: int, alpha: int, beta: int, to_move: str) -> int:
        nonlocal nodes
        nodes += 1
        if depth == 0:
            return quiescence(alpha, beta, to_move)
        if is_terminal(b, to_move):
            return -9999 if to_move == me else 9999
        val = -10**18
        moves = order_moves(b, generate_legal_moves(b, to_move))
        for m in moves:
            cap = apply_move(b, m)
            score = -search(depth - 1, -beta, -alpha, other(to_move))
            undo_apply_move(b, m, cap)
            if score > val:
                val = score
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
            if time_up():
                break
        return val if val != -10**18 else (-9999 if to_move == me else 9999)

    for depth in range(1, max_depth + 1):
        local_best = best
        local_best_score = -10**18
        for m in root_moves:
            if time_up():
                break
            cap = apply_move(b, m)
            score = -search(depth - 1, -10**12, 10**12, other(me))
            undo_apply_move(b, m, cap)
            if score > local_best_score:
                local_best_score = score
                local_best = m
        if time_up():
            break
        best = local_best
        best_score = local_best_score
        reached_depth = depth

    return best, best_score, nodes, reached_depth


# ======= HTTP endpoints =======


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/ai-move", methods=["POST"])
def ai_move_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    fen = data.get("fen", "")
    depth = int(data.get("depth", 3))
    time_ms = int(data.get("time_ms", 1200))
    engine = data.get("engine", "ab")
    model_path = data.get("model_path")
    try:
        board, side = parse_fen(fen)
    except Exception as e:
        return jsonify({"error": f"bad fen: {e}"}), 400

    if engine == "nn":
        try:
            if best_move_nn is None:
                raise RuntimeError("nn engine unavailable")
            # log model path existence to backend stdout
            exists = False
            try:
                import os
                exists = bool(model_path) and os.path.isfile(model_path)
            except Exception:
                pass
            print(f"/ai-move nn: model_path={model_path!r} exists={exists}")
            move, meta = best_move_nn(fen, model_path)
            return jsonify({"move": move, "engine": "nn", **meta})
        except Exception as e:
            import traceback
            print("/ai-move nn error:", e)
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
    else:
        best, score, nodes, reached_depth = ai_best_move(board, side, max_depth=depth, time_limit_ms=time_ms)
        if best is None:
            return jsonify({"move": None, "score": 0, "nodes": nodes, "depth": reached_depth, "engine": "ab"})
        return jsonify({
            "move": best.to_dict(),
            "score": score,
            "nodes": nodes,
            "depth": reached_depth,
            "engine": "ab",
        })


if __name__ == "__main__":
    # Bind to 127.0.0.1:5000 and print a clickable link to open the frontend
    url = "http://127.0.0.1:5000/"
    print(f"Open this link in your browser: {url}")
    app.run(host="127.0.0.1", port=5000, debug=False)


