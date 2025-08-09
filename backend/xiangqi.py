#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Xiangqi (Chinese Chess) core rules shared by backend and training.

English comments per user preference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

Board = List[List[str]]  # 10x9


@dataclass(frozen=True)
class Move:
    from_row: int
    from_col: int
    to_row: int
    to_col: int

    def to_dict(self) -> Dict[str, int]:
        return {"fr": self.from_row, "fc": self.from_col, "tr": self.to_row, "tc": self.to_col}


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


def parse_fen(fen: str) -> tuple[Board, str]:
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


def to_fen(b: Board, stm: str) -> str:
    rows = []
    for r in range(10):
        line = ""
        empty = 0
        for c in range(9):
            p = b[r][c]
            if p == ".":
                empty += 1
            else:
                if empty:
                    line += str(empty)
                    empty = 0
                line += p
        if empty:
            line += str(empty)
        rows.append(line)
    return f"{'/'.join(rows)} {stm}"


def apply_move(b: Board, m: Move) -> str:
    cap = b[m.to_row][m.to_col]
    b[m.to_row][m.to_col] = b[m.from_row][m.from_col]
    b[m.from_row][m.from_col] = "."
    return cap


def undo_apply_move(b: Board, m: Move, cap: str) -> None:
    b[m.from_row][m.from_col] = b[m.to_row][m.to_col]
    b[m.to_row][m.to_col] = cap


def find_king(b: Board, color: str) -> Optional[tuple[int, int]]:
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


def gen_piece_moves(b: Board, r: int, c: int, p: str) -> list[Move]:
    color = piece_color(p)
    assert color is not None
    res: list[Move] = []
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


def generate_legal_moves(b: Board, color: str) -> list[Move]:
    moves: list[Move] = []
    for r in range(10):
        for c in range(9):
            p = b[r][c]
            if p == ".":
                continue
            if piece_color(p) != color:
                continue
            moves.extend(gen_piece_moves(b, r, c, p))
    legal: list[Move] = []
    for m in moves:
        cap = apply_move(b, m)
        if not is_in_check(b, color):
            legal.append(m)
        undo_apply_move(b, m, cap)
    return [m for m in legal if not generals_face_after(b, m)]


def is_terminal(b: Board, to_move: str) -> bool:
    # No legal moves means loss for the side to move (checkmate or stalemate in Xiangqi)
    return len(generate_legal_moves(b, to_move)) == 0


