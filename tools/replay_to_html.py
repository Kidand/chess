#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay generator: read one game JSON (as saved by training), reconstruct the game
using backend move rules, and emit a self-contained HTML to replay the game
step-by-step (1 second per move).

Usage:
  python tools/replay_to_html.py --input path/to/game.json --output frontend/replay.html

You may also pipe a JSON object via STDIN:
  cat game.json | python tools/replay_to_html.py --output frontend/replay.html

English comments per user preference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.xiangqi import parse_fen, to_fen, generate_legal_moves, apply_move, other
from backend.encoding import index_to_move


def read_game_json(input_arg: str | None) -> Dict[str, Any]:
    if input_arg and input_arg != "-":
        text = Path(input_arg).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    text = text.strip()
    # support JSONL with a single line
    if "\n" in text:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise SystemExit("No valid JSON object found in input")
    return json.loads(text)


def reconstruct_states(game: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, int]]]:
    start_fen: str = game["start_fen"]
    moves: List[int] = list(game.get("moves", []))
    b, side = parse_fen(start_fen)
    states: List[str] = [start_fen]
    applied: List[Dict[str, int]] = []
    for idx in moves:
        fr, fc, tr, tc = index_to_move(int(idx))
        legals = generate_legal_moves(b, side)
        mv = None
        for m in legals:
            if m.from_row == fr and m.from_col == fc and m.to_row == tr and m.to_col == tc:
                mv = m
                break
        fallback = False
        if mv is None:
            if not legals:
                break
            mv = legals[0]
            fallback = True
            fr, fc, tr, tc = mv.from_row, mv.from_col, mv.to_row, mv.to_col
        apply_move(b, mv)
        side = other(side)
        states.append(to_fen(b, side))
        applied.append({
            "fr": fr, "fc": fc, "tr": tr, "tc": tc, "fallback": 1 if fallback else 0,
        })
    return states, applied


def emit_html(game: Dict[str, Any], states: List[str], applied: List[Dict[str, int]]) -> str:
    # Minimal self-contained HTML player with 1s interval per move
    # Embed data directly to avoid external dependencies.
    data_js = json.dumps({
        "meta": {
            "timestamp": game.get("timestamp"),
            "seg": game.get("seg"),
            "rank": game.get("rank"),
            "idx": game.get("idx"),
            "result": game.get("result"),
            "winner": game.get("winner"),
            "reason": game.get("reason"),
            "plies": game.get("plies"),
            "caps": game.get("caps"),
        },
        "states": states,
        "moves": applied,
    }, ensure_ascii=False)

    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>中国象棋 · 回放</title>
  <style>
    :root {{ --bg:#f7f3e8; --line:#7b5b34; --red:#c03a2b; --black:#2c3e50; --hilite:#ffd54f; --sel:#4fc3f7; }}
    html,body{{ height:100%; margin:0; }}
    body{{ display:flex; flex-direction:column; align-items:center; gap:10px; background:var(--bg); font-family:system-ui,-apple-system,Segoe UI,Roboto; }}
    h1{{ font-size:20px; margin:10px 0 0; color:#5b4632; }}
    .wrap{{ display:flex; gap:16px; flex-wrap:wrap; justify-content:center; align-items:flex-start; margin: 0 12px 20px; }}
    .panel{{ min-width:300px; background:#fff; border-radius:12px; padding:12px; box-shadow:0 8px 22px rgba(0,0,0,.12); }}
    .row{{ display:flex; align-items:center; justify-content:space-between; gap:8px; margin:8px 0; }}
    label{{ font-size:14px; color:#3d2f22; }}
    select,button,input{{ font-size:14px; padding:8px 10px; border-radius:8px; border:1px solid #d7c9b8; background:#fff; }}
    button{{ cursor:pointer; }}
    .badge{{ padding:2px 6px; border-radius:6px; font-size:12px; background:#eee; }}
    .hint{{ color:#6b543e; font-size:12px; }}
    .footer{{ color:#7a6a56; font-size:12px; margin-bottom:10px; }}

    /* Board grid */
    .board{{ display:grid; grid-template-columns: repeat(9, 56px); grid-template-rows: repeat(10, 56px); gap:2px; background:#d9c4a6; padding:8px; border-radius:8px; box-shadow:0 10px 30px rgba(0,0,0,.15); }}
    .sq{{ width:56px; height:56px; display:flex; align-items:center; justify-content:center; background:#f7ebd9; border-radius:6px; position:relative; font-weight:700; font-size:24px; color:#2c3e50; }}
    .sq:nth-child(odd){{ background:#fff4e4; }}
    .piece.red{{ color: var(--red); }}
    .piece.black{{ color: var(--black); }}
    .hl::after{{ content:''; position:absolute; inset:2px; border:2px solid var(--sel); border-radius:6px; pointer-events:none; }}
    .mono{{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
  </style>
</head>
<body>
  <h1>中国象棋 · 回放</h1>
  <div class=\"wrap\">
    <canvas id=\"board\" width=\"540\" height=\"600\"></canvas>
    <div class=\"panel\">
      <div class=\"row\"><label>结果：</label>
        <span><span id=\"result\" class=\"badge\"></span> 原因 <span id=\"reason\" class=\"badge\"></span></span>
      </div>
      <div class=\"row\"><label>对局信息：</label>
        <span class=\"mono\">Seg <span id=\"seg\"></span> · Rank <span id=\"rank\"></span> · Idx <span id=\"idx\"></span></span>
      </div>
      <div class=\"row\"><label>步数 / 吃子：</label>
        <span class=\"mono\"><span id=\"plies\"></span> / <span id=\"caps\"></span></span>
      </div>
      <div class=\"row\"><label>时间：</label>
        <span id=\"timestamp\" class=\"mono\"></span>
      </div>
      <div class=\"row\" style=\"justify-content:flex-start; gap:8px;\">
        <button id=\"btnStart\">开始</button>
        <button id=\"btnPause\">暂停</button>
        <button id=\"btnPrev\">上一步</button>
        <button id=\"btnNext\">下一步</button>
      </div>
      <div class=\"row\"><label>进度：</label>
        <span class=\"mono\">第 <span id=\"step\">0</span> 步 / 共 <span id=\"total\"></span> 步</span>
      </div>
      <div class=\"hint\">提示：点击“开始”后每 1 秒自动前进一步，可随时“暂停”。</div>
    </div>
  </div>

  <script>
  const payload = {data_js};
  const states = payload.states || [];
  const moves = payload.moves || [];
  const meta = payload.meta || {{}};

  // UI meta
  for (const [k,v] of Object.entries(meta)) {{
    const el = document.getElementById(k);
    if (el) el.textContent = String(v ?? '');
  }}
  // Pretty result text: 红胜/黑胜/和棋
  function prettyResult(meta) {{
    if (meta && typeof meta === 'object') {{
      if (meta.winner === 'r') return '红胜';
      if (meta.winner === 'b') return '黑胜';
      if (meta.result === 0) return '和棋';
      if (meta.result === 1) return '胜';
      if (meta.result === -1) return '负';
    }}
    return '';
  }}
  const resEl = document.getElementById('result');
  if (resEl) {{
    const txt = prettyResult(meta);
    if (txt) resEl.textContent = txt;
  }}
  document.getElementById('total').textContent = String(Math.max(0, states.length-1));

  // Mapping letters to Chinese pieces (same as frontend)
  const TEXT = {{K:'帅',A:'仕',E:'相',H:'马',R:'车',C:'炮',S:'兵',k:'将',a:'士',e:'象',h:'马',r:'车',c:'炮',s:'卒'}};

  const ctx = document.getElementById('board').getContext('2d');
  const GRID_W = 60, GRID_H = 60, ORIGIN_X = 30, ORIGIN_Y = 30;
  if (!CanvasRenderingContext2D.prototype.roundRect) {{
    CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {{
      const min = Math.min(w, h) / 2; r = Math.min(r, min);
      this.beginPath(); this.moveTo(x + r, y);
      this.arcTo(x + w, y, x + w, y + h, r);
      this.arcTo(x + w, y + h, x, y + h, r);
      this.arcTo(x, y + h, x, y, r);
      this.arcTo(x, y, x + w, y, r); this.closePath(); return this;
    }}
  }}

  function parseFen(fen) {{
    const [rows, stm] = fen.trim().split(' ');
    const lines = rows.split('/');
    const b = Array.from({{length:10}},()=>Array(9).fill('.'));
    for(let r=0;r<10;r++){{
      let c=0; for(const ch of lines[r]){{
        if(/[1-9]/.test(ch)){{ c += parseInt(ch,10); }}
        else {{ b[r][c++] = ch; }}
      }}
    }}
    return {{board:b, stm}};
  }}

  function drawLine(c1,r1,c2,r2){{
    ctx.beginPath();
    ctx.moveTo(ORIGIN_X + c1*GRID_W, ORIGIN_Y + r1*GRID_H);
    ctx.lineTo(ORIGIN_X + c2*GRID_W, ORIGIN_Y + r2*GRID_H);
    ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--line').trim();
    ctx.lineWidth=2; ctx.stroke();
  }}

  function drawBoard(){{
    const canvas = document.getElementById('board');
    ctx.clearRect(0,0,canvas.width,canvas.height);
    // Outer frame
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--line').trim(); ctx.lineWidth=2;
    ctx.strokeRect(ORIGIN_X-5, ORIGIN_Y-5, GRID_W*8+10, GRID_H*9+10);
    // Grid lines
    ctx.beginPath();
    for(let r=0;r<10;r++){{
      const y = ORIGIN_Y + r*GRID_H;
      ctx.moveTo(ORIGIN_X, y);
      ctx.lineTo(ORIGIN_X + GRID_W*8, y);
    }}
    // Vertical lines (break at river)
    for(let c=0;c<9;c++){{
      const x = ORIGIN_X + c*GRID_W;
      ctx.moveTo(x, ORIGIN_Y);
      ctx.lineTo(x, ORIGIN_Y + GRID_H*4);
      ctx.moveTo(x, ORIGIN_Y + GRID_H*5);
      ctx.lineTo(x, ORIGIN_Y + GRID_H*9);
    }}
    ctx.stroke();
    // Palace diagonals
    drawLine(3,0,5,2); drawLine(5,0,3,2);
    drawLine(3,7,5,9); drawLine(5,7,3,9);
    // River text
    ctx.fillStyle = '#a08866'; ctx.font = '16px serif';
    ctx.fillText('楚河', ORIGIN_X + GRID_W*1.5, ORIGIN_Y + GRID_H*4.6);
    ctx.fillText('汉界', ORIGIN_X + GRID_W*5.5, ORIGIN_Y + GRID_H*4.6);
  }}

  function rect(c,r,rad,color){{
    const x = ORIGIN_X + c*GRID_W; const y = ORIGIN_Y + r*GRID_H;
    ctx.beginPath(); ctx.roundRect(x-rad,y-rad,rad*2,rad*2,6); ctx.fillStyle=color+'55'; ctx.fill();
  }}

  function drawPiece(c,r,p){{
    const x = ORIGIN_X + c*GRID_W; const y = ORIGIN_Y + r*GRID_H;
    ctx.save();
    // Disk
    ctx.beginPath(); ctx.arc(x,y,23,0,Math.PI*2);
    ctx.fillStyle = '#fffdf7'; ctx.fill();
    ctx.strokeStyle = '#cbb79c'; ctx.lineWidth=2; ctx.stroke();
    // Text
    const red = (p===p.toUpperCase());
    const color = getComputedStyle(document.documentElement).getPropertyValue(red?'--red':'--black').trim();
    ctx.fillStyle = color; ctx.font = '20px serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(TEXT[p]||'?', x, y+1);
    ctx.restore();
  }}

  function renderBoard(fen, lastMove){{
    const {{ board }} = parseFen(fen);
    drawBoard();
    if (lastMove) {{ rect(lastMove.tc, lastMove.tr, 26, '#4fc3f7'); rect(lastMove.fc, lastMove.fr, 26, '#ffd54f'); }}
    for(let r=0;r<10;r++){{
      for(let c=0;c<9;c++){{
        const p = board[r][c]; if(p==='.') continue; drawPiece(c,r,p);
      }}
    }}
  }}

  let cur = 0; // state index
  let timer = null;
  function show(i) {{
    cur = Math.max(0, Math.min(i, states.length-1));
    const last = cur > 0 ? moves[cur-1] : null;
    renderBoard(states[cur], last);
    document.getElementById('step').textContent = String(cur);
  }}
  function start() {{
    if (timer) return;
    timer = setInterval(() => {{
      if (cur >= states.length-1) {{ pause(); return; }}
      show(cur+1);
    }}, 1000); // 1 second per move
  }}
  function pause() {{ if (timer) {{ clearInterval(timer); timer = null; }} }}

  document.getElementById('btnStart').onclick = start;
  document.getElementById('btnPause').onclick = pause;
  document.getElementById('btnPrev').onclick = () => show(cur-1);
  document.getElementById('btnNext').onclick = () => show(cur+1);

  // initial render
  show(0);
  </script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", type=str, default="tools/game.json", help="Input JSON file (or '-' for STDIN)")
    ap.add_argument("--output", "-o", type=str, default="tools/replay.html", help="Output HTML path")
    args = ap.parse_args()

    game = read_game_json(args.input)
    states, applied = reconstruct_states(game)
    html = emit_html(game, states, applied)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} with {len(states)-1} moves.")


if __name__ == "__main__":
    main()


