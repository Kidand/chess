// English comments per user preference.
// Frontend: rendering, human move validation, and calling backend AI.

const BACKEND_URL = 'http://127.0.0.1:5000';
let ENGINE = 'ab'; // 'ab' or 'nn'
let MODEL_PATH = '';

// ======= Board & Pieces =======
const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const GRID_W = 60; // cell width
const GRID_H = 60; // cell height
const ORIGIN_X = 30; // left padding
const ORIGIN_Y = 30; // top padding

// Board representation: 10 rows x 9 cols. Uppercase = Red, lowercase = Black
// Pieces: K(帅/将) A(仕/士) E(相/象) H(马) R(车) C(炮) S(兵/卒)
const START_FEN = "rheakaehr/9/1c5c1/s1s1s1s1s/9/9/S1S1S1S1S/1C5C1/9/RHEAKAEHR r";

// State
let board = []; // 10x9 array of chars or '.'
let side = 'r'; // 'r' or 'b' to move
let humanColor = 'r';
let selected = null; // {r,c}
let legalMovesCache = null; // speed-up for drawings
let gameOver = false;
let moveHistory = [];

const logEl = document.getElementById('log');
const statusEl = document.getElementById('status');
const turnBadge = document.getElementById('turnBadge');

const depthSel = document.getElementById('depthSel');
const sideSel = document.getElementById('sideSel');
const timeBox = document.getElementById('timeBox');
document.getElementById('newBtn').onclick = () => newGame();
document.getElementById('undoBtn').onclick = () => undo();
document.getElementById('hintBtn').onclick = () => showHint();
sideSel.onchange = () => { humanColor = sideSel.value; if (!gameOver) maybeAITurn(); draw(); };
// Optional controls if present
const engineSel = document.getElementById('engineSel');
const modelInput = document.getElementById('modelInput');
if (engineSel) engineSel.onchange = () => { ENGINE = engineSel.value; };
if (modelInput) modelInput.onchange = () => { MODEL_PATH = modelInput.value; };

canvas.addEventListener('click', onClick);

// ======= Utils =======
function inBounds(r,c){ return r>=0 && r<10 && c>=0 && c<9; }
function cloneBoard(src){ return src.map(row=>row.slice()); }
function isRed(p){ return p && p!=='.' && p===p.toUpperCase(); }
function pieceColor(p){ if(p==='.'||!p)return null; return isRed(p)?'r':'b'; }
function other(color){ return color==='r'?'b':'r'; }

// Mapping piece to display text
const TEXT = {K:'帅',A:'仕',E:'相',H:'马',R:'车',C:'炮',S:'兵',k:'将',a:'士',e:'象',h:'马',r:'车',c:'炮',s:'卒'};

function parseFEN(fen){
  const [rows, stm] = fen.split(' ');
  const lines = rows.split('/');
  board = Array.from({length:10},()=>Array(9).fill('.'));
  for(let r=0;r<10;r++){
    let c=0; for(const ch of lines[r]){
      if(/[1-9]/.test(ch)){ c += parseInt(ch,10); }
      else { board[r][c++] = ch; }
    }
  }
  side = stm==='b'?'b':'r';
  gameOver=false; moveHistory.length=0; selected=null; legalMovesCache=null; updateStatus();
  draw();
}

function toFEN(b=board, stm=side){
  let rows=[];
  for(let r=0;r<10;r++){
    let line=""; let empty=0;
    for(let c=0;c<9;c++){
      const p=b[r][c];
      if(p==='.') empty++; else { if(empty){ line+=empty; empty=0; } line+=p; }
    }
    if(empty) line+=empty; rows.push(line);
  }
  return rows.join('/')+" "+stm;
}

function newGame(){
  parseFEN(START_FEN);
  if(humanColor==='b') maybeAITurn();
}

// ======= Rendering =======
function drawBoard(){
  ctx.clearRect(0,0,canvas.width,canvas.height);

  // Outer frame
  ctx.strokeStyle = getCSS('--line'); ctx.lineWidth=2;
  ctx.strokeRect(ORIGIN_X-5, ORIGIN_Y-5, GRID_W*8+10, GRID_H*9+10);

  // Grid lines
  ctx.beginPath();
  for(let r=0;r<10;r++){
    const y = ORIGIN_Y + r*GRID_H;
    ctx.moveTo(ORIGIN_X, y);
    ctx.lineTo(ORIGIN_X + GRID_W*8, y);
  }
  // Vertical lines (break at river)
  for(let c=0;c<9;c++){
    const x = ORIGIN_X + c*GRID_W;
    ctx.moveTo(x, ORIGIN_Y);
    ctx.lineTo(x, ORIGIN_Y + GRID_H*4);
    ctx.moveTo(x, ORIGIN_Y + GRID_H*5);
    ctx.lineTo(x, ORIGIN_Y + GRID_H*9);
  }
  ctx.stroke();

  // Palace diagonals
  drawLine(3,0,5,2); drawLine(5,0,3,2);
  drawLine(3,7,5,9); drawLine(5,7,3,9);

  // River text
  ctx.fillStyle = '#a08866'; ctx.font = '16px serif';
  ctx.fillText('楚河', ORIGIN_X + GRID_W*1.5, ORIGIN_Y + GRID_H*4.6);
  ctx.fillText('汉界', ORIGIN_X + GRID_W*5.5, ORIGIN_Y + GRID_H*4.6);
}

function drawLine(c1,r1,c2,r2){
  ctx.beginPath();
  ctx.moveTo(ORIGIN_X + c1*GRID_W, ORIGIN_Y + r1*GRID_H);
  ctx.lineTo(ORIGIN_X + c2*GRID_W, ORIGIN_Y + r2*GRID_H);
  ctx.strokeStyle=getCSS('--line'); ctx.lineWidth=2; ctx.stroke();
}

function getCSS(v){ return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

function drawPieces(){
  // Highlight selection and legal moves
  if(selected){
    const {r,c} = selected;
    circle(c,r,22, getCSS('--sel'));
    const moves = legalMovesCache || generateLegalMoves(board, side);
    for(const m of moves){ if(m.fr===r && m.fc===c){
      rect(m.tc, m.tr, 26, getCSS('--hilite'));
    }}
  }

  for(let r=0;r<10;r++){
    for(let c=0;c<9;c++){
      const p = board[r][c]; if(p==='.') continue;
      drawPiece(c,r,p);
    }
  }
}

function circle(c,r,rad,color){
  const x = ORIGIN_X + c*GRID_W; const y = ORIGIN_Y + r*GRID_H;
  ctx.beginPath(); ctx.arc(x,y,rad,0,Math.PI*2); ctx.fillStyle=color+'55'; ctx.fill();
}
function rect(c,r,rad,color){
  const x = ORIGIN_X + c*GRID_W; const y = ORIGIN_Y + r*GRID_H;
  ctx.beginPath(); ctx.roundRect(x-rad,y-rad,rad*2,rad*2,6); ctx.fillStyle=color+'55'; ctx.fill();
}

function drawPiece(c,r,p){
  const x = ORIGIN_X + c*GRID_W; const y = ORIGIN_Y + r*GRID_H;
  ctx.save();
  // Disk
  ctx.beginPath(); ctx.arc(x,y,23,0,Math.PI*2);
  ctx.fillStyle = '#fffdf7'; ctx.fill();
  ctx.strokeStyle = '#cbb79c'; ctx.lineWidth=2; ctx.stroke();

  // Text
  ctx.fillStyle = isRed(p)? getCSS('--red') : getCSS('--black');
  ctx.font = '20px serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
  ctx.fillText(TEXT[p]||'?', x, y+1);

  ctx.restore();
}

function draw(){
  drawBoard();
  drawPieces();
  updateStatus();
}

function updateStatus(){
  turnBadge.textContent = side==='r'?'红方走':'黑方走';
  // Do not override winner message when game is over
  if(!gameOver){
    statusEl.textContent = '对局中';
  }
}

function canvasToCell(x,y){
  const c = Math.round((x-ORIGIN_X)/GRID_W);
  const r = Math.round((y-ORIGIN_Y)/GRID_H);
  return {r,c};
}

function onClick(e){ if(gameOver) return; if(side!==humanColor) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left; const y = e.clientY - rect.top;
  const {r,c} = canvasToCell(x,y); if(!inBounds(r,c)) return;
  const p = board[r][c];
  const pc = pieceColor(p);

  if(selected){
    // If clicked same color piece, change selection
    if(pc===humanColor){ selected = {r,c}; draw(); return; }
    // Try move
    const moves = legalMovesCache || generateLegalMoves(board, side);
    const found = moves.find(m=>m.fr===selected.r && m.fc===selected.c && m.tr===r && m.tc===c);
    if(found){ makeMove(found); draw(); setTimeout(()=>maybeAITurn(), 50); }
    else { // invalid target
      selected=null; draw();
    }
  } else {
    if(pc===humanColor){ selected = {r,c}; legalMovesCache = generateLegalMoves(board, side); draw(); }
  }
}

// ======= Move generation =======
function generateLegalMoves(b, color){
  // Generate all pseudo-legal then filter by king safety
  let moves = [];
  for(let r=0;r<10;r++) for(let c=0;c<9;c++){
    const p = b[r][c]; if(p==='.') continue; if(pieceColor(p)!==color) continue;
    moves.push(...genPieceMoves(b,r,c,p));
  }
  // Filter: cannot leave your king in check
  const legal=[];
  for(const m of moves){
    const saved = applyMove(b,m);
    if(!isInCheck(b,color)) legal.push(m);
    undoApplyMove(b,m,saved);
  }
  // Special rule: generals cannot face directly
  return legal.filter(m=>!generalsFaceAfter(b,m));
}

function genPieceMoves(b,r,c,p){
  const color = pieceColor(p);
  const res=[]; const forward = (color==='r')?-1:1; // red at bottom -> row increases downward, so forward is -1
  const palaceRows = color==='r'?[7,8,9]:[0,1,2];
  const inPalace = (rr,cc)=> palaceRows.includes(rr) && cc>=3 && cc<=5;

  switch(p.toLowerCase()){
    case 'k':{ // General
      const steps=[[1,0],[-1,0],[0,1],[0,-1]];
      for(const [dr,dc] of steps){ const rr=r+dr, cc=c+dc; if(!inBounds(rr,cc)) continue; if(!inPalace(rr,cc)) continue; if(pieceColor(b[rr][cc])!==color) res.push(mv(r,c,rr,cc)); }
      break;
    }
    case 'a':{ // Advisor (diagonal 1 inside palace)
      const steps=[[1,1],[1,-1],[-1,1],[-1,-1]];
      for(const [dr,dc] of steps){ const rr=r+dr, cc=c+dc; if(!inBounds(rr,cc)) continue; if(!inPalace(rr,cc)) continue; if(pieceColor(b[rr][cc])!==color) res.push(mv(r,c,rr,cc)); }
      break;
    }
    case 'e':{ // Elephant (two-point diagonal, cannot cross river, block on eye)
      const steps=[[2,2],[2,-2],[-2,2],[-2,-2]];
      for(const [dr,dc] of steps){
        const rr=r+dr, cc=c+dc; const er=r+dr/2, ec=c+dc/2;
        if(!inBounds(rr,cc)) continue;
        // river: red cannot go above row 4, black cannot go below row 5
        if(color==='r' && rr<5) continue; if(color==='b' && rr>4) continue;
        if(b[er][ec]!=='.') continue; // blocked eye
        if(pieceColor(b[rr][cc])!==color) res.push(mv(r,c,rr,cc));
      }
      break;
    }
    case 'h':{ // Horse (knight), block at leg
      const legs=[[0,1, -1,2],[0,1, 1,2],[1,0, 2,1],[-1,0,-2,1],[0,-1,-1,-2],[0,-1,1,-2],[1,0,2,-1],[-1,0,-2,-1]];
      for(const [lr,lc, dr,dc] of legs){ const br=r+lr, bc=c+lc; const rr=r+dr, cc=c+dc; if(!inBounds(rr,cc)) continue; if(!inBounds(br,bc)) continue; if(b[br][bc]!=='.') continue; if(pieceColor(b[rr][cc])!==color) res.push(mv(r,c,rr,cc)); }
      break;
    }
    case 'r':{ // Rook (chariot)
      slide(r,c, 1,0); slide(r,c,-1,0); slide(r,c,0,1); slide(r,c,0,-1);
      function slide(sr,sc,dr,dc){ let rr=sr+dr, cc=sc+dc; while(inBounds(rr,cc)){
        if(b[rr][cc]==='.') { res.push(mv(sr,sc,rr,cc)); rr+=dr; cc+=dc; }
        else { if(pieceColor(b[rr][cc])!==color) res.push(mv(sr,sc,rr,cc)); break; }
      }}
      break;
    }
    case 'c':{ // Cannon
      // move like rook without capture
      slideMove(r,c, 1,0); slideMove(r,c,-1,0); slideMove(r,c,0,1); slideMove(r,c,0,-1);
      function slideMove(sr,sc,dr,dc){ let rr=sr+dr, cc=sc+dc; while(inBounds(rr,cc) && b[rr][cc]==='.') { res.push(mv(sr,sc,rr,cc)); rr+=dr; cc+=dc; }
        // capture over one screen
        while(inBounds(rr,cc) && b[rr][cc]==='.') { rr+=dr; cc+=dc; }
        if(inBounds(rr,cc)){
          rr+=dr; cc+=dc; // jump over screen
          while(inBounds(rr,cc)){
            if(b[rr][cc]!=='.'){ if(pieceColor(b[rr][cc])!==color) res.push(mv(r,c,rr,cc)); break; }
            rr+=dr; cc+=dc;
          }
        }
      }
      break;
    }
    case 's':{ // Soldier
      const candidates = [[forward,0]];
      // after crossing river, can move horizontally
      if((color==='r' && r<=4) || (color==='b' && r>=5)) candidates.push([0,1],[0,-1]);
      for(const [dr,dc] of candidates){ const rr=r+dr, cc=c+dc; if(!inBounds(rr,cc)) continue; if(pieceColor(b[rr][cc])!==color) res.push(mv(r,c,rr,cc)); }
      break;
    }
  }
  return res;
}

function mv(fr,fc,tr,tc){ return {fr,fc,tr,tc}; }

function applyMove(b,m){ // returns captured piece to restore later
  const cap = b[m.tr][m.tc];
  b[m.tr][m.tc] = b[m.fr][m.fc];
  b[m.fr][m.fc] = '.';
  return cap;
}
function undoApplyMove(b,m,cap){ b[m.fr][m.fc] = b[m.tr][m.tc]; b[m.tr][m.tc]=cap; }

function findKing(b,color){ const target = color==='r'?'K':'k';
  for(let r=0;r<10;r++) for(let c=0;c<9;c++){ if(b[r][c]===target) return {r,c}; }
  return null;
}

function isInCheck(b,color){
  const king = findKing(b,color); if(!king) return true; // captured
  // enemy moves can capture king?
  const enemy = other(color);
  for(let r=0;r<10;r++) for(let c=0;c<9;c++){
    const p=b[r][c]; if(p==='.') continue; if(pieceColor(p)!==enemy) continue;
    const moves = genPieceMoves(b,r,c,p);
    for(const m of moves){ if(m.tr===king.r && m.tc===king.c) return true; }
  }
  return false;
}

function generalsFaceAfter(b,m){
  const saved = applyMove(b,m);
  const face = generalsFace(b);
  undoApplyMove(b,m,saved);
  return face;
}

function generalsFace(b){
  // If kings on same file with no pieces between, it's illegal
  let rk=findKing(b,'r'), bk=findKing(b,'b'); if(!rk||!bk) return false;
  if(rk.c!==bk.c) return false; const c=rk.c;
  let minr=Math.min(rk.r,bk.r)+1, maxr=Math.max(rk.r,bk.r)-1;
  for(let r=minr;r<=maxr;r++){ if(b[r][c]!=='.') return false; }
  return true;
}

function makeMove(m){
  const cap = applyMove(board,m);
  moveHistory.push({m,cap});
  side = other(side);
  selected=null; legalMovesCache=null;
  logMove(m, cap);
  // Checkmate detection
  if(isTerminal(board, side)) endGame();
}

function undo(){ if(!moveHistory.length||gameOver) return; // undo one ply (player + maybe AI)
  const last = moveHistory.pop(); undoApplyMove(board,last.m,last.cap); side=other(side);
  if(moveHistory.length && pieceColor(board[last.m.tr][last.m.tc])===humanColor){
    const last2 = moveHistory.pop(); undoApplyMove(board,last2.m,last2.cap); side=other(side);
  }
  gameOver=false; draw();
}

function logMove(m,cap){
  const p = board[m.tr][m.tc];
  const colorName = pieceColor(p)==='r'? '红':'黑';
  const txt = `${colorName}${TEXT[p]}: (${m.fc},${m.fr}) → (${m.tc},${m.tr})${cap&&cap!=='.'?` ×${TEXT[cap]}`:''}`;
  const div = document.createElement('div'); div.textContent = txt; logEl.appendChild(div); logEl.scrollTop=logEl.scrollHeight;
}

function isTerminal(b, toMove){
  const lm = generateLegalMoves(b,toMove);
  if(lm.length===0){
    gameOver = true;
    const checked = isInCheck(b, toMove);
    // In Xiangqi, stalemate (困毙) is a loss for the side to move
    if(checked){
      statusEl.textContent = (toMove==='r'? '黑方胜（将死）':'红方胜（将死）');
    } else {
      statusEl.textContent = (toMove==='r'? '黑方胜（困毙）':'红方胜（困毙）');
    }
    return true;
  }
  return false;
}

function maybeAITurn(){ if(gameOver) return; if(side!==humanColor){ setTimeout(()=>aiMove(), 100); } }

function endGame(){ gameOver=true; draw(); }

function showHint(){ if(gameOver) return; if(side!==humanColor) return; const moves = generateLegalMoves(board, side); if(!moves.length) return; const choice = moves[Math.floor(Math.random()*moves.length)]; selected={r:choice.fr,c:choice.fc}; legalMovesCache=[choice]; draw(); }

function log(msg){ const div=document.createElement('div'); div.textContent=msg; logEl.appendChild(div); logEl.scrollTop=logEl.scrollHeight; }

// ======= AI via backend =======
async function aiMove(){
  try{
    const maxDepth = parseInt(depthSel.value,10);
    const timeLimit = Math.max(200, Math.min(5000, parseInt(timeBox.value,10)||1200));
    const payload = { fen: toFEN(board, side), depth: maxDepth, time_ms: timeLimit, engine: ENGINE, model_path: MODEL_PATH };
    const resp = await fetch(`${BACKEND_URL}/ai-move`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    if(!resp.ok){ throw new Error(`HTTP ${resp.status}`); }
    const data = await resp.json();
    if(data && data.move){
      log(`AI nodes=${data.nodes||0} score=${(data.score||0).toFixed?data.score.toFixed(1):data.score}`);
      makeMove(data.move); draw();
    } else {
      // Fallback: random legal move
      const moves = generateLegalMoves(board, side);
      if(moves.length){ makeMove(moves[0]); draw(); }
    }
  }catch(err){
    log(`AI error: ${err.message}`);
    const moves = generateLegalMoves(board, side);
    if(moves.length){ makeMove(moves[0]); draw(); }
  }
}

// ======= Bootstrap =======
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
    const min = Math.min(w, h) / 2; r = Math.min(r, min);
    this.beginPath(); this.moveTo(x + r, y);
    this.arcTo(x + w, y, x + w, y + h, r);
    this.arcTo(x + w, y + h, x, y + h, r);
    this.arcTo(x, y + h, x, y, r);
    this.arcTo(x, y, x + w, y, r); this.closePath(); return this;
  }
}

newGame();
draw();


