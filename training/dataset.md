### 数据集说明（Xiangqi AlphaZero 自对弈 JSONL）

#### 文件结构
- `datasets/wl_games_win.jsonl`：仅记录胜局（红胜，result=1）。
- `datasets/wl_games_loss.jsonl`：仅记录败局（红负，result=-1）。
- 每行一条 JSON 记录；和棋不写入这两个文件（当前实现）。

#### 记录字段与含义
- **timestamp**: string  
  - UTC 时间戳（ISO-8601，末尾 `Z`），例如 `2025-08-10T02:20:44.290400Z`。
- **seg**: int  
  - 训练分段编号（从 1 递增）。
- **rank**: int  
  - 分布式训练进程编号（0..world_size-1）。
- **idx**: int  
  - 该 rank 在当前 `seg` 内生成的第几局（从 1 递增）。
- **result**: int  
  - 从起始方（红方）视角：`1=红胜，0=和，-1=红负`。当前数据集中仅记录 `±1`。
- **reason**: int（终局原因编码）  
  - `1=长无吃和局`；`2=对手无合法着法（胜）`；`3=达最大步数512（和）`；`4=重复局面三次（和）`；`5=认输（负）`。
- **plies**: int  
  - 半步数（一个着法为 1 ply）。通常等于 `moves` 长度。
- **caps**: int  
  - 全局累计吃子次数。
- **moves**: int[]  
  - 着法索引序列（范围 `0..8099`，对应 90×90 的 from-to 组合）。
  - 90 格展开规则：10×9 棋盘（H=10, W=9, NUM_SQUARES=90）。
  - 解码公式：`frfc = idx // 90; trtc = idx % 90; fr,fc = divmod(frfc, 9); tr,tc = divmod(trtc, 9)`。
- **start_fen**: string  
  - 开局 FEN（结尾 `r` 表示红先）。
- **mcts**: object  
  - `num_simulations, cpuct, dirichlet_alpha, dirichlet_frac, cap_boost, check_boost, fpu_value, c_base, c_init`（与自对弈时使用的 MCTS 配置一致）。
- **temperature_moves**: int  
  - 前 N 手使用温度 T=1（更随机），之后 T=0（贪心）。
- **no_capture_draw_plies**: int  
  - 连续无吃子达到该阈值判和（当前和棋不写入 win/loss 文件）。
- **envs_per_rank**: int  
  - 每个 rank 的并行自对弈线程数。
- **mcts_batch**: int  
  - 记录用途参数（不改变 MCTS 逻辑）。

#### 着法编码参考
```31:35:/Users/solidus/Documents/dev/chess/backend/encoding.py
def index_to_move(idx: int) -> Tuple[int, int, int, int]:
    frfc, trtc = divmod(idx, NUM_SQUARES)
    fr, fc = divmod(frfc, BOARD_W)
    tr, tc = divmod(trtc, BOARD_W)
    return fr, fc, tr, tc
```

#### 快速读取与统计示例
```python
# Read JSONL and compute basic stats
import json, os, glob
from collections import Counter, defaultdict

def stream_records(paths):
    for p in paths:
        with open(p, 'r') as f:
            for line in f:
                yield json.loads(line)

paths = ['datasets/wl_games_win.jsonl', 'datasets/wl_games_loss.jsonl']
paths = [p for p in paths if os.path.exists(p)]

seg_counter = Counter()
reason_counter = Counter()
seg_len = defaultdict(list)
seg_caps = defaultdict(list)

for rec in stream_records(paths):
    seg = rec['seg']
    seg_counter[seg] += 1
    reason_counter[rec['reason']] += 1
    seg_len[seg].append(rec['plies'])
    seg_caps[seg].append(rec['caps'])

print('games per seg:', dict(seg_counter))
print('reasons:', dict(reason_counter))
print('avg plies per seg:', {k: sum(v)/len(v) for k,v in seg_len.items()})
print('avg caps per seg:', {k: sum(v)/len(v) for k,v in seg_caps.items()})
```

#### 解码部分着法示例
```python
# Decode first N moves to (fr, fc, tr, tc)
from backend.encoding import index_to_move

def decode_moves(move_ids, n=10):
    # Returns list of (fr, fc, tr, tc)
    return [index_to_move(int(i)) for i in move_ids[:n]]

# Example usage
# moves = rec['moves']
# print(decode_moves(moves, 5))
```

#### 分析与处理建议
- **胜负对齐**: `result` 以红方为视角；若需按先后手拆分，请额外携带当前 `side` 并回放或在分析层统一约定。
- **半步与全步**: `plies` 为半步数；全步数可用 `plies // 2`。
- **一致性**: 正常情况下 `len(moves) == plies`。若提前终局导致差异，可基于回放校验。
- **随机性**: 前 `temperature_moves` 使用 T=1，且根节点注入 Dirichlet 噪声，分析开局多样性时需考虑。

#### 版本与扩展
- 当前仅记录胜/负局到 `wl_games_win.jsonl / wl_games_loss.jsonl`。若未来需要和棋样本，建议新增 `wl_games_draw.jsonl` 或统一加 `schema_version` 字段，便于兼容多版本分析。