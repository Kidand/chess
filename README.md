## 中国象棋 AlphaZero（自博弈训练 + 游玩）

本项目是一个“可以游玩 + 可自博弈训练”的中国象棋（Xiangqi）引擎与工具集，采用 AlphaZero 风格的策略-价值网络与 MCTS 搜索，支持多卡分布式自对弈训练、数据落盘与回放可视化。

参考论文：Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"（`https://arxiv.org/pdf/1712.01815`）。


### 功能概览
- **前端对弈**：在浏览器中与 AI 下棋（内置 α-β 引擎；也可切换到神经网络引擎）。
- **后端服务**：Flask HTTP 服务，提供走子合法性、AI 落子、模型加载等接口。
- **自博弈训练**：多进程/多卡自对弈生成数据，训练 AlphaZero 策略-价值网络；已修复常见“全和棋吸引子”与 MCTS 回传统计问题。
- **数据落盘**：按段（segment）与卡（rank）实时写入胜/负对局 JSONL，并在每段结束聚合合并。
- **回放工具**：将任意一局 JSON 生成可交互 HTML 回放，逐步复现走子。


### 目录结构
```
backend/           # 规则、编码、后端服务与 NN 引擎适配
  xiangqi.py       # 中国象棋规则与合法着生成
  encoding.py      # (C,H,W) 输入与 8100 维策略向量编码
  app.py           # Flask 服务：/ai-move /load-model + 前端静态资源
  nn_engine.py     # 神经网络 + MCTS 推理封装（与训练共享模型/MCTS）

frontend/          # 浏览器前端（Canvas 渲染 + 操作交互）
  index.html
  app.js

training/          # AlphaZero 训练管线
  az_model.py      # 策略-价值网络（ResNet 主干，支持 flat/structured 两种 policy 头）
  az_mcts.py       # AlphaZero MCTS（含吃子/将军优先与根噪声；支持分档吃子加权）
  az_selfplay.py   # 自博弈回路（价值以“当前行棋方”为视角）
  az_train.py      # 训练入口（多卡进度汇总、数据写盘与分段聚合）
  az_aug.py        # 数据增强（左右镜像）
  az_config.py     # 配置结构（默认超参）
  az_replay.py     # 回放缓冲（训练数据集）
  dataset.md       # JSONL 数据字段说明（参考）

tools/
  replay_to_html.py  # 将单局 JSON 生成交互式 HTML 回放
  replay.html        # 示例输出
```


### 环境与安装
- 建议环境：Python 3.10+，CUDA 11+（如使用 GPU 训练），Linux（或 WSL2）。
- 安装依赖：
```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -r requirements.txt
```


### 运行前端与后端（游玩）
1) 启动后端服务（默认 127.0.0.1:5000）：
```bash
python -m backend.app
```
2) 打开浏览器访问：`http://127.0.0.1:5000/`
- 前端默认使用 α-β 引擎。可在界面切换到 NN 引擎，并通过“加载模型”指定已训练的权重（`*.pt`）。


### 自博弈训练（多卡示例）
基本命令（与本仓库默认兼容）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO \
NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1 TORCH_NCCL_BLOCKING_WAIT=1 NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node=8 -m training.az_train \
  --out runs/xq_az_cont \
  --segments 16 --selfplay_per_seg 32 \
  --channels 256 --blocks 12 \
  --batch_size 512 --lr 1e-3 \
  --num_simulations 128 --mcts_batch 128 \
  --envs_per_rank 8 --temperature_moves 12 \
  --no_capture_draw_plies 50
```

常用参数（`python -m training.az_train -h` 可查看全部）：
- `--out`：输出目录（权重与数据将写入此目录）。
- `--segments`：分段数；每段先自博弈生成数据，再训练。
- `--selfplay_per_seg`：每段、每个 rank 的自博弈局数（总局数=该值×world_size）。
- `--channels / --blocks`：模型规模（ResNet 通道数与残差块数）。
- `--batch_size`：每 rank 的 batch 大小。
- `--lr`：初始学习率（内置余弦退火）。
- `--num_simulations`：MCTS 仿真次数（越大越强但越慢）。
- `--envs_per_rank`：每个 rank 的并行自博弈线程数。
- `--temperature_moves`：前 N 个半步使用温度 T=1（探索），之后 T=0（贪心）。
- `--no_capture_draw_plies`：连续无吃子达到该阈值判和。
- `--resume`：从已有权重恢复训练（`*.pt`）。
 - `--policy_head`：`flat` 或 `structured`。structured 使用 99 个动作平面输出 policy（更小参数，更易合法性 mask）。默认 `flat`，向后兼容。


### 训练产出
- 权重：
  - `out/az_seg_{k}.pt`：每段结束保存一次（仅 rank 0）。
  - `out/az_final.pt`：训练全部结束的最终权重（仅 rank 0）。
- 对局 JSONL：实时写入、分段分卡分片，段末自动聚合（仅胜负局）
  - 分片：
    - `out/datasets/seg_{k}/win_rank{r}.jsonl`
    - `out/datasets/seg_{k}/loss_rank{r}.jsonl`
  - 段末聚合（rank 0 自动生成）：
    - `out/datasets/seg_{k}/win.jsonl`
    - `out/datasets/seg_{k}/loss.jsonl`
  - 记录字段见 `training/dataset.md`（如 `result / winner / reason / moves / plies / caps / mcts` 等）。
  - 默认不记录和棋到 JSONL（仅进入回放训练）。


### 设计要点与已修复问题
- **价值视角**：训练样本的 `value` 以“当前行棋方”为视角标注（奇偶步对最终 z 做符号翻转），避免红方固定视角导致的对称性偏置与“和棋吸引子”。
- **MCTS 回传**：修复了将访问数累计到父节点的错误，现沿“父→子”边回传访问与价值（AlphaZero 规范），根结点策略不再退化为首合法步。
- **探索与进攻倾向**：
  - 在 MCTS 中对吃子与将军适度放大先验（`cap_boost / check_boost`）。
  - 新增“按被吃子分档的先验加权（tiered capture prior）”，并按段退火：
    - 段 1–4：强引导（车×2.0、炮×1.6、马×1.5、兵/卒×1.2、仕/士/相/象×1.1；将军×1.6）
    - 段 5–8：减半（将军×1.3）
    - 段 ≥9：关闭（回到 1.0）
- **分布式日志**：rank 0 汇总全局进度条（总量=各 rank 自博弈局数之和），后缀展示 `R/B/D`、平均半步与吃子计数。
- **对局写盘**：胜负对局实时写入 `{out}/datasets/seg_{k}` 子目录，段末自动合并，便于分析与回放。


### 推荐超参与分段训练命令（已内置分段加权）
> 以下命令已自动启用“分档吃子加权 + 分段退火”（无需额外开关）；需要更强实力时可逐段提升 `--num_simulations` 并降低 `--lr`。

- 阶段 A（前 8 段，快速制造胜负）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO \
NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1 TORCH_NCCL_BLOCKING_WAIT=1 NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node=8 -m training.az_train \
  --out runs/xq_az_stageA \
  --segments 8 --selfplay_per_seg 32 \
  --channels 256 --blocks 12 \
  --batch_size 512 --lr 2e-3 \
  --num_simulations 192 --envs_per_rank 8 \
  --temperature_moves 14 --no_capture_draw_plies 40
```

- 阶段 B（后 8 段，稳定提升，自动弱化/关闭加权）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO \
NCCL_ASYNC_ERROR_HANDLING=1 NCCL_BLOCKING_WAIT=1 TORCH_NCCL_BLOCKING_WAIT=1 NCCL_IB_DISABLE=1 \
torchrun --standalone --nproc_per_node=8 -m training.az_train \
  --out runs/xq_az_stageB \
  --segments 8 --selfplay_per_seg 32 \
  --channels 256 --blocks 12 \
  --batch_size 512 --lr 1e-3 \
  --num_simulations 256 --envs_per_rank 8 \
  --temperature_moves 12 --no_capture_draw_plies 60 \
  --resume runs/xq_az_stageA/az_seg_8.pt
```

说明：如追求“更快见效”，也可临时使用小模型（例如 `--channels 192 --blocks 10`）以提高对局吞吐；稳定后再回到大模型长训。


### 将单局导出为可交互回放 HTML
```bash
# 方式一：直接指定一行 JSON（或手工拼一个 JSON 对象）
python tools/replay_to_html.py --input path/to/game.json --output tools/replay.html

# 方式二：从 JSONL 中拷贝一行，通过管道传入
head -n 1 runs/xq_az_cont/datasets/seg_1/win.jsonl | \
  python tools/replay_to_html.py --input - --output tools/replay.html
```


### 模型结构（简述）
- 输入：15 通道（红 7 + 黑 7 + 行棋方 1），棋盘 10×9。
- 主干：`channels × blocks` 的 ResNet。
- 策略头（两种）：
  - `flat`：Conv(1×1, 64) → BN → ReLU → Flatten(64×10×9) → Linear → 8100 logits（与旧版完全兼容）。
  - `structured`：Conv(1×1, K=99) → 输出 (B,99,10,9) 展平为 K×90；动作以“方向/步数/棋型”平面编码。
- 价值头：Conv(1×1) + BN + ReLU + FC(256) + FC(1) + Tanh。


### 中国象棋规则实现（要点）
- 合法着生成：分子力走法 + 王不入将线（“将帅照面”）约束 + 自身不被将军约束。
- 判和规则：
  - 连续无吃子达到阈值（可配置 `--no_capture_draw_plies`）。
  - 重复局面三次（自博弈中判和）。
  - 达到最大半步数（512）判和。
- 终局：行棋方无合法着法（将死/困毙）判负。


### 故障排查（Tips）
- 进度条不动或统计异常：请确认各 rank 单局完成后能实时累计（日志后缀会更新 `R/B/D`/`avg_plies`/`avg_caps`）。
- 数据文件未出现：查看 `{out}/datasets/seg_{k}/win_rank{r}.jsonl` 是否持续增长；段末合并文件 `win.jsonl / loss.jsonl` 会在进入训练前生成。
- NCCL/分布式问题：建议 Linux + CUDA 环境；Windows 用户可使用 WSL2；必要时将后端改为 `gloo`（CPU）以验证流程。


### 许可与致谢
- 基于 AlphaZero 思想的自博弈训练实现，参考了围棋/国际象棋/将棋相关开源实现与论文。
- 规则与可视化均为教学及研究用途，欢迎 Issue/PR 共同完善。


### 引用
如果本项目对你的研究或产品有帮助，推荐引用 AlphaZero 论文并在仓库中致谢本项目。


