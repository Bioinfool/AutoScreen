# AutoScreen

**主动学习决策与任务编排引擎**——在有限评价预算下，选择下一批最有价值的化合物。

虚拟筛选（离线 Replay / AutoDock Vina）与机器人/高通量实验都是可插拔的**执行后端**，不是决策层本身。AutoScreen 不控制机械臂、液路或实验协议，也不替代 LIMS。

```text
候选分子库
    ↓
AutoScreen（代理模型 + 采集策略 + 约束）
    ↓ 提交 Job
┌─────────────┬──────────────┬─────────────────┐
│ ReplayOracle│ Vina 对接    │ Robot HTTP 协议 │
└─────────────┴──────────────┴─────────────────┘
    ↓ Observation
更新模型 → 下一轮
```

## 三种执行后端

| Executor | 作用 |
|----------|------|
| `ReplayExecutor` | 公开数据集隐藏标签，可复现的离线主动学习评测 |
| `VinaExecutor` | 调用 AutoDock Vina，真实对接虚拟筛选 |
| `RobotExecutor` | HTTP 连接机器人/实验平台（本仓库附带 `robot_mock` 模拟服务） |

## 快速开始

```bash
pip install -e ".[robot,dev,prep]"

# 小库冒烟（~1 万）
python scripts/prep_fps_serial.py --name Enamine10k
python scripts/prep_moo_labels.py --name Enamine10k
python -m autoscreen.cli run --config configs/replay_enamine10k.yaml

# 主评测库（~5 万，推荐）
python scripts/prep_fps_serial.py --name Enamine50k
python scripts/prep_moo_labels.py --name Enamine50k
python -m autoscreen.cli run --config configs/replay_enamine50k.yaml

# 机器人扩展：模拟服务的 truth 必须与 campaign 的 moo_csv 一致
# PowerShell: $env:AUTOSCREEN_TRUTH_MOO="data/Enamine10k_moo.csv.gz"
export AUTOSCREEN_TRUTH_MOO=data/Enamine10k_moo.csv.gz
uvicorn robot_mock.app:app --host 0.0.0.0 --port 8080
python -m autoscreen.cli run --config configs/robot_mock.yaml
```

Docker（决策引擎 + 模拟机器人两服务）：

```bash
docker compose up --build -d robot-mock
docker compose run --rm autoscreen python -m autoscreen.cli run --config configs/robot_mock.yaml
```

## 数据与库规模

示例数据在 [`data/`](data/)：

| 规模 | 库文件 | 用途 |
|------|--------|------|
| ~1 万 | `Enamine10k.*` | 冒烟 / 快速调试（仓库自带） |
| ~5 万 | `Enamine50k.*` | **主评测候选库（推荐）** |

每种规模包含：`*.csv.gz`（SMILES）、`*_scores.csv.gz`（预计算对接分）、`*_moo.csv.gz`（dock/QED/SA）、`*.h5`（指纹，本地用 `prep_fps_serial.py` 生成，不进 git）。

`pool_idx` 是「库与 MOO 对齐后」的下标，不是原始 CSV 行号；换库时请同步改 YAML 的三条路径，并让 `robot_mock` 的 `AUTOSCREEN_TRUTH_MOO` 指向同一份 `*_moo.csv.gz`。

更大库（如百万级 AmpC）可按同样格式放入 `data/`，改 YAML 即可，无需改代码。

## 叙事边界

- 可以说：主动学习虚拟筛选 + 可接入自动化实验的编排与协议层  
- 不可以说：已完成真实机器人药筛（除非已接入真实平台并跑通实验）

## 开发与测试

```bash
pytest -q
```

模块：`autoscreen/core/`、`autoscreen/executors/`、`autoscreen/protocol/`、`robot_mock/`。
