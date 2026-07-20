# AutoScreen

**主动学习决策与任务编排引擎**——在有限评价预算下选择下一批最有价值的化合物。

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

算法思路受 [MolPAL](https://github.com/coleygroup/molpal) 启发；编排、执行器与通讯协议为自研。仓库内 `molpal/` 主要提供示例数据与可选对照，**不是**“改了 MolPAL 源码冒充上游贡献”。

## 三种执行后端

| Executor | 作用 |
|----------|------|
| `ReplayExecutor` | 公开数据集隐藏标签，可复现的离线主动学习评测 |
| `VinaExecutor` | 调用 AutoDock Vina，真实对接虚拟筛选 |
| `RobotExecutor` | HTTP 连接机器人/实验平台（本仓库附带 `robot_mock` 模拟服务） |

## 快速开始

```bash
pip install -e ".[robot,dev]"
python scripts/prep_fps_serial.py          # 首次：预计算指纹
python scripts/prep_moo_labels.py          # 首次：多目标标签（若尚未生成）

# 阶段1 主线：离线 Replay 主动学习
autoscreen run --config configs/replay_enamine10k.yaml

# 阶段2：先起模拟机器人，再跑 robot campaign
uvicorn robot_mock.app:app --host 0.0.0.0 --port 8080
autoscreen run --config configs/robot_mock.yaml
```

Docker（决策引擎 + 模拟机器人两服务）：

```bash
docker compose up --build -d robot-mock
docker compose run --rm autoscreen autoscreen run --config configs/robot_mock.yaml
```

## 叙事边界（请诚实使用）

- 可以说：主动学习虚拟筛选 + 可接入自动化实验的编排与协议层  
- 不可以说：已完成真实机器人药筛（除非你已接入真实平台并跑通实验）

## 开发与测试

```bash
pytest -q
```

详见各阶段模块：`autoscreen/core/`、`autoscreen/executors/`、`autoscreen/protocol/`、`robot_mock/`。
