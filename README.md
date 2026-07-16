# AutoScreen

面向自主实验平台的**多目标主动学习虚拟筛选**框架，基于开源 [MolPAL](https://github.com/coleygroup/molpal) 二次开发。

**项目目的**：为实验室机器人药筛流程预先搭建一套完整、可即插即用的软件资产。当前用模拟后端跑通全部闭环，
只需实现一个客户端接口即可直接接入，主动学习主循环无需改动。

## 在 MolPAL 之上做的功能扩展

- **单目标 → 多目标主动学习**：同时优化活性（对接分数）、成药性（QED）、可合成性（SA），每个目标一个随机森林代理，纯 CPU。
- **多种批次采集策略**：random / greedy / 加权标量化(ParEGO) / UCB / 超体积改进(Pareto-HVI)，并用超体积、Pareto 召回率做定量对比。
- **同步 → 异步孔板实验流程**：孔状态机（SUBMITTED→RUNNING→COMPLETED/FAILED/QC_REJECTED）、部分结果回收、实验失败、QC 门控、断点续跑。
- **孔板感知批次选择**：96 孔布局（80 实验 + 阳/阴/空白对照 + 重复孔），兼顾采集分数与结构多样性，并做可合成性/库存可行性过滤。
- **机器人平台接口**：抽象 `RobotClient`（Mock 与 HTTP 两套实现 + JSON 契约），主循环与后端解耦，接真实平台即插即用。

## 快速开始（Docker，推荐）

镜像已把代码、示例数据和预计算指纹全部打包，开箱即用：

```bash
docker compose build
docker compose run --rm autoscreen bash
```

进入容器后：

```bash
# 1) 多目标主动学习：5 种采集策略对比（结果写到 results/）
python -m autoscreen.run --strategy all --seeds 3 --iters 10 --out results
python -m autoscreen.analyze --results results/results.json --out results

# 2) 异步孔板实验 campaign（模拟机器人，结果写到 runs/）
python -m autoscreen.run_campaign --rounds 6 --out runs/demo
# 中断后断点续跑：
python -m autoscreen.run_campaign --rounds 4 --out runs/demo --resume
```

`results/` 与 `runs/` 已挂载回宿主机，容器结束后结果仍在。

## 快速开始（本地 Python）

```bash
pip install -e molpal --no-deps
pip install numpy scikit-learn rdkit h5py pymoo tqdm
python scripts/prep_fps_serial.py      # 预计算指纹（首次）
python -m autoscreen.run --strategy all --seeds 3 --iters 10 --out results
python -m autoscreen.run_campaign --rounds 6 --out runs/demo
```

## 目录结构

```text
autoscreen/
├── data.py / model.py / metrics.py   # 多目标数据、RF 代理、超体积等指标
├── acquire.py / run.py / analyze.py  # 采集策略、主循环、结果聚合出图
├── plate.py                          # 孔板感知批次选择
├── experiment.py / mock_backend.py   # 实验状态机 + 模拟机器人后端
├── robot_client.py                   # 机器人/LIMS 客户端接口（Mock + HTTP 桩）
└── campaign.py / run_campaign.py     # 异步 campaign 编排 + 断点续跑
scripts/                              # 指纹预计算、多目标标签生成
molpal/                               # MolPAL 源码（代理基座）
```

## 接入真实机器人平台

只需实现 `RobotClient`（参考 `autoscreen/robot_client.py` 里的 `HttpRobotClient` 桩与 JSON 契约），
Campaign 代码不变：

```python
class LabRobotClient(RobotClient):
    def submit_plate(self, campaign_id, layout): ...   # 提交板布局，返回 job_id
    def fetch_results(self, job_id): ...               # 轮询结果 → CompoundResult
    def is_done(self, job_id): ...
```

## 说明

- 所有目标内部统一为"越大越好"：`activity = -dock`、`qed = qed`、`sa_ease = -sa`。
- 绘图使用内置 `svgplot.py` 直接输出 SVG，无需 matplotlib。
- MolPAL 原有的对接/单目标筛选功能保留，详见 `molpal/` 官方文档。
