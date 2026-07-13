# AutoScreen

基于 [MolPAL](https://github.com/coleygroup/molpal) 的主动学习虚拟筛选扩展项目。

## 本地结构

```text
AutoScreen/
├── molpal/              # MolPAL 源码（clone 后在此开发）
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
└── docker-compose.yml
```

## 1. 代码来源（Clone）

当前本地是直接 clone 上游仓库，**不会自动向上游贡献代码**：

```bash
git clone https://github.com/coleygroup/molpal.git molpal
cd molpal
git checkout -b autoscreeen/dev
```

你的修改默认只保存在本地（或你自己 push 的仓库）。只有当你主动向上游提 Pull Request，且维护者合并后，才会进入原项目。

如果后续想备份到自己的 GitHub，可新建独立仓库（例如 `AutoScreen`），再添加 remote：

```bash
git remote add autoscreeen https://github.com/<你的用户名>/AutoScreen.git
git push -u autoscreeen autoscreeen/dev
```

## 2. Docker 启动

在项目根目录执行：

```bash
docker compose build
docker compose up -d
docker compose exec molpal bash
```

## 3. 容器内冒烟测试

```bash
# 单元测试
pytest -q

# 最小主动学习复现（Enamine10k, lookup objective, RF + greedy）
molpal run \
  --config examples/config/Enamine10k_retrain.ini \
  --name smoke_10k \
  --metric greedy \
  --init-size 0.01 \
  --batch-sizes 0.01 \
  --model rf \
  --max-iters 2
```

## 4. 常用命令

```bash
# 进入已运行容器
docker compose exec molpal bash

# 停止
docker compose down

# 重新构建（依赖变更后）
docker compose build --no-cache
```

## 说明

- 当前 Docker 镜像使用 **CPU 版** 依赖（与 MolPAL CI 一致），适合第一步复现和开发。
- 若需要 GPU（MPN 模型训练），后续可再加 `docker/Dockerfile.gpu`。
- `molpal` 目录以 volume 挂载，容器内代码修改会同步到本地。

## 使用已有 conda 环境（如 `hgtdr`）

可以不新建虚拟环境，但 `pip install -e . --no-deps` 不会自动装依赖，需要补装：

```powershell
cd g:\AutoScreen\molpal
conda activate hgtdr
pip install -e . --no-deps
pip install "ray==2.20.0" "aiohttp==3.9.5" configargparse h5py tabulate tensorflow tensorflow-addons tqdm pytorch-lightning
```

冒烟测试（注意参数是 `--output-dir`，不是 `--name`；Ray 会由 MolPAL 自动启动，一般不必手动 `ray start`）：

```powershell
# 建议先预计算指纹（Windows 上 Ray 并行指纹容易卡住）
python ..\scripts\prep_fps_serial.py

molpal run --config examples/config/Enamine10k_retrain.ini --output-dir smoke_10k --fps libraries/Enamine10k.h5 --metric greedy --init-size 0.01 --batch-sizes 0.01 --model rf --max-iters 2
```

Windows 注意：
- 建议固定 `ray==2.20.0`，较新版本在本机可能出现启动超时。
- 若遇 `aiohttp` SSL 报错，使用 `aiohttp==3.9.5`。
- 首次运行会预计算 10k 分子指纹，CPU 上可能需要几分钟。

## 5. GitHub 与 GPU 服务器同步

推荐用 **一个 AutoScreen 仓库** 作为唯一同步源（GitHub），本地和服务器都只跟它同步。

### 一次性：把项目推到 GitHub

```powershell
# 1. 在 GitHub 网页新建空仓库，例如 AutoScreen（不要勾选 README）

# 2. 去掉 molpal 内嵌 git，合并为单一仓库
cd g:\AutoScreen
Remove-Item -Recurse -Force molpal\.git

# 3. 初始化并首次提交
git init
git add .
git commit -m "Initial AutoScreen: MolPAL base, docker, Windows fps workaround"
git branch -M main
git remote add origin https://github.com/<你的用户名>/AutoScreen.git
git push -u origin main
```

### 日常同步流程

```text
本地改代码 → git add / commit → git push
                              ↓
GPU 服务器 ← git pull ← GitHub（唯一真相源）
```

本地：

```powershell
git add .
git commit -m "feat: add mock robot objective"
git push
```

GPU 服务器：

```bash
cd ~/AutoScreen
git pull

# 首次在服务器上
git clone https://github.com/<你的用户名>/AutoScreen.git
cd AutoScreen
conda env create -f molpal/environment.yml   # 或复用已有环境
conda activate molpal
pip install -e molpal --no-deps
pip install "ray>=1.11" configargparse h5py rdkit scikit-learn tabulate tensorflow tensorflow-addons tqdm pytorch-lightning

# 指纹文件不进 git，在服务器上重新生成
python scripts/prep_fps_serial.py
```

### 建议分支策略

| 分支 | 用途 |
|------|------|
| `main` | 能跑通的稳定版本 |
| `dev` | 日常开发（Mock Oracle、Campaign 等） |
| `exp/gpu` | 服务器上 MPN / 大库实验 |

```powershell
# 本地开功能分支
git checkout -b dev

# 服务器拉同一分支
git fetch origin
git checkout dev
git pull
```

### 不要指望 Git 同步的东西

| 内容 | 处理方式 |
|------|----------|
| `libraries/*.h5` 指纹 | 各机器运行 `scripts/prep_fps_serial.py` |
| `smoke_10k/` 运行结果 | 重新跑或放 `benchmarks/` 只提交 CSV 摘要 |
| 大数据集 AmpC 等 | `wget`/`scp` 单独传，或服务器本地下载 |
| conda 环境 | 用 `environment.yml` 重建，不打包整个 env |

### 服务器大文件可选方案

实验结果、checkpoint 若很大，用 Git LFS 或网盘/rsync，不要直接堆进普通 git：

```bash
# 例：只同步某次实验结果目录
rsync -avz ./molpal/smoke_10k/ user@gpu-server:~/AutoScreen/molpal/smoke_10k/
```

### 注意

- 当前 `molpal/` 内仍有独立 `.git` 时，根目录 `git add` 无法跟踪其中改动；按上面步骤删除 `molpal\.git` 后再初始化。
- 上游 MolPAL 更新时，可用 `git remote add molpal-upstream https://github.com/coleygroup/molpal.git` 再 cherry-pick，不必再维护两个仓库。
