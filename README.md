# Tiny DiT × DDPM：从零学习扩散模型

面向已经理解 Transformer、希望亲手实现扩散模型的学习项目。目标是在 **MacBook Air M5 / 24 GB** 上，使用 PyTorch 实现 MNIST 手写数字生成：先无条件生成，再指定数字类别生成。

**M0 已完成，M1–M3 已通过运行验收；M3 Tiny DiT 核心代码由学习者实现，CPU/MPS 检查通过。下一阶段为 M4 训练循环，M4–M7 尚未实现。**

## 学习目标

- 理解 DDPM 的前向加噪、噪声预测目标和反向采样。
- 将图片转换为 patch token，用小型 Transformer 预测噪声。
- 独立实现训练循环、保存与恢复、生成图片和类别条件。
- 最后对照原版 DiT，理解 adaLN-Zero、潜空间扩散与本项目的区别。

DDPM 是扩散训练与采样的方法，Transformer 是去噪网络的架构。本项目首先直接在 MNIST 像素空间建模；它是教学用的简化实现，并非原版 DiT 的完整复现。

## Milestone 路线

| 阶段 | 内容 | 验收产物 | 状态 |
| --- | --- | --- | --- |
| M0 | uv、Python、PyTorch、Git 与设备检查 | 环境检查通过，依赖已锁定 | 已完成 |
| M1 | MNIST 加载、归一化、可视化 | 原始图片网格与数据形状检查 | 运行验收通过，待学习自检 |
| M2 | DDPM 噪声日程与前向加噪 | 同一图片的逐步加噪图、数学性质检查 | 运行验收通过 |
| M3 | Tiny DiT 与时间步条件 | 输入/输出形状正确，前向和反向传播通过 | CPU/MPS 运行验收通过 |
| M4 | 噪声预测训练与小样本调试 | 小样本训练曲线、checkpoint 与恢复验证 | 未开始 |
| M5 | DDPM 反向采样与完整训练 | 固定种子生成图、去噪过程图 | 未开始 |
| M6 | 数字类别条件与 CFG | 按类别排列的生成图与 CFG 对照 | 未开始 |
| M7 | adaLN-Zero 与原版 DiT 对照（选修） | 结构对照笔记与一次受控实验 | 未开始 |

每阶段的原理、实现任务和验收标准见 [docs/MILESTONES.md](docs/MILESTONES.md)。完成当前阶段并能解释结果后，再进入下一阶段。

## M1：开始运行与阅读

```bash
uv run python -m scripts.inspect_data
uv run python -m scripts.inspect_data --split test --no-download
```

首次运行自动下载 MNIST，之后读取本地缓存。命令扫描所选 split 的全部 batch，将带数字标签的图片网格和 JSON 检查报告保存在 `outputs/m1/`。

先读 [M1 学习说明](docs/M1.md)，再读 [数据模块](ddpm/data.py) 和 [检查脚本](scripts/inspect_data.py)。本阶段只处理 CPU 上的数据，不需要启动 MPS 或训练模型。

实测训练集 60,000 张、测试集 10,000 张；首批图片为 `[64, 1, 28, 28]` float32，像素范围 `[-1, 1]`，标签为 `[64]` int64。最后一批分别为 32 和 16 张，均已保留。原始像素往返检查、图片与原始标签对应检查、固定 seed 的打乱顺序检查均通过。

## M2：前向加噪

学习者已完成 [diffusion.py](ddpm/diffusion.py) 中的噪声日程、系数提取、随机时间步和前向加噪。公式、形状与自检题见 [M2 学习说明](docs/M2.md)。

```bash
# 按模块复查，例如噪声日程
uv run pytest tests/test_diffusion.py -q -k schedule

# M2 独立验收与可视化（之后新增阶段的未完成练习不会影响此命令）
uv run pytest tests/test_diffusion.py -q
uv run python -m scripts.visualize_forward
```

2026-10-29 验收：15 项测试全部通过，代码及格式检查通过；加噪图片、噪声日程图与原始数值报告已生成于 `outputs/m2/` 并查看。默认日程最后一步 alpha_bar 约为 `4.03583e-5`，核心结果保留了超出 `[-1,1]` 的噪声像素。

## M3：Tiny DiT 去噪网络

学习者已完成 [model.py](ddpm/model.py) 中的 patch 拆分/还原、时间编码、Transformer block 与 TinyDiT 组装。每个函数包含功能、参数和返回值说明，关键步骤保留中文提示；结构说明与验收记录见 [M3 学习说明](docs/M3.md)。

```bash
# 单独复查 M3-1 和 M3-2 的像素排列
uv run pytest tests/test_model.py -q -k patch

# 完整测试与 CPU/MPS 检查
uv run pytest -q
uv run python -m scripts.check_model --device cpu
uv run python -m scripts.check_model --device mps --output-dir outputs/m3/mps
```

2026-10-30 验收：M3 的 22 项测试及 M2 的 15 项回归测试全部通过，代码与格式检查通过。默认模型有 **935,568** 个可训练参数，CPU/MPS 均通过前向、反向、有限梯度与一次参数更新检查，输入输出均为 `[2,1,28,28]`。报告保存在 `outputs/m3/model_check.json` 与 `outputs/m3/mps/model_check.json`。检查使用合成数据，仅验证网络可运行，训练效果将在后续阶段验证。

## 环境与常用命令

在终端进入项目目录：

```bash
cd /Users/zliu/Projects/DiT/ddpm
uv sync --locked
uv run python scripts/check_env.py
```

项目使用 Python 3.12；uv 会按 `.python-version` 选择解释器，在 `.venv` 创建环境，按 `uv.lock` 安装依赖。日常使用 `uv run`，无需手动激活环境。首次安装需要网络。

```bash
# 严格检查本机 MPS 是否可用，并实际运行一次训练运算
uv run python scripts/check_env.py --device mps

# CPU 对照检查
uv run python scripts/check_env.py --device cpu

# 代码检查
uv run ruff check .
uv run ruff format --check .

# 已实现阶段的验收与回归测试
uv run pytest
```

依赖由 uv 管理：使用 `uv add 包名` 或 `uv add --dev 包名` 添加依赖；同时提交 `pyproject.toml` 和 `uv.lock`。有意升级时使用 `uv lock --upgrade` 并重新验证环境。

### M0 验证记录（2026-10-28）

- macOS 26.6 / arm64，Python 3.12.15，uv 0.13.5。
- PyTorch 2.14.1、torchvision 0.29.1、matplotlib 3.11.2、tqdm 4.70.1。
- `uv sync --locked` 成功，MPS 构建与可用性均为 `True`。
- MPS 和 CPU 上均通过注意力网络的前向、反向、有限梯度与参数更新检查。
- torchvision 归一化与 matplotlib 绘图检查通过；Ruff 检查和格式检查通过。
- 此记录仅验证 M0 环境；M0 验证时尚未下载 MNIST、训练扩散模型或验证生成质量。

## 初始实验配置

以下是后续实现时的起点，目前尚无训练入口。

| 项目 | 初始配置 |
| --- | --- |
| 数据 | MNIST，`[B, 1, 28, 28]`，归一化至 `[-1, 1]` |
| Patch | `4 × 4`，49 个 token |
| Transformer | 4 层，隐藏维度 128，4 个注意力头，MLP ratio 4 |
| 条件 | 先时间步；M6 再加入类别 |
| 输出 | 与图片同形状的噪声预测；不学习方差 |
| 扩散 | DDPM，线性 beta 日程，初始 T=1000 |
| 数值精度 | float32 |
| Batch size | 从 64 试起，按实测速度与内存调整 |
| 数据加载 | 先 `num_workers=0`，验证正确性后再调整 |
| 设备 | MPS 可用时使用 MPS，CPU 用于对照与调试 |

训练耗时需通过实际运行测量。先用少量数据和短运行验证逻辑，再执行完整训练。遇到 MPS 不支持的运算时先定位问题，使用 CPU 对照，不默认开启隐式 CPU fallback。

## 目录

```text
ddpm/
├── AGENTS.md             # 后续协作与逐步教学约定
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── ddpm/
│   ├── __init__.py
│   ├── data.py           # MNIST、归一化与 DataLoader
│   ├── diffusion.py      # M2 已完成的前向加噪
│   └── model.py          # M3 已完成的 Tiny DiT 噪声预测器
├── docs/
│   ├── MILESTONES.md
│   ├── M1.md             # M1 阅读说明、运行结果与练习
│   ├── M2.md             # M2 公式、练习顺序与验收记录
│   └── M3.md             # M3 网络结构、练习顺序与验收方式
├── tests/
│   ├── test_diffusion.py # M2 数学性质与边界检查
│   └── test_model.py     # M3 像素顺序、条件、注意力与梯度
└── scripts/
    ├── __init__.py
    ├── check_env.py      # 仅检查环境
    ├── inspect_data.py   # M1 扫描、绘图与数据报告
    ├── visualize_forward.py # M2 前向加噪绘图
    └── check_model.py    # M3 CPU/MPS 前向、反向与参数更新检查
```

后续按阶段扩展 `ddpm/`、`tests/`，创建训练/采样脚本。数据存入 `data/`，实验图片、日志和权重存入 `outputs/`，均不进入 Git。便于学习的结果说明和小型示例图可有意保存至 `docs/`。

## Git

远程地址：`git@github.com:zliuz/DiT-ddpm.git`。本地默认分支为 `main`。远程绑定不表示已经推送或验证远程权限。

建议在每个 milestone 验收后提交一次。提交应包含对应代码、必要检查与学习记录；不要提交数据集、虚拟环境或权重。

## 参考资料

- [DDPM 论文](https://arxiv.org/abs/2006.11239)
- [DiT 官方实现](https://github.com/facebookresearch/DiT)
- [PyTorch MPS 文档](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [uv 项目管理](https://docs.astral.sh/uv/guides/projects/)
