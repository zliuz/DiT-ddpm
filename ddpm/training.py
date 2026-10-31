"""M4 训练练习：6 处核心 TODO 由学习者实现，不包含反向采样。

M4 的数据选择、时间步与噪声都在 CPU 采样，再将训练 batch 送到计算设备。
当前 TinyDiT 的 forward 没有 dropout 等随机运算，因此保存 CPU torch RNG
即可延续本阶段的随机训练序列；将来新增随机算子时必须扩展 checkpoint。
"""

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ddpm.diffusion import DiffusionSchedule, q_sample, sample_timesteps


@dataclass(frozen=True)
class TrainConfig:
    """训练与模型配置；保存时整体记录，恢复时要求完全一致。

    额外训练步数、输出路径与设备由入口单独管理，不影响本配置的比较。
    """

    mode: str = "overfit"  # overfit 固定图片/t/噪声；random 每步重新采样。
    batch_size: int = 8  # 每次更新的图片数 B；本阶段从小 batch 调试。
    num_samples: int = 64  # 使用训练集最前面的图片数；random 从中有放回抽样。
    seed: int = 0  # 新实验的初始化种子；恢复时以 checkpoint 中的 RNG 为准。
    learning_rate: float = 1e-3  # AdamW 学习率；作为小样本调试起点。
    weight_decay: float = 0.0  # AdamW 权重衰减系数；调试拟合先使用 0。
    num_steps: int = 1000  # 扩散日程长度 T；不是优化器更新次数。
    beta_start: float = 1e-4  # 线性日程的第一项 beta。
    beta_end: float = 0.02  # 线性日程的最后一项 beta。
    image_size: int = 28  # 图片边长 S；命令行入口固定使用 MNIST 的 28。
    in_channels: int = 1  # 通道数 C；命令行入口固定使用 MNIST 的单通道。
    patch_size: int = 4  # patch 边长 P；必须整除 S。
    hidden_size: int = 128  # token 隐藏维度 D；偶数且能被注意力头数整除。
    depth: int = 4  # 独立 Transformer block 的数量。
    num_heads: int = 4  # 每个 block 的注意力头数。
    mlp_ratio: int = 4  # block 中间 MLP 的宽度倍率。

    def __post_init__(self) -> None:
        """功能：在配置创建时检查数值与几何限制，不构造模型。

        参数：
            无额外参数；读取 TrainConfig 的字段。
        返回：
            None；非法配置抛出 ValueError。
        """
        if self.mode not in ("overfit", "random"):
            raise ValueError("mode must be overfit or random")
        if self.batch_size <= 0 or self.num_samples < self.batch_size:
            raise ValueError("num_samples must be at least positive batch_size")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.num_steps < 2 or not 0 < self.beta_start <= self.beta_end < 1:
            raise ValueError("Invalid diffusion schedule")
        if min(self.image_size, self.in_channels, self.patch_size) <= 0:
            raise ValueError("image_size, in_channels, patch_size must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("patch_size must divide image_size")
        if (
            self.hidden_size < 2
            or self.hidden_size % 2
            or self.num_heads <= 0
            or self.hidden_size % self.num_heads
            or self.depth <= 0
            or self.mlp_ratio <= 0
        ):
            raise ValueError("Invalid Transformer configuration")


@dataclass(frozen=True)
class TrainingBatch:
    """一次更新的 CPU 目标：x_t [B,C,S,S]、t [B]、target_noise [B,C,S,S]。

    x_t/target_noise 为 float32，t 为 int64。overfit 在所有更新中复用这三项。
    """

    x_t: Tensor  # M2 前向加噪得到的输入，不是干净图片。
    t: Tensor  # 每张图片对应的离散时间步，范围为 [0,T)。
    target_noise: Tensor  # 加噪时实际使用的 epsilon，也是 MSE 的监督目标。


@dataclass
class TrainingState:
    """可延续的训练进度；step 统计已经成功完成的优化器更新次数。"""

    step: int = 0  # 新实验从 0 开始；每次 optimizer.step 成功后加 1。
    losses: list[float] = field(default_factory=list)  # 每步更新前的 MSE，长度等于 step。
    fixed_batch: TrainingBatch | None = None  # overfit 的固定目标；random 保持 None。


def make_training_batch(
    x_0: Tensor,
    schedule: DiffusionSchedule,
    *,
    t: Tensor | None = None,
    noise: Tensor | None = None,
) -> TrainingBatch:
    """功能：复用 M2，为一组干净图片构造噪声预测任务。

    参数：
        x_0：CPU float32 干净图片 [B,C,S,S]；通常归一化到 [-1,1]。
        schedule：CPU 扩散日程，长度 T。
        t：可选的 CPU int64 [B] 时间步；为 None 时每张图独立均匀采样。
        noise：可选的 CPU float32 同形状噪声；为 None 时重新采样标准高斯噪声。
    返回：
        TrainingBatch，包含 CPU 上的 x_t、t 和实际加噪使用的 target_noise。
        同时指定 t/noise 可构造完全固定的调试目标；不修改 x_0。
    """
    if x_0.device.type != "cpu" or schedule.betas.device.type != "cpu":
        raise ValueError("M4 samples training batches on CPU")
    # TODO M4-1：导入并复用 M2 的 sample_timesteps 与 q_sample。
    # t=None 时才抽取时间步；q_sample 返回的 epsilon 才是监督目标。
    # 不要使用 x_0 作为 target，也不要为了固定目标在每次更新中重新设置种子。
    if t is None:
        t = sample_timesteps(x_0.shape[0], len(schedule.betas), device="cpu")

    x_t, target_noise = q_sample(x_0, t, schedule, noise=noise)
    return TrainingBatch(x_t=x_t, t=t, target_noise=target_noise)


def noise_prediction_loss(model: nn.Module, x_t: Tensor, t: Tensor, target_noise: Tensor) -> Tensor:
    """功能：计算模型预测噪声与实际加噪噪声之间的均方误差。

    参数：
        model：支持 model(x_t,t) 的噪声预测器；由调用方设置 train/eval 模式。
        x_t：float32 [B,C,S,S] 带噪图片，与模型在同一设备。
        t：int64 [B] 时间步，与 x_t 同设备。
        target_noise：float32 [B,C,S,S] 实际噪声，形状与 x_t 完全相同。
    返回：
        有梯度连接的零维 Tensor，对 batch、通道和全部像素取平均。
        此函数不清梯度、不更新参数，不返回 float 或 detach 后的 loss。
    """
    # 先校验再前向：形状/设备不匹配时尽早失败，避免白跑一次模型。
    if x_t.dtype != torch.float32:
        raise ValueError("x_t must be float32")
    if t.dtype != torch.int64 or t.shape != (x_t.shape[0],):
        raise ValueError("t must be int64 [B]")
    if t.device != x_t.device:
        raise ValueError("t must be on the same device as x_t")
    if target_noise.shape != x_t.shape:
        raise ValueError("target_noise must match x_t shape; no broadcasting")
    if target_noise.dtype != x_t.dtype or target_noise.device != x_t.device:
        raise ValueError("target_noise must match x_t dtype and device")

    # TODO M4-2：前向得到 epsilon_pred，确认输出与 target_noise 完全同形状。
    epsilon_pred = model(x_t, t)
    if epsilon_pred.shape != target_noise.shape:
        raise ValueError("model output shape must match target_noise shape")

    # 使用均值 MSE；可以导入 torch.nn.functional，不能把 target 换成 x_0。
    # 对 batch、通道、像素全部取平均；返回 0-dim Tensor，保持计算图。
    # 不对预测做 clamp/tanh，也不要在 backward 前调用 loss.item()。
    return F.mse_loss(epsilon_pred, target_noise, reduction="mean")


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TrainingBatch,
    *,
    device: str | torch.device = "cpu",
) -> float:
    """功能：执行一次有数值检查的梯度更新。

    参数：
        model：已经移动到 device 的噪声预测器。
        optimizer：绑定该模型参数的优化器，本阶段使用 AdamW。
        batch：CPU 上的输入、时间步与目标；在本函数移动到 device。
        device：计算设备 cpu 或 mps；不得自动回退到另一个设备。
    返回：
        本次更新前的 MSE，类型为 Python float，便于记录曲线。
        loss 或梯度含 NaN/Inf 时抛出 RuntimeError，并且不能更新参数。
    """
    # TODO M4-3：model.train()；移动 batch 中的三个张量；清除上一轮梯度。
    model.train()
    x_t = batch.x_t.to(device)
    t = batch.t.to(device)
    target_noise = batch.target_noise.to(device)

    # 顺序：zero_grad(set_to_none=True) → noise_prediction_loss → backward → step。
    optimizer.zero_grad(set_to_none=True)

    # 在 backward 前检查 loss 有限；在 step 前检查全部可训练参数的梯度存在且有限。
    loss = noise_prediction_loss(model, x_t, t, target_noise)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss {loss.detach().item()}; parameters not updated")

    loss.backward()

    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    if any(g is None or not torch.isfinite(g).all() for g in gradients):
        # 抛出前不调用 optimizer.step()，因此参数与优化器状态都保持不变。
        raise RuntimeError("missing or non-finite gradient; parameters not updated")

    optimizer.step()

    # 参数更新后才将 loss.detach().item() 返回；不要把零梯度累积成多次更新。
    return loss.detach().item()


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: Tensor,
    schedule: DiffusionSchedule,
    config: TrainConfig,
    *,
    steps: int,
    device: str | torch.device = "cpu",
    state: TrainingState | None = None,
) -> TrainingState:
    """功能：在固定小数据集上追加若干次训练更新，支持两种目标采样模式。

    参数：
        model：已在 device 上的 TinyDiT；其配置与 config 对齐。
        optimizer：对应的优化器；恢复时必须已经加载其状态。
        images：CPU float32 [num_samples,C,S,S]，固定顺序的干净图片子集。
        schedule：CPU 日程，其参数与 config 一致。
        config：模型、日程、优化器及实验配置，见 TrainConfig 各字段注释。
        steps：本次额外完成的优化器更新次数，必须为正；不是累计目标步数。
        device：执行模型计算的设备；图片选择和目标采样仍在 CPU。
        state：已有进度或 None；None 表示新实验，不能跨调用重复使用默认实例。
    返回：
        更新后的 TrainingState；step 与 losses 的长度同步增长 steps。
        overfit 保留同一 fixed_batch，random 的 fixed_batch 为 None。
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    expected = (config.num_samples, config.in_channels, config.image_size, config.image_size)
    if tuple(images.shape) != expected or images.device.type != "cpu":
        raise ValueError("images must be the configured CPU subset")
    if images.dtype != torch.float32 or schedule.betas.device.type != "cpu":
        raise ValueError("Use float32 images and a CPU schedule")
    if state is not None and (state.step < 0 or len(state.losses) != state.step):
        raise ValueError("state.step must equal len(state.losses)")
    # TODO M4-4：None 时创建 TrainingState；已有 state 时接着更新，不能重置 step。
    # overfit：仅在 fixed_batch=None 时，对 images[:batch_size] 构造一次 batch。
    #   后续更新和恢复后都复用这份 x_t/t/target_noise，不重新抽样。
    # random：每步用 CPU torch.randint 有放回抽取 batch_size 张图片；
    #   再调用 make_training_batch 重新采样时间步和噪声，不复用 fixed_batch。
    # 两种模式都调用 train_step；成功后 step += 1，losses 追加一个 float。
    # 函数内部不要设置随机种子，不要直接实现第二份单步更新逻辑。
    if state is None:
        state = TrainingState()

    if config.mode == "overfit":
        # 固定目标只在第一次（或恢复后缺失时）构造，后续更新复用同一份。
        if state.fixed_batch is None:
            state.fixed_batch = make_training_batch(images[: config.batch_size], schedule)
        fixed_batch: TrainingBatch | None = state.fixed_batch
    else:
        if state.fixed_batch is not None:
            raise ValueError("random mode must not carry a fixed_batch")
        fixed_batch = None

    for _ in range(steps):
        if fixed_batch is not None:
            batch = fixed_batch
        else:
            indices = torch.randint(0, config.num_samples, (config.batch_size,))
            batch = make_training_batch(images[indices], schedule)

        loss = train_step(model, optimizer, batch, device=device)
        state.step += 1
        state.losses.append(loss)

    return state


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    state: TrainingState,
) -> None:
    """功能：保存能继续训练的 checkpoint，而非仅导出网络权重。

    参数：
        path：checkpoint 文件路径；在写入前创建父目录。
        model：要保存 state_dict 的网络；允许在 CPU 或 MPS 上。
        optimizer：保存其 state_dict，包括 AdamW 的动量与更新计数。
        config：用 dataclasses.asdict 转成普通字典并保存。
        state：保存 step、完整 losses，以及 overfit 的 fixed_batch。
    返回：
        None。格式见 docs/M4.md；使用普通字典、Tensor 和基本类型，
        便于 torch.load(..., weights_only=True) 加载，不序列化整个模型对象。
    """
    if state.step < 0 or len(state.losses) != state.step:
        raise ValueError("state.step must equal len(state.losses)")
    # TODO M4-5：payload 至少包括 format_version=1、config、model、optimizer、
    # step、losses、fixed_batch、rng_state（torch.get_rng_state()，CPU 字节张量）。
    # fixed_batch=None 或字典 {x_t,t,target_noise}，其中三个张量保存为 CPU 副本。
    # 用 state_dict 保存模型/优化器，不把 nn.Module 或 dataclass 实例塞入 payload。
    # 保存随机状态本身不能消耗随机数。创建父目录后用 torch.save 写入 path。
    # 1. 检查目录
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fixed_batch = None
    if state.fixed_batch is not None:
        fixed_batch = {
            "x_t": state.fixed_batch.x_t.detach().cpu(),
            "t": state.fixed_batch.t.detach().cpu(),
            "target_noise": state.fixed_batch.target_noise.detach().cpu(),
        }
    elif config.mode == "overfit" and state.step > 0:
        raise ValueError("overfit state with step > 0 must carry a fixed_batch")

    # 2. 构建 payload
    payload = {
        "format_version": 1,
        "config": dataclasses.asdict(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": state.step,
        "losses": list(state.losses),
        "fixed_batch": fixed_batch,
        "rng_state": torch.get_rng_state(),
    }

    # 3. 保存 payload
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_config: TrainConfig,
) -> TrainingState:
    """功能：校验 checkpoint 配置，恢复网络、优化器、进度及 CPU 随机状态。

    参数：
        path：由本项目保存、来源可信的 checkpoint 文件。
        model：已按 expected_config 创建并移动到目标设备的网络。
        optimizer：已绑定 model 参数的优化器；必须恢复其状态后再继续训练。
        expected_config：当前配置；要求与保存配置完全相同，避免静默更换实验。
    返回：
        TrainingState；fixed_batch 恢复为 CPU TrainingBatch 或 None。
        配置、版本或进度不合法时抛出 ValueError；配置校验失败不得修改模型、
        优化器或 RNG。CPU 上应能复现同一随机序列的下一次更新。
    """
    # TODO M4-6：torch.load 使用 map_location="cpu", weights_only=True。
    # 先校验 format_version、config、step 与 losses，之后才修改运行状态。
    # overfit 且 step>0 必须有固定 batch；random 不应带有固定 batch。
    # model.load_state_dict(..., strict=True)；optimizer.load_state_dict(...)。
    # 将 fixed_batch 字典还原为 TrainingBatch；CPU rng_state 留在 CPU。
    # 最后 torch.set_rng_state(...)，避免重建模型等操作改变恢复后的随机序列。
    # 不在此处 manual_seed，也不要用新优化器替换调用方传入的 optimizer。
    # 1. 加载 checkpoint
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)

    # 2. 校验格式版本
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format version: {payload.get('format_version')}")

    # 3. 校验配置；字段缺失或多余都视为损坏，而不是静默改用默认值。
    saved_config = payload.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("Checkpoint config is missing or malformed")
    try:
        loaded_config = TrainConfig(**saved_config)
    except TypeError as error:
        raise ValueError("Checkpoint config is missing or malformed") from error
    if loaded_config != expected_config:
        raise ValueError("Checkpoint config does not match expected config")

    # 4. 校验 step 与 losses
    step = payload.get("step")
    losses = payload.get("losses")
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("Checkpoint step must be a nonnegative int")
    if not isinstance(losses, list) or len(losses) != step:
        raise ValueError("Checkpoint step and losses are inconsistent")

    # 5. 校验 fixed_batch；全部校验通过前不改动模型、优化器或 RNG。
    fixed_batch_data = payload.get("fixed_batch")
    if expected_config.mode == "overfit" and step > 0 and fixed_batch_data is None:
        raise ValueError("Overfit mode with step > 0 must have a fixed batch")
    if expected_config.mode == "random" and fixed_batch_data is not None:
        raise ValueError("Random mode should not have a fixed batch")
    fixed_batch = None
    if fixed_batch_data is not None:
        if not isinstance(fixed_batch_data, dict) or set(fixed_batch_data) != {
            "x_t",
            "t",
            "target_noise",
        }:
            raise ValueError("Checkpoint fixed_batch is malformed")
        fixed_batch = TrainingBatch(
            x_t=fixed_batch_data["x_t"].cpu(),
            t=fixed_batch_data["t"].cpu(),
            target_noise=fixed_batch_data["target_noise"].cpu(),
        )

    # 6. 校验随机状态
    rng_state = payload.get("rng_state")
    if not isinstance(rng_state, Tensor) or rng_state.dtype != torch.uint8:
        raise ValueError("Checkpoint rng_state must be a uint8 CPU tensor")

    # 7. 恢复模型和优化器状态
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])

    # 8. 最后恢复 CPU 随机状态，避免上面的加载消耗随机数影响序列。
    torch.set_rng_state(rng_state)

    # 9. 返回 TrainingState
    return TrainingState(step=step, losses=list(losses), fixed_batch=fixed_batch)
