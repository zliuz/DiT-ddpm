"""M2 练习：按 TODO 1–4 完成前向加噪；这里不包含参考答案。

形状记号：T 是扩散总步数，B 是本批图片数，C 是通道数，H/W 是图片高度/宽度。
时间步 t 是数组下标，范围 [0, T-1]；t=0 对应第一次加噪，不对应干净图片。
"""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class DiffusionSchedule:
    """三个长度为 T 的 float32 张量；下标 t=0 表示第一次加噪。"""

    betas: Tensor  # [T]：每个局部加噪步骤新增的噪声方差 beta，不是噪声标准差。
    alphas: Tensor  # [T]：单步信号方差系数 alpha=1-beta；振幅系数还需开平方。
    alpha_bars: Tensor  # [T]：从第 0 项到当前项的 alpha 累乘，描述累计信号方差系数。


def make_linear_schedule(
    num_steps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    *,
    device: str | torch.device = "cpu",
) -> DiffusionSchedule:
    """返回 beta、alpha 和 alpha 的累计乘积，均为 [T]。

    参数：
        num_steps：总扩散步数 T，至少为 2；默认 1000，即下标 0–999。
        beta_start：第一个步骤的噪声方差，默认 0.0001；必须大于 0。
        beta_end：最后一个步骤的局部噪声方差，默认 0.02；不是累计噪声量。
            要求 beta_start <= beta_end < 1，避免产生无效的 alpha。
        device：日程张量的存放设备，如 "cpu"、"mps" 或 torch.device 对象。
            默认 CPU；后续提取系数时应与时间步、图片位于同一设备。
            星号 * 表示 device 只能按关键字传入，避免与 beta 参数混淆。

    返回：
        DiffusionSchedule：包含三个 float32 张量，每个长度均为 num_steps。
    """
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if not 0 < beta_start <= beta_end < 1:
        raise ValueError("Expected 0 < beta_start <= beta_end < 1")

    # TODO 1：构造噪声日程。
    # 1. beta 从 beta_start 线性增加至 beta_end，包含两个端点，共 num_steps 个值。
    # linspace 的前三个参数依次为起点、终点、元素个数；dtype/device 指定类型与设备。
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32, device=device)
    # 2. alpha_t = 1 - beta_t。
    alphas = 1 - betas
    # 3. alpha_bar_t 是 alpha 从下标 0 到 t 的乘积；注意不是求和，也不是全局乘积。
    # dim=0：沿唯一的时间轴逐项累乘；dtype 指定计算类型，设备自然继承自 alphas。
    # API 提醒：torch.cumprod 不接受 device 关键字；请自行检查下一行的参数。
    alpha_bars = torch.cumprod(alphas, dim=0, dtype=torch.float32)
    # 将三个张量装入 DiffusionSchedule 返回，不要额外在数组开头插入 1。
    return DiffusionSchedule(betas=betas, alphas=alphas, alpha_bars=alpha_bars)


def extract(values: Tensor, t: Tensor, x_shape: tuple[int, ...]) -> Tensor:
    """从 [T] 系数表按 [B] 时间步提取系数，返回 [B, 1, ..., 1]。

    参数：
        values：长度为 T 的一维系数表，例如 schedule.alpha_bars。
            它描述所有时间步，而不是一个 batch 的图片。
        t：形状 [B] 的 int64 时间步张量；每个值必须在 [0, T-1]。
            第 i 项指定第 i 张图片的时间步，允许不同图片使用相同时间步。
        x_shape：待加噪张量的完整形状，通常传 x_0.shape，即 [B,C,H,W]。
            这里只需要形状，不需要图片内容；用于决定广播所需的单例维度个数。

    返回：
        与 values 同类型、同设备的系数张量；图片输入时为 [B,1,1,1]。
        其余维度为 1，让每张图片的系数能广播到该图片的所有通道和像素。
    """
    if values.ndim != 1:
        raise ValueError("values must have shape [T]")
    if t.ndim != 1 or t.dtype != torch.int64:
        raise ValueError("t must be an int64 tensor with shape [B]")
    if not x_shape or x_shape[0] <= 0 or t.shape[0] != x_shape[0]:
        raise ValueError("t and x_shape must have the same positive batch size")
    if values.device != t.device:
        raise ValueError("values and t must be on the same device")
    if t.min().item() < 0 or t.max().item() >= len(values):
        raise ValueError("t must lie in [0, T-1]")

    # 为 batch 中每张图片提取自己的系数，并准备广播。
    # 例如 t=[2, 0, 2]，就应按此顺序取出 values 中的第 2、0、2 项。
    coefficients = values[t]  # [B]：按 t 的原顺序提取，重复下标也会保留。
    # 索引结果是 [B]；图片是 [B, C, H, W]，所以要将系数变成 [B, 1, 1, 1]。
    # 提示：使用索引或 gather，再 reshape；单例维度个数由 len(x_shape) 决定。
    # 不要复制成完整图片，也不要把每张图片的时间步错误地应用到宽度轴。
    # x_shape[0] 是 B；len(x_shape)-1 是除 batch 轴外的轴数，四维图片时等于 3。
    return coefficients.reshape((x_shape[0],) + (1,) * (len(x_shape) - 1))


def sample_timesteps(
    batch_size: int,
    num_steps: int,
    *,
    device: str | torch.device = "cpu",
) -> Tensor:
    """为每张图片独立均匀采样一个时间步，返回 [B] int64。

    参数：
        batch_size：当前 batch 的图片数 B，必须为正；通常取 x_0.shape[0]。
            最后一批可能不足设定的批大小，应使用实际图片数。
        num_steps：可选时间步的总数 T，必须为正；与所用日程的长度一致。
        device：输出时间步张量的设备，应与图片和日程张量一致；默认 CPU。

    返回：
        [batch_size] 的整数张量，每个元素独立均匀取自 0–num_steps-1。
        相同的时间步可以重复，独立采样不代表时间步必须互不相同。
    """
    if batch_size <= 0 or num_steps <= 0:
        raise ValueError("batch_size and num_steps must be positive")

    # TODO 3：采样 batch_size 个整数，范围为 0 到 num_steps-1。
    # 提示：查看 torch.randint，尤其注意 high 是否包含在取值范围内。
    # 不要在此函数内部重置随机种子，种子由调用者设置。
    # randint：low=0 包含，high=num_steps 不包含；(batch_size,) 指定一维输出形状。
    # dtype=int64 是用于索引系数表的整数类型，不能使用 float32 时间步。
    t = torch.randint(0, num_steps, (batch_size,), dtype=torch.int64, device=device)
    return t


def q_sample(
    x_0: Tensor,
    t: Tensor,
    schedule: DiffusionSchedule,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """返回 (x_t, epsilon)，形状均为 [B, C, H, W]；不修改输入。

    参数：
        x_0：干净图片张量，float32，形状 [B,C,H,W]。
            本项目 MNIST 通常为 [B,1,28,28]，像素已归一化到 [-1,1]。
        t：每张图片对应的时间步，[B] int64，取值为 [0,T-1]。
            它指定目标噪声程度，不表示此函数内部需要执行多少次循环。
        schedule：覆盖全部 T 个时间步的 DiffusionSchedule，使用其 alpha_bars。
            不同 batch 可复用同一个日程，不需要为每张图片重新生成。
        noise：可选的固定 epsilon，形状、类型和设备必须与 x_0 一致。
            为 None 时生成标准正态噪声；传入张量时直接使用，便于对照和测试。

    返回：
        x_t：所选时间步的带噪图片，形状和类型同 x_0，像素可能超出 [-1,1]。
        epsilon：本次实际使用的噪声，之后可作为噪声预测模型的训练目标。
    """
    if x_0.ndim != 4 or x_0.dtype != torch.float32:
        raise ValueError("x_0 must be float32 with shape [B, C, H, W]")
    if x_0.device != t.device:
        raise ValueError("x_0 and t must be on the same device")
    if noise is not None and (
        noise.shape != x_0.shape or noise.dtype != x_0.dtype or noise.device != x_0.device
    ):
        raise ValueError("noise must match x_0 in shape, dtype, and device")

    # TODO 4：一次直接采样 q(x_t | x_0)，不需要循环模拟前面所有步骤。
    # 1. noise=None 时生成与 x_0 同形状/设备/类型的标准正态噪声；否则用给定 noise。
    if noise is None:
        # randn_like：每个元素采自 N(0,1)，并继承 x_0 的形状、dtype 和 device。
        noise = torch.randn_like(x_0)
    # 2. 用 extract 从 schedule.alpha_bars 中提取当前 batch 的累计信号比例。
    alpha_bar_t = extract(schedule.alpha_bars, t, x_0.shape)  # [B,1,1,1]，不是 [T]。
    # 3. 依据公式组合：x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon。
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    # 4. 返回 (x_t, 实际使用的 epsilon)，后者在 M4 中会成为训练目标。
    # 提示：torch.randn_like、torch.sqrt。不要原地修改输入，也不要 clamp x_t。
    # 注意噪声系数是 sqrt(1-alpha_bar_t)，不是 1-sqrt(alpha_bar_t)。
    return x_t, noise
