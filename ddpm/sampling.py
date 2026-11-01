"""M5 练习：DDPM 反向采样；4 处核心计算由学习者按 TODO 填写。

时间步沿用 M2 的 0 基下标：t=0 是最后一次反向更新，t=T-1 最接近纯噪声。
反向循环执行 T 次，依次从 x_{T-1} 得到 x_{T-2} ... 直到 t=0 的更新给出最终图片。
形状记号：B=批大小，C=通道数，S=图片边长，T=扩散日程长度。
"""

from collections.abc import Sequence
from math import sqrt

import torch
from torch import Tensor, nn

from ddpm.diffusion import DiffusionSchedule


def _validate_step(t: int, schedule: DiffusionSchedule) -> None:
    """功能：检查反向步骤下标，阻止 Python 负索引和越界。

    参数：
        t：待检查的步骤；必须是 Python int，不能是 bool 或 Tensor。
        schedule：提供总步数 T 的日程。
    返回：
        None；非法时抛出 ValueError。
    """
    if isinstance(t, bool) or not isinstance(t, int):
        raise ValueError("t must be a Python int")
    if not 0 <= t < len(schedule.betas):
        raise ValueError("t must lie in [0, T-1]")


def _validate_image(value: Tensor, name: str) -> None:
    """功能：检查图片张量的基本形状和类型，不检查数值范围。

    参数：
        value：待检查张量，应为 [B,C,S,S] float32 且 B>0。
        name：错误信息中使用的变量名。
    返回：
        None；非法时抛出 ValueError。
    """
    if value.ndim != 4 or value.shape[0] <= 0:
        raise ValueError(f"{name} must have shape [B,C,S,S] with B > 0")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must be float32")


def estimate_x0(
    x_t: Tensor,
    t: int,
    predicted_noise: Tensor,
    schedule: DiffusionSchedule,
) -> Tensor:
    """功能：由带噪图片和预测噪声估计干净图片 x0_hat，不做裁剪。

    参数：
        x_t：float32 [B,C,S,S] 当前带噪图片；可在 CPU 或 MPS。
        t：本次反向步骤的整数下标，范围 [0,T-1]；整个 batch 共用同一步。
        predicted_noise：模型给出的 epsilon 预测；形状、类型、设备必须与 x_t 相同。
        schedule：扩散日程；系数会转成 Python float，日程留在 CPU 也能参与计算。
    返回：
        x0_hat：与 x_t 同形状、同设备、同类型的张量，保留梯度连接。
        估计值可能超出 [-1,1]，本函数和调用方都不做 clamp。
    """
    _validate_step(t, schedule)
    _validate_image(x_t, "x_t")
    if (
        predicted_noise.shape != x_t.shape
        or predicted_noise.dtype != x_t.dtype
        or predicted_noise.device != x_t.device
    ):
        raise ValueError("predicted_noise must match x_t shape, dtype, and device")

    # TODO M5-1：由 q_sample 的逆关系估计干净图片。
    # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    # 已知 x_t 和 epsilon_hat 时解出 x0_hat：
    # 用 float(schedule.alpha_bars[t]) 取系数，避免 CPU 日程与 MPS 张量设备不匹配。
    alpha_bar_t = float(schedule.alpha_bars[t])
    x0_hat = (x_t - sqrt(1 - alpha_bar_t) * predicted_noise) / sqrt(alpha_bar_t)
    # 不要 clamp；不要把 alpha_bar_t 和 alpha_t 混用。
    return x0_hat


def posterior_mean(
    x_t: Tensor,
    t: int,
    predicted_noise: Tensor,
    schedule: DiffusionSchedule,
) -> Tensor:
    """功能：计算反向分布 p(x_{t-1}|x_t) 的均值，尚未加入随机噪声。

    参数：
        x_t：float32 [B,C,S,S] 当前带噪图片。
        t：本次反向步骤的整数下标，范围 [0,T-1]。
        predicted_noise：模型预测的 epsilon，形状与 x_t 相同。
        schedule：扩散日程；需要 betas、alphas、alpha_bars。
    返回：
        mu：[B,C,S,S] 张量，与 x_t 同设备，保留梯度连接。
        使用未裁剪的 x0_hat 做后验插值；t=0 时 mu 等于 x0_hat。
    """
    _validate_step(t, schedule)

    # TODO M5-2：先调用 M5-1 得到 x0_hat，再按后验分布组合 x0_hat 与 x_t。
    # 记 alpha_bar_prev = alpha_bar_{t-1}，并规定 alpha_bar_{-1}=1：
    #   mu = sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t) * x0_hat
    #      + sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x_t
    # t=0 时没有前一项日程系数，alpha_bar_prev 必须取 1.0，不能读 alpha_bars[-1]。
    # 等价写法 mu = (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_t)；
    # 这里要求用 x0_hat 的写法，便于对照公式。不要添加噪声，也不要 clamp。
    x0_hat = estimate_x0(x_t, t, predicted_noise, schedule)

    alpha_bar_prev = float(schedule.alpha_bars[t - 1]) if t > 0 else 1.0
    beta_t = float(schedule.betas[t])
    alpha_t = float(schedule.alphas[t])
    alpha_bar_t = float(schedule.alpha_bars[t])
    mu = (
        sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t) * x0_hat
        + sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x_t
    )
    return mu


def p_sample(
    model: nn.Module,
    x_t: Tensor,
    t: int,
    schedule: DiffusionSchedule,
    *,
    noise: Tensor | None = None,
) -> Tensor:
    """功能：执行一次反向更新，由 x_t 得到 x_{t-1}。

    参数：
        model：噪声预测器，调用方式为 model(x_t, t_batch)；须已在 x_t 的设备上。
        x_t：float32 [B,C,S,S] 当前带噪图片。
        t：本次步骤的整数下标，范围 [0,T-1]；会转成 [B] int64 传给模型。
        schedule：扩散日程。
        noise：可选的标准高斯噪声 z，形状与 x_t 相同；为 None 时在 CPU 采样再搬到
            x_t 设备，使同一 CPU 种子在不同设备上得到相同噪声。
    返回：
        x_{t-1}：与 x_t 同形状、同设备、同类型的张量。
        t=0 时返回确定性的后验均值，不添加噪声；t>0 时加固定后验方差的噪声。
    """
    _validate_step(t, schedule)
    _validate_image(x_t, "x_t")
    if noise is not None and (
        noise.shape != x_t.shape or noise.dtype != x_t.dtype or noise.device != x_t.device
    ):
        raise ValueError("noise must match x_t shape, dtype, and device")

    # TODO M5-3：一次完整的反向更新。
    # 1. 用 torch.full 构造 [B] int64 的 t_batch，设备与 x_t 相同；整个 batch 同一步。
    t_batch = torch.full(size=(x_t.shape[0],), fill_value=t, dtype=torch.int64, device=x_t.device)
    # 2. eps_hat = model(x_t, t_batch)，检查输出形状与 x_t 一致。
    eps_hat = model(x_t, t_batch)
    if eps_hat.shape != x_t.shape:
        raise ValueError("model output shape must match x_t shape")
    # 3. mu = posterior_mean(x_t, t, eps_hat, schedule)。
    mu = posterior_mean(x_t, t, eps_hat, schedule)
    # 4. t=0：直接返回 mu（忽略传入的 noise），这是最后一步的零方差。
    #    t>0：取固定后验方差 beta_tilde_t = beta_t*(1-alpha_bar_{t-1})/(1-alpha_bar_t)，
    #    其中 alpha_bar_{-1}=1；z 为 noise 或在 CPU 上 randn 后搬到 x_t.device；
    #    返回 mu + sqrt(beta_tilde_t) * z。
    # 本函数不负责 model.eval()、no_grad 或设置随机种子，这些由采样循环和调用方负责。
    if t == 0:
        return mu
    else:
        beta_tilde_t = (
            float(schedule.betas[t])
            * (1 - float(schedule.alpha_bars[t - 1]))
            / (1 - float(schedule.alpha_bars[t]))
        )
        if noise is None:
            z = torch.randn_like(x_t, device="cpu").to(x_t.device)
        else:
            z = noise
        return mu + sqrt(beta_tilde_t) * z


def p_sample_loop(
    model: nn.Module,
    shape: Sequence[int],
    schedule: DiffusionSchedule,
    *,
    device: str | torch.device = "cpu",
    snapshot_steps: Sequence[int] | None = None,
) -> tuple[Tensor, dict[int, Tensor]]:
    """功能：从标准高斯噪声出发，按 t=T-1,...,0 依次反向更新。

    参数：
        model：噪声预测器，须已移动到 device；本函数会把它切到 eval()。
        shape：初始噪声形状 [B,C,S,S]，四个正整数；C/S 必须与网络匹配。
        schedule：扩散日程。
        device：执行计算与存放最终图片的设备。
        snapshot_steps：可选的下标集合；保存「执行第 t 步之前」的 x_t 的 CPU 副本。
            例如 T=1000 时传 [999, 750, 500, 250, 0] 可画去噪过程。
    返回：
        (image, snapshots)：image 是 device 上的最终结果；snapshots 是
        {t: x_t.cpu()} 字典，键按实际反向顺序插入。初始噪声由 CPU randn 生成后
        搬到 device，因此调用方只需设置 CPU 种子即可复现整条轨迹。
        本函数不设置种子，也不检查输出是否有限；两次调用之间的模式与梯度行为
        由 eval() 和 torch.no_grad() 保证。
    """
    shape = tuple(shape)
    if len(shape) != 4 or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape
    ):
        raise ValueError("shape must be four positive integers [B,C,S,S]")
    wanted: set[int] = set()
    if snapshot_steps is not None:
        for step in snapshot_steps:
            _validate_step(step, schedule)
            wanted.add(step)

    # TODO M5-4：完整采样循环。
    # 1. x_T ~ N(0,I)：先在 CPU 用 torch.randn(shape) 生成 float32，再 .to(device)。
    x_T = torch.randn(shape, dtype=torch.float32, device="cpu").to(device)
    # 2. model.eval() 并进入 torch.no_grad()，采样阶段不需要梯度。
    model.eval()
    # 3. for t in range(T-1, -1, -1)：若 t 在 wanted 中，先保存当前 x 的 CPU 副本，
    #    再调用 p_sample 得到下一步；循环结束时得到最终图片。
    with torch.no_grad():
        x_t = x_T
        snapshots: dict[int, Tensor] = {}
        for t in range(len(schedule.betas) - 1, -1, -1):
            if t in wanted:
                snapshots[t] = x_t.cpu()
            x_t = p_sample(model, x_t, t, schedule)
    # 4. 返回 (image, snapshots)；不要在函数内部 manual_seed，也不要把 x 做 clamp。
    return x_t, snapshots
