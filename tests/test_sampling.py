"""M5 验收：直接测试学习者的反向采样实现；TODO 未填写时应失败，不跳过测试。"""

import math

import pytest
import torch
from torch import Tensor, nn

from ddpm.diffusion import DiffusionSchedule, make_linear_schedule, q_sample
from ddpm.model import TinyDiT
from ddpm.sampling import estimate_x0, p_sample, p_sample_loop, posterior_mean


def _schedule() -> DiffusionSchedule:
    """功能：构造短日程以加快手算与循环测试。参数：无。返回：T=8 的日程。"""
    return make_linear_schedule(num_steps=8, beta_start=0.01, beta_end=0.2)


def _images() -> Tensor:
    """功能：构造超出 [-1,1] 的固定图片，顺带检查实现是否偷偷裁剪。参数：无。返回：张量。"""
    return torch.linspace(-2.5, 2.5, 2 * 1 * 4 * 4).reshape(2, 1, 4, 4)


def _noised(schedule: DiffusionSchedule, t: int) -> tuple[Tensor, Tensor, Tensor]:
    """功能：用 M2 的 q_sample 构造 x_t 及其真实噪声。参数：日程与时间步。返回：三元组。"""
    torch.manual_seed(0)
    x_0 = _images()
    noise = torch.randn_like(x_0)
    t_batch = torch.full((x_0.shape[0],), t, dtype=torch.int64)
    x_t, _ = q_sample(x_0, t_batch, schedule, noise=noise)
    return x_0, x_t, noise


class _RecordingModel(nn.Module):
    """返回固定输出并记录调用参数；用于隔离采样公式，不训练网络。"""

    def __init__(self, output: Tensor | None = None, value: float = 0.0) -> None:
        """功能：保存固定输出或常数。参数：output 为张量，value 为常数。返回：None。"""
        super().__init__()
        self.output = output
        self.value = value
        self.t_calls: list[Tensor] = []
        self.training_flags: list[bool] = []
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """功能：记录 t 与训练模式，返回固定噪声。参数：x_t、t。返回：噪声张量。"""
        self.t_calls.append(t.detach().clone())
        self.training_flags.append(self.training)
        return self.output if self.output is not None else torch.full_like(x_t, self.value)


@pytest.mark.parametrize("t", [0, 3, 7])
def test_estimate_x0_recovers_clean_image(t: int) -> None:
    """功能：用真实 x_t 与真实噪声应精确还原本阶段 x0。参数：时间步。返回：None。"""
    schedule = _schedule()
    x_0, x_t, noise = _noised(schedule, t)
    estimated = estimate_x0(x_t, t, noise, schedule)
    torch.testing.assert_close(estimated, x_0, atol=1e-5, rtol=1e-5)
    assert estimated.max() > 1  # 输入超出 [-1,1]，确认没有 clamp。


@pytest.mark.parametrize("t", [-1, 8])
def test_estimate_x0_rejects_invalid_step(t: int) -> None:
    """功能：负下标与越界都必须失败，不能利用 Python 负索引。参数：非法 t。返回：None。"""
    schedule = _schedule()
    x_t = torch.zeros(2, 1, 4, 4)
    with pytest.raises(ValueError):
        estimate_x0(x_t, t, torch.zeros_like(x_t), schedule)


def test_estimate_x0_rejects_broadcast_shape() -> None:
    """功能：噪声形状不一致时拒绝广播。参数：无。返回：None。"""
    schedule = _schedule()
    with pytest.raises(ValueError):
        estimate_x0(torch.zeros(2, 1, 4, 4), 0, torch.zeros(1, 1, 4, 4), schedule)


@pytest.mark.parametrize("t", [0, 1, 6])
def test_posterior_mean_matches_analytic_posterior(t: int) -> None:
    """功能：与 q(x_{t-1}|x_t,x_0) 的后验均值公式手算对照。参数：时间步。返回：None。"""
    schedule = _schedule()
    x_0, x_t, noise = _noised(schedule, t)
    actual = posterior_mean(x_t, t, noise, schedule)

    beta_t = schedule.betas[t].item()
    alpha_t = schedule.alphas[t].item()
    alpha_bar_t = schedule.alpha_bars[t].item()
    alpha_bar_prev = 1.0 if t == 0 else schedule.alpha_bars[t - 1].item()
    expected = (
        math.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t) * x_0
        + math.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * x_t
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_posterior_mean_last_step_is_x0_estimate() -> None:
    """功能：t=0 时后验均值应退化为未裁剪的 x0_hat。参数：无。返回：None。"""
    schedule = _schedule()
    x_0, x_t, noise = _noised(schedule, 0)
    torch.testing.assert_close(posterior_mean(x_t, 0, noise, schedule), x_0, atol=1e-5, rtol=1e-5)


def test_p_sample_last_step_ignores_noise() -> None:
    """功能：t=0 不加噪声，即使传入不同 z 也应返回同一结果。参数：无。返回：None。"""
    schedule = _schedule()
    x_0, x_t, noise = _noised(schedule, 0)
    model = _RecordingModel(output=noise)
    first = p_sample(model, x_t, 0, schedule, noise=torch.zeros_like(x_t))
    second = p_sample(model, x_t, 0, schedule, noise=torch.full_like(x_t, 7.0))
    assert torch.equal(first, second)
    torch.testing.assert_close(first, x_0, atol=1e-5, rtol=1e-5)


def test_p_sample_adds_fixed_posterior_variance_noise() -> None:
    """功能：t>0 时噪声系数应为 beta_tilde 的平方根。参数：无。返回：None。"""
    schedule = _schedule()
    t = 3
    _, x_t, noise = _noised(schedule, t)
    model = _RecordingModel(output=noise)
    first_z = torch.randn_like(x_t)
    second_z = torch.randn_like(x_t)
    first = p_sample(model, x_t, t, schedule, noise=first_z)
    second = p_sample(model, x_t, t, schedule, noise=second_z)

    alpha_bar_t = schedule.alpha_bars[t].item()
    alpha_bar_prev = schedule.alpha_bars[t - 1].item()
    beta_tilde = schedule.betas[t].item() * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
    torch.testing.assert_close(first - second, math.sqrt(beta_tilde) * (first_z - second_z))
    torch.testing.assert_close(
        first - math.sqrt(beta_tilde) * first_z, posterior_mean(x_t, t, noise, schedule)
    )


def test_p_sample_passes_batched_step_to_model() -> None:
    """功能：模型必须收到与 batch 对齐的 [B] int64 时间步。参数：无。返回：None。"""
    schedule = _schedule()
    _, x_t, noise = _noised(schedule, 5)
    model = _RecordingModel(output=noise)
    p_sample(model, x_t, 5, schedule, noise=torch.zeros_like(x_t))
    received = model.t_calls[0]
    assert received.dtype == torch.int64 and received.shape == (2,)
    assert torch.equal(received, torch.tensor([5, 5]))


def test_p_sample_samples_noise_on_cpu_for_seed_reproducibility() -> None:
    """功能：noise=None 时由 CPU 随机数驱动，同种子应完全复现。参数：无。返回：None。"""
    schedule = _schedule()
    _, x_t, noise = _noised(schedule, 4)
    model = _RecordingModel(output=noise)
    torch.manual_seed(123)
    first = p_sample(model, x_t, 4, schedule)
    torch.manual_seed(123)
    second = p_sample(model, x_t, 4, schedule)
    torch.manual_seed(124)
    third = p_sample(model, x_t, 4, schedule)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


@pytest.mark.parametrize("t", [-1, 8])
def test_p_sample_rejects_invalid_step(t: int) -> None:
    """功能：反向步骤越界时失败。参数：非法 t。返回：None。"""
    schedule = _schedule()
    x_t = torch.zeros(2, 1, 4, 4)
    with pytest.raises(ValueError):
        p_sample(_RecordingModel(), x_t, t, schedule)


def test_p_sample_rejects_broadcast_noise() -> None:
    """功能：传入噪声形状不一致时拒绝广播。参数：无。返回：None。"""
    schedule = _schedule()
    x_t = torch.zeros(2, 1, 4, 4)
    with pytest.raises(ValueError):
        p_sample(_RecordingModel(), x_t, 1, schedule, noise=torch.zeros(1, 1, 4, 4))


def test_loop_steps_down_from_t_minus_one_and_shapes() -> None:
    """功能：调用次数、方向、输出形状与快照键。参数：无。返回：None。"""
    schedule = _schedule()
    model = _RecordingModel(value=0.0)
    torch.manual_seed(0)
    image, snapshots = p_sample_loop(model, (2, 1, 4, 4), schedule, snapshot_steps=[7, 3, 0])
    assert [int(step[0]) for step in model.t_calls] == list(range(7, -1, -1))
    assert image.shape == (2, 1, 4, 4) and image.dtype == torch.float32
    assert set(snapshots) == {7, 3, 0}
    assert all(snapshot.shape == image.shape for snapshot in snapshots.values())
    assert all(snapshot.device.type == "cpu" for snapshot in snapshots.values())


def test_loop_initial_snapshot_is_cpu_seeded_noise() -> None:
    """功能：t=T-1 的快照应等于同种子生成 CPU 初始噪声。参数：无。返回：None。"""
    schedule = _schedule()
    torch.manual_seed(7)
    expected_initial = torch.randn(2, 1, 4, 4)
    torch.manual_seed(7)
    _, snapshots = p_sample_loop(
        _RecordingModel(value=0.0), (2, 1, 4, 4), schedule, snapshot_steps=[7]
    )
    assert torch.equal(snapshots[7], expected_initial)


def test_loop_is_reproducible_with_seed() -> None:
    """功能：相同种子得到完全相同的采样结果。参数：无。返回：None。"""
    schedule = _schedule()
    model = _RecordingModel(value=0.1)
    torch.manual_seed(42)
    first, _ = p_sample_loop(model, (2, 1, 4, 4), schedule)
    torch.manual_seed(42)
    second, _ = p_sample_loop(model, (2, 1, 4, 4), schedule)
    torch.manual_seed(43)
    third, _ = p_sample_loop(model, (2, 1, 4, 4), schedule)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_loop_uses_eval_and_no_grad() -> None:
    """功能：采样应切到 eval()、禁用梯度并保持输出无梯度。参数：无。返回：None。"""
    schedule = _schedule()
    model = _RecordingModel(value=0.1)
    model.train()
    image, snapshots = p_sample_loop(model, (2, 1, 4, 4), schedule, snapshot_steps=[1])
    assert model.training_flags and not any(model.training_flags)
    assert not model.training
    assert not image.requires_grad
    assert all(not snapshot.requires_grad for snapshot in snapshots.values())
    assert model.anchor.grad is None


def test_loop_rejects_invalid_shape_and_snapshot() -> None:
    """功能：拒绝错误形状与越界快照。参数：无。返回：None。"""
    schedule = _schedule()
    model = _RecordingModel()
    with pytest.raises(ValueError):
        p_sample_loop(model, (2, 4, 4), schedule)
    with pytest.raises(ValueError):
        p_sample_loop(model, (0, 1, 4, 4), schedule)
    with pytest.raises(ValueError):
        p_sample_loop(model, (1, 1, 4, 4), schedule, snapshot_steps=[8])


def test_loop_with_real_tiny_dit_produces_finite_images() -> None:
    """功能：真实 TinyDiT 的采样输出形状与数值有效。参数：无。返回：None。"""
    schedule = _schedule()
    torch.manual_seed(0)
    model = TinyDiT(image_size=4, patch_size=2, hidden_size=8, depth=1, num_heads=2)
    image, _ = p_sample_loop(model, (2, 1, 4, 4), schedule)
    assert image.shape == (2, 1, 4, 4)
    assert torch.isfinite(image).all()
