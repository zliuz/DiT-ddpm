"""M2 验收：TODO 未完成时应失败；手算小例子不是核心实现的参考答案。"""

import pytest
import torch

from ddpm.diffusion import (
    DiffusionSchedule,
    extract,
    make_linear_schedule,
    q_sample,
    sample_timesteps,
)


@pytest.fixture
def tiny_schedule() -> DiffusionSchedule:
    # 一个可手算的两步日程；独立于 TODO 1，便于单独调试 TODO 4。
    # 这些 beta 是刻意放大的测试值，不是 MNIST 默认的训练日程。
    return DiffusionSchedule(
        betas=torch.tensor([0.36, 0.4375]),
        alphas=torch.tensor([0.64, 0.5625]),
        alpha_bars=torch.tensor([0.64, 0.36]),
    )


def test_schedule_hand_calculated_values() -> None:
    schedule = make_linear_schedule(3, beta_start=0.1, beta_end=0.3)
    torch.testing.assert_close(schedule.betas, torch.tensor([0.1, 0.2, 0.3]))
    torch.testing.assert_close(schedule.alphas, torch.tensor([0.9, 0.8, 0.7]))
    torch.testing.assert_close(schedule.alpha_bars, torch.tensor([0.9, 0.72, 0.504]))


def test_schedule_default_signal_decreases_to_near_zero() -> None:
    schedule = make_linear_schedule()
    for values in (schedule.betas, schedule.alphas, schedule.alpha_bars):
        assert values.shape == (1000,)
        assert values.dtype == torch.float32
        assert torch.isfinite(values).all()
    assert torch.all(schedule.betas[1:] > schedule.betas[:-1])
    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])
    # 下标 0 已经加噪，最后一个默认时间步的原图信号应很弱。
    assert 0 < schedule.alpha_bars[0] < 1
    assert 0 < schedule.alpha_bars[-1] < 1e-3


@pytest.mark.parametrize(
    "kwargs", [{"num_steps": 1}, {"beta_start": 0.0}, {"beta_start": 0.5, "beta_end": 0.1}]
)
def test_schedule_invalid_arguments(kwargs: dict) -> None:
    """kwargs：pytest 每次传入的一组非法关键字参数，用 **kwargs 展开后调用。"""
    with pytest.raises(ValueError):
        make_linear_schedule(**kwargs)


def test_extract_mixed_and_repeated_timesteps_broadcast_per_image() -> None:
    values = torch.tensor([0.2, 0.4, 0.7])
    coefficients = extract(values, torch.tensor([2, 0, 2, 1]), (4, 1, 3, 5))
    assert coefficients.shape == (4, 1, 1, 1)
    torch.testing.assert_close(coefficients.flatten(), torch.tensor([0.7, 0.2, 0.7, 0.4]))
    # 特意让 batch size 与图片宽度不同，捕获错误的广播方向。
    images = torch.ones(4, 1, 3, 5) * coefficients
    for index, expected in enumerate([0.7, 0.2, 0.7, 0.4]):
        torch.testing.assert_close(images[index], torch.full((1, 3, 5), expected))


@pytest.mark.parametrize("t", [torch.tensor([-1]), torch.tensor([3])])
def test_extract_rejects_out_of_bounds_time(t: torch.Tensor) -> None:
    """t：[1] int64；测试 -1 与 T=3 两个越界下标，合法下标只有 0、1、2。"""
    with pytest.raises(ValueError):
        extract(torch.tensor([0.2, 0.4, 0.7]), t, (1, 1, 28, 28))


def test_timesteps_shape_range_and_seed() -> None:
    torch.manual_seed(42)
    first = sample_timesteps(4096, 4)
    torch.manual_seed(42)
    second = sample_timesteps(4096, 4)
    assert first.shape == (4096,)
    assert first.dtype == torch.int64
    assert set(first.tolist()) == {0, 1, 2, 3}
    assert torch.equal(first, second)
    # 函数内部不应重新设置种子，否则连续两次采样会完全相同。
    third = sample_timesteps(4096, 4)
    assert not torch.equal(second, third)


@pytest.mark.parametrize("batch_size,num_steps", [(0, 10), (2, 0)])
def test_timesteps_invalid_sizes(batch_size: int, num_steps: int) -> None:
    """batch_size/num_steps：分别用 B=0 或 T=0 检查是否拒绝无效的采样规模。"""
    with pytest.raises(ValueError):
        sample_timesteps(batch_size, num_steps)


def test_q_sample_zero_noise_isolates_signal(tiny_schedule: DiffusionSchedule) -> None:
    """tiny_schedule：pytest 自动注入的两步日程，alpha_bar 分别为 0.64、0.36。"""
    # 小图形状 [B=2,C=1,H=2,W=3]；两张图的 t 分别为 0、1，检查不同系数的广播。
    x_0 = torch.ones(2, 1, 2, 3)
    noise = torch.zeros_like(x_0)
    actual, used_noise = q_sample(x_0, torch.tensor([0, 1]), tiny_schedule, noise)
    torch.testing.assert_close(actual[0], torch.full((1, 2, 3), 0.8))
    torch.testing.assert_close(actual[1], torch.full((1, 2, 3), 0.6))
    assert torch.equal(used_noise, noise)


def test_q_sample_zero_image_isolates_noise(tiny_schedule: DiffusionSchedule) -> None:
    """tiny_schedule：两步测试日程；将原图设为零，只观察噪声系数是否正确。"""
    x_0 = torch.zeros(2, 1, 2, 3)
    noise = torch.ones_like(x_0)
    actual, _ = q_sample(x_0, torch.tensor([0, 1]), tiny_schedule, noise)
    torch.testing.assert_close(actual[0], torch.full((1, 2, 3), 0.6))
    torch.testing.assert_close(actual[1], torch.full((1, 2, 3), 0.8))


def test_q_sample_does_not_clip_or_modify_inputs(tiny_schedule: DiffusionSchedule) -> None:
    """tiny_schedule：两步测试日程；固定 noise 令结果越过 [-1,1]，检查未被裁剪。"""
    x_0 = torch.tensor([0.5, -0.5]).reshape(2, 1, 1, 1).expand(2, 1, 2, 3).clone()
    noise = torch.tensor([2.0, -2.0]).reshape(2, 1, 1, 1).expand_as(x_0).clone()
    original_x, original_noise = x_0.clone(), noise.clone()
    actual, used_noise = q_sample(x_0, torch.tensor([0, 1]), tiny_schedule, noise)
    torch.testing.assert_close(actual[0], torch.full((1, 2, 3), 1.6))
    torch.testing.assert_close(actual[1], torch.full((1, 2, 3), -1.9))
    assert torch.equal(x_0, original_x) and torch.equal(noise, original_noise)
    assert torch.equal(used_noise, original_noise)


def test_q_sample_generated_noise_is_normal_and_reproducible(
    tiny_schedule: DiffusionSchedule,
) -> None:
    """tiny_schedule：两步测试日程；不传 noise，检查函数自行生成的 epsilon。"""
    # 64×64 是测试用尺寸，用更多噪声样本检查均值/标准差，不是将 MNIST 改为 64×64。
    x_0 = torch.zeros(2, 1, 64, 64)
    t = torch.tensor([0, 1])
    torch.manual_seed(42)
    actual, noise = q_sample(x_0, t, tiny_schedule)
    torch.manual_seed(42)
    repeated, repeated_noise = q_sample(x_0, t, tiny_schedule)
    assert actual.shape == x_0.shape and noise.shape == x_0.shape
    assert actual.dtype == x_0.dtype and noise.device == x_0.device
    assert torch.equal(actual, repeated) and torch.equal(noise, repeated_noise)
    assert abs(noise.mean().item()) < 0.05
    assert abs(noise.std().item() - 1) < 0.05
    torch.testing.assert_close(actual[0], noise[0] * 0.6)
    torch.testing.assert_close(actual[1], noise[1] * 0.8)
