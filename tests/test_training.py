"""M4 验收：直接测试学习者的实现；TODO 未填写时应失败，不跳过测试。"""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch import Tensor

from ddpm import training
from ddpm.diffusion import DiffusionSchedule, make_linear_schedule, q_sample
from ddpm.model import TinyDiT
from ddpm.training import (
    TrainConfig,
    TrainingBatch,
    TrainingState,
    load_checkpoint,
    make_training_batch,
    noise_prediction_loss,
    run_training,
    save_checkpoint,
    train_step,
)


def _config(**changes) -> TrainConfig:
    """功能：构造快速 CPU 验收配置。参数：changes 为覆盖字段。返回：TrainConfig。"""
    config = TrainConfig(
        image_size=8,
        patch_size=2,
        hidden_size=16,
        depth=1,
        num_heads=2,
        batch_size=2,
        num_samples=4,
        num_steps=20,
        learning_rate=5e-3,
    )
    return replace(config, **changes)


def _parts(config: TrainConfig) -> tuple[TinyDiT, torch.optim.AdamW, DiffusionSchedule]:
    """功能：按配置创建真实 TinyDiT 和训练依赖。参数：config 为实验配置。返回：三元组。"""
    model = TinyDiT(
        image_size=config.image_size,
        in_channels=config.in_channels,
        patch_size=config.patch_size,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    schedule = make_linear_schedule(config.num_steps, config.beta_start, config.beta_end)
    return model, optimizer, schedule


def _images() -> Tensor:
    """功能：创建固定的小型合成图片，避免下载数据。参数：无。返回：[4,1,8,8] float32。"""
    return torch.linspace(-1, 1, 4 * 8 * 8).reshape(4, 1, 8, 8)


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "sample"},
        {"batch_size": 0},
        {"num_samples": 1},
        {"learning_rate": float("nan")},
        {"patch_size": 3},
    ],
)
def test_config_rejects_invalid_values(changes: dict) -> None:
    """功能：拒绝非法训练配置。参数：changes 为非法覆盖项。返回：None。"""
    with pytest.raises(ValueError):
        _config(**changes)


def test_batch_fixed_targets_match_forward_diffusion() -> None:
    """功能：固定 t/噪声时复用 M2 公式且不修改原图。参数：无。返回：None。"""
    images = _images()[:2]
    before = images.clone()
    schedule = make_linear_schedule(20)
    t = torch.tensor([0, 19])
    noise = torch.full_like(images, 2.0)
    actual = make_training_batch(images, schedule, t=t, noise=noise)
    expected, _ = q_sample(images, t, schedule, noise=noise)
    torch.testing.assert_close(actual.x_t, expected)
    assert torch.equal(actual.t, t) and torch.equal(actual.target_noise, noise)
    assert torch.equal(images, before)


def test_batch_random_targets_are_fresh_and_seeded() -> None:
    """功能：检查随机目标范围、新鲜性与种子复现。参数：无。返回：None。"""
    schedule = make_linear_schedule(20)
    torch.manual_seed(10)
    first = make_training_batch(_images(), schedule)
    second = make_training_batch(_images(), schedule)
    torch.manual_seed(10)
    repeated = make_training_batch(_images(), schedule)
    assert first.t.dtype == torch.int64 and first.t.shape == (4,)
    assert ((first.t >= 0) & (first.t < 20)).all()
    assert first.x_t.device.type == "cpu" and first.target_noise.dtype == torch.float32
    assert not torch.equal(first.target_noise, second.target_noise)
    assert torch.equal(first.t, repeated.t)
    assert torch.equal(first.target_noise, repeated.target_noise)
    assert torch.equal(first.x_t, repeated.x_t)


def test_loss_uses_noise_target_and_preserves_gradient() -> None:
    """功能：已知预测为 1、目标为 2 时均值 MSE 应为 1。参数：无。返回：None。"""
    model, _, _ = _parts(_config())
    with torch.no_grad():
        model.output_projection.weight.zero_()
        model.output_projection.bias.fill_(1.0)
    x_t = torch.zeros(2, 1, 8, 8)
    loss = noise_prediction_loss(model, x_t, torch.tensor([0, 19]), torch.full_like(x_t, 2))
    assert loss.ndim == 0 and loss.requires_grad
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert model.output_projection.bias.grad is not None
    assert model.output_projection.bias.grad.abs().sum() > 0


def test_loss_rejects_broadcast_target() -> None:
    """功能：拒绝广播后的错误监督形状。参数：无。返回：None。"""
    model, _, _ = _parts(_config())
    with pytest.raises(ValueError):
        noise_prediction_loss(
            model, torch.zeros(2, 1, 8, 8), torch.tensor([0, 1]), torch.zeros(1, 1, 8, 8)
        )


def test_train_step_updates_parameters_and_sets_train_mode() -> None:
    """功能：检查一次真实更新、训练模式与返回值。参数：无。返回：None。"""
    torch.manual_seed(0)
    model, optimizer, schedule = _parts(_config())
    model.eval()
    batch = make_training_batch(_images()[:2], schedule)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = train_step(model, optimizer, batch)
    assert isinstance(loss, float) and torch.isfinite(torch.tensor(loss))
    assert model.training
    assert any(
        not torch.equal(old, new) for old, new in zip(before, model.parameters(), strict=True)
    )
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_train_step_clears_stale_gradients() -> None:
    """功能：已有巨大的残留梯度不应改变本步结果。参数：无。返回：None。"""
    torch.manual_seed(0)
    clean, optimizer, schedule = _parts(_config())
    dirty, dirty_optimizer, _ = _parts(_config())
    dirty.load_state_dict(clean.state_dict())
    batch = make_training_batch(_images()[:2], schedule)
    for parameter in dirty.parameters():
        parameter.grad = torch.full_like(parameter, 1e6)
    train_step(clean, optimizer, batch)
    train_step(dirty, dirty_optimizer, batch)
    for left, right in zip(clean.parameters(), dirty.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


@pytest.mark.parametrize("bad_gradient", [False, True])
def test_train_step_rejects_nonfinite_before_update(bad_gradient: bool) -> None:
    """功能：loss 或梯度非有限时禁止更新。参数：bad_gradient 选择梯度故障。返回：None。"""
    model, optimizer, schedule = _parts(_config())
    batch = make_training_batch(_images()[:2], schedule)
    if bad_gradient:
        # 只注入故障，不提供任何训练算法的参考实现。
        def corrupt_gradient(gradient: Tensor) -> Tensor:
            """功能：注入 NaN。参数：gradient 为原梯度。返回：同形状的 NaN 梯度。"""
            return torch.full_like(gradient, float("nan"))

        handle = model.output_projection.bias.register_hook(corrupt_gradient)
    else:
        batch = TrainingBatch(batch.x_t, batch.t, torch.full_like(batch.target_noise, float("nan")))
        handle = None
    before = [parameter.detach().clone() for parameter in model.parameters()]
    try:
        with pytest.raises(RuntimeError):
            train_step(model, optimizer, batch)
    finally:
        if handle is not None:
            handle.remove()
    for old, new in zip(before, model.parameters(), strict=True):
        assert torch.equal(old, new)
    assert not optimizer.state


@pytest.mark.parametrize("steps", [0, -1])
def test_loop_rejects_nonpositive_steps(steps: int) -> None:
    """功能：拒绝无效更新次数。参数：steps 为零或负数。返回：None。"""
    config = _config()
    model, optimizer, schedule = _parts(config)
    with pytest.raises(ValueError):
        run_training(model, optimizer, _images(), schedule, config, steps=steps)


@pytest.mark.parametrize("mode, expected_calls", [("overfit", 1), ("random", 3)])
def test_loop_sampling_policy(mode: str, expected_calls: int, monkeypatch) -> None:
    """功能：检查两种模式实际构造目标的次数。参数：mode、次数与 pytest spy。返回：None。"""
    torch.manual_seed(0)
    config = _config(mode=mode)
    model, optimizer, schedule = _parts(config)
    batches = []
    original = training.make_training_batch

    def record_batch(*args, **kwargs) -> TrainingBatch:
        """功能：记录实际 batch，不替代实现。参数：透传构造参数。返回：原函数结果。"""
        batch = original(*args, **kwargs)
        batches.append(batch)
        return batch

    monkeypatch.setattr(training, "make_training_batch", record_batch)
    state = run_training(model, optimizer, _images(), schedule, config, steps=3)
    assert state.step == len(state.losses) == 3
    assert all(
        isinstance(loss, float) and torch.isfinite(torch.tensor(loss)) for loss in state.losses
    )
    assert len(batches) == expected_calls
    if mode == "overfit":
        assert state.fixed_batch is not None
        assert torch.equal(state.fixed_batch.target_noise, batches[0].target_noise)
    else:
        assert state.fixed_batch is None
        assert not torch.equal(batches[0].target_noise, batches[1].target_noise)


def test_loop_overfit_reduces_fixed_target_loss() -> None:
    """功能：小型真实网络应能拟合固定噪声目标。参数：无。返回：None。"""
    torch.manual_seed(0)
    config = _config()
    model, optimizer, schedule = _parts(config)
    state = run_training(model, optimizer, _images(), schedule, config, steps=80)
    assert state.step == len(state.losses) == 80
    assert sum(state.losses[-10:]) / 10 < state.losses[0] * 0.6


def test_checkpoint_rejects_inconsistent_progress(tmp_path: Path) -> None:
    """功能：禁止保存步数与历史不一致的状态。参数：tmp_path 为隔离目录。返回：None。"""
    config = _config()
    model, optimizer, _ = _parts(config)
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "bad.pt", model, optimizer, config, TrainingState(step=1))


def test_checkpoint_roundtrip_restores_all_state(tmp_path: Path) -> None:
    """功能：保存与恢复模型、AdamW、固定目标和 RNG。参数：tmp_path 为隔离目录。返回：None。"""
    torch.manual_seed(0)
    config = _config()
    model, optimizer, schedule = _parts(config)
    state = run_training(model, optimizer, _images(), schedule, config, steps=2)
    path = tmp_path / "nested" / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, config, state)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["format_version"] == 1 and payload["config"] == asdict(config)
    assert payload["step"] == 2 and payload["losses"] == state.losses
    assert payload["rng_state"].device.type == "cpu"
    assert payload["fixed_batch"]["x_t"].device.type == "cpu"
    expected_random = torch.rand(5)
    restored_model, restored_optimizer, _ = _parts(config)
    restored = load_checkpoint(path, restored_model, restored_optimizer, config)
    assert torch.equal(torch.rand(5), expected_random)
    assert restored.step == state.step and restored.losses == state.losses
    for left, right in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(left, right)
        assert torch.equal(
            optimizer.state[left]["exp_avg"], restored_optimizer.state[right]["exp_avg"]
        )
        assert torch.equal(
            optimizer.state[left]["exp_avg_sq"], restored_optimizer.state[right]["exp_avg_sq"]
        )
        assert torch.equal(optimizer.state[left]["step"], restored_optimizer.state[right]["step"])
    assert restored.fixed_batch is not None and state.fixed_batch is not None
    for name in ("x_t", "t", "target_noise"):
        assert torch.equal(getattr(restored.fixed_batch, name), getattr(state.fixed_batch, name))


@pytest.mark.parametrize("changes", [{"num_steps": 30}, {"learning_rate": 1e-4}])
def test_checkpoint_mismatch_does_not_mutate_state(tmp_path: Path, changes: dict) -> None:
    """功能：不兼容配置应在修改运行状态前失败。参数：目录与不匹配字段。返回：None。"""
    config = _config()
    model, optimizer, schedule = _parts(config)
    state = run_training(model, optimizer, _images(), schedule, config, steps=1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, config, state)
    different = replace(config, **changes)
    fresh, fresh_optimizer, _ = _parts(different)
    before = [parameter.detach().clone() for parameter in fresh.parameters()]
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        load_checkpoint(path, fresh, fresh_optimizer, different)
    for old, new in zip(before, fresh.parameters(), strict=True):
        assert torch.equal(old, new)
    assert not fresh_optimizer.state
    assert torch.equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("format_version", 999),
        ("step", 100),
        ("fixed_batch", None),
    ],
)
def test_checkpoint_rejects_invalid_payload(tmp_path: Path, field: str, bad_value) -> None:
    """功能：拒绝版本、进度或固定目标缺失。参数：隔离目录及破坏字段。返回：None。"""
    config = _config()
    model, optimizer, schedule = _parts(config)
    state = run_training(model, optimizer, _images(), schedule, config, steps=1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, config, state)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload[field] = bad_value
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_checkpoint(path, model, optimizer, config)


@pytest.mark.parametrize("mode", ["overfit", "random"])
def test_checkpoint_resume_matches_uninterrupted_cpu(tmp_path: Path, mode: str) -> None:
    """功能：恢复后的两步应与连续运行一致。参数：隔离目录与训练模式。返回：None。"""
    torch.manual_seed(0)
    config = _config(mode=mode)
    model, optimizer, schedule = _parts(config)
    state = run_training(model, optimizer, _images(), schedule, config, steps=3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, config, state)
    continuous = run_training(model, optimizer, _images(), schedule, config, steps=2, state=state)
    # 模拟另一个进程先构造模型、优化器，随后再恢复随机状态。
    torch.manual_seed(999)
    restored_model, restored_optimizer, restored_schedule = _parts(config)
    loaded = load_checkpoint(path, restored_model, restored_optimizer, config)
    assert loaded.step == len(loaded.losses) == 3
    resumed = run_training(
        restored_model,
        restored_optimizer,
        _images(),
        restored_schedule,
        config,
        steps=2,
        state=loaded,
    )
    assert resumed.step == len(resumed.losses) == 5
    assert resumed.losses == continuous.losses
    for left, right in zip(model.parameters(), restored_model.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    if mode == "overfit":
        assert continuous.fixed_batch is not None and resumed.fixed_batch is not None
        assert torch.equal(continuous.fixed_batch.x_t, resumed.fixed_batch.x_t)
