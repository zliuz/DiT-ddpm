"""M5：完整 MNIST 训练入口；按固定间隔用反向采样链生成固定种子的样本。"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from ddpm.data import make_mnist_loader
from ddpm.diffusion import DiffusionSchedule, make_linear_schedule
from ddpm.model import TinyDiT
from ddpm.sampling import p_sample_loop
from ddpm.training import (
    TrainConfig,
    TrainingState,
    load_checkpoint,
    make_training_batch,
    save_checkpoint,
    train_step,
)
from scripts.sample import save_image_grid, save_trajectory, select_snapshot_steps


def _fixed_seed_sample(
    model: torch.nn.Module,
    config: TrainConfig,
    schedule: DiffusionSchedule,
    *,
    device: str,
    seed: int,
    count: int,
    snapshot_steps: list[int] | None = None,
) -> tuple[Tensor, dict[int, Tensor]]:
    """功能：用独立种子采样，结束后恢复训练随机状态。

    参数：
        model：当前的噪声预测器，会被采样函数临时切到 eval()。
        config：提供图片形状与通道数。
        schedule：完整扩散日程，长度应与训练一致。
        device：采样计算设备。
        seed：本次采样的 CPU 种子；相同模型与种子得到相同图片。
        count：生成图片数。
        snapshot_steps：可选的去噪过程快照下标。
    返回：
        (images, snapshots)。训练用的 CPU RNG 状态在返回前已恢复，采样不会
        改变后续训练抽样的随机序列。
    """
    rng_state = torch.get_rng_state()
    try:
        torch.manual_seed(seed)
        shape = (count, config.in_channels, config.image_size, config.image_size)
        return p_sample_loop(
            model, shape, schedule, device=device, snapshot_steps=snapshot_steps or []
        )
    finally:
        # M5 的初始噪声与每步噪声都在 CPU 生成，因此恢复 CPU 状态即可。
        torch.set_rng_state(rng_state)


def check_sampling_ready(
    model: torch.nn.Module,
    config: TrainConfig,
    device: str,
) -> None:
    """功能：用两步日程预检采样链，避免训练很久后才发现 TODO 未完成。

    参数：
        model：已移动到 device 的网络；预检后会恢复为 train()。
        config：提供图片形状与 beta 范围。
        device：预检计算设备。
    返回：
        None；核心 TODO 未实现时抛出 NotImplementedError，其他数值问题照常抛出。
    """
    rng_state = torch.get_rng_state()
    try:
        probe_schedule = make_linear_schedule(2, config.beta_start, config.beta_end)
        shape = (1, config.in_channels, config.image_size, config.image_size)
        p_sample_loop(model, shape, probe_schedule, device=device)
    finally:
        torch.set_rng_state(rng_state)
        model.train()


def write_report(
    output_dir: Path,
    config: TrainConfig,
    state: TrainingState,
    *,
    device: str,
    start_step: int,
    train_seconds: float,
    sample_seconds: float,
    steps_per_epoch: int,
    samples: list[str],
) -> None:
    """功能：保存累计 loss 曲线与本次训练配置、进度和产物列表。

    参数：
        output_dir：输出目录，已存在或由此函数创建。
        config：本次实验配置。
        state：累计进度；losses 长度必须等于 step。
        device：实际计算设备。
        start_step：本次训练前已完成的更新数；恢复时大于零。
        train_seconds：只统计 train_step 的累计耗时。
        sample_seconds：只统计周期性采样与保存 checkpoint 的耗时。
        steps_per_epoch：按当前 batch 大小计算的一个 epoch 更新数。
        samples：周期性样本图片的相对文件名列表。
    返回：
        None；写入 loss.png 与 report.json。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    recent = state.losses[-min(10, len(state.losses)) :]
    report = {
        "config": asdict(config),
        "device": device,
        "start_step": start_step,
        "step": state.step,
        "additional_steps": state.step - start_step,
        "steps_per_epoch": steps_per_epoch,
        "train_seconds": train_seconds,
        "sample_seconds": sample_seconds,
        "loss_first_recorded": state.losses[0],
        "loss_last_recorded": state.losses[-1],
        "loss_last_10_mean": sum(recent) / len(recent),
        "losses": state.losses,
        "samples": samples,
        "checkpoint": "checkpoint.pt",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(range(1, state.step + 1), state.losses, linewidth=0.8)
    axis.set(xlabel="Optimizer update", ylabel="Noise MSE", title="M5: full MNIST training")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "loss.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """功能：遍历完整 MNIST，按 M4 的单步逻辑训练，并定期采样固定种子图片。

    参数：
        无函数入参；命令行控制设备、更新次数、batch、学习率、模型规模与采样间隔。
    返回：
        None；生成 checkpoint.pt、周期样本、最终样本与去噪过程图、loss.png、
        report.json。--steps 表示本次额外更新数，不是 epoch 数或累计步数。
        M5 采样 TODO 未完成时在训练前以状态码 2 退出。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="M1 已下载的 MNIST 目录，默认 data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m5/full"),
        help="训练产物目录，默认 outputs/m5/full",
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu", help="计算设备")
    parser.add_argument("--steps", type=int, default=3000, help="本次额外更新数，默认 3000")
    parser.add_argument("--batch-size", type=int, default=64, help="每步图片数 B，默认 64")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW 学习率")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW 权重衰减")
    parser.add_argument("--seed", type=int, default=0, help="训练初始化与数据顺序种子")
    parser.add_argument("--num-steps", type=int, default=1000, help="扩散日程长度 T")
    parser.add_argument("--patch-size", type=int, default=4, help="patch 边长 P，默认 4")
    parser.add_argument("--hidden-size", type=int, default=128, help="隐藏维度 D，默认 128")
    parser.add_argument("--depth", type=int, default=4, help="Transformer block 数，默认 4")
    parser.add_argument("--num-heads", type=int, default=4, help="注意力头数，默认 4")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=1000,
        help="每多少步保存一次样本和 checkpoint；0 表示只保存最终结果",
    )
    parser.add_argument("--num-samples", type=int, default=16, help="每次采样的图片数，默认 16")
    parser.add_argument("--sample-seed", type=int, default=0, help="固定采样种子，默认 0")
    parser.add_argument("--resume", type=Path, default=None, help="从本项目 checkpoint 继续更新")
    args = parser.parse_args()
    if (
        args.steps <= 0
        or args.batch_size <= 0
        or args.num_samples <= 0
        or args.sample_every < 0
        or args.sample_seed < 0
    ):
        parser.error("steps, batch-size, num-samples must be positive; sample-every/seed >= 0")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; use --device cpu or check the environment")

    try:
        loader = make_mnist_loader(
            args.data_dir,
            batch_size=args.batch_size,
            train=True,
            shuffle=True,
            seed=args.seed,
            download=False,
        )
        config = TrainConfig(
            mode="random",
            batch_size=args.batch_size,
            num_samples=len(loader.dataset),
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_steps=args.num_steps,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
        )
        torch.manual_seed(config.seed)
        model = TinyDiT(
            image_size=config.image_size,
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
        ).to(args.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        schedule = make_linear_schedule(config.num_steps, config.beta_start, config.beta_end)
        # 先预检采样链：M5 核心 TODO 未完成时不要开始长时间训练。
        check_sampling_ready(model, config, args.device)
        state = TrainingState()
        if args.resume is not None:
            state = load_checkpoint(args.resume, model, optimizer, config)
        start_step = state.step
        steps_per_epoch = len(loader)
        data_iterator = iter(loader)
        train_seconds = 0.0
        sample_seconds = 0.0
        samples: list[str] = []
        trajectory_steps = select_snapshot_steps(config.num_steps, 6)

        if args.device == "mps":
            torch.mps.synchronize()
        for _ in range(args.steps):
            try:
                images, _labels = next(data_iterator)  # M5 仍是无条件生成，标签不使用。
            except StopIteration:
                # 每个 epoch 重新开始；生成器状态继续推进，因此各轮打乱顺序不同。
                data_iterator = iter(loader)
                images, _labels = next(data_iterator)

            batch = make_training_batch(images, schedule)
            step_started = time.perf_counter()
            loss = train_step(model, optimizer, batch, device=args.device)
            train_seconds += time.perf_counter() - step_started
            state.step += 1
            state.losses.append(loss)

            if args.sample_every and state.step % args.sample_every == 0:
                sample_started = time.perf_counter()
                # 先保存 checkpoint，再采样；采样出错也不会丢失到这一步的进度。
                save_checkpoint(args.output_dir / "checkpoint.pt", model, optimizer, config, state)
                sample_images, _ = _fixed_seed_sample(
                    model,
                    config,
                    schedule,
                    device=args.device,
                    seed=args.sample_seed,
                    count=args.num_samples,
                )
                path = args.output_dir / f"sample_step_{state.step:06d}.png"
                save_image_grid(
                    path,
                    sample_images,
                    f"step {state.step}, seed {args.sample_seed}, T={config.num_steps}",
                )
                samples.append(path.name)
                sample_seconds += time.perf_counter() - sample_started
                print(
                    f"step {state.step}: loss={loss:.6f}, sample saved "
                    f"(train {train_seconds:.1f}s, sample {sample_seconds:.1f}s)"
                )
        if args.device == "mps":
            torch.mps.synchronize()

        save_checkpoint(args.output_dir / "checkpoint.pt", model, optimizer, config, state)
        sample_started = time.perf_counter()
        final_images, snapshots = _fixed_seed_sample(
            model,
            config,
            schedule,
            device=args.device,
            seed=args.sample_seed,
            count=args.num_samples,
            snapshot_steps=trajectory_steps,
        )
        save_image_grid(
            args.output_dir / "samples_final.png",
            final_images,
            f"final sample: step {state.step}, seed {args.sample_seed}, T={config.num_steps}",
        )
        save_trajectory(
            args.output_dir / "trajectory_final.png",
            snapshots,
            trajectory_steps,
            final_images,
            f"final denoising: step {state.step}, seed {args.sample_seed}",
        )
        sample_seconds += time.perf_counter() - sample_started
    except NotImplementedError as error:
        parser.exit(2, f"练习尚未完成：{error}。请填写 ddpm/sampling.py 或 ddpm/training.py。\n")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    write_report(
        args.output_dir,
        config,
        state,
        device=args.device,
        start_step=start_step,
        train_seconds=train_seconds,
        sample_seconds=sample_seconds,
        steps_per_epoch=steps_per_epoch,
        samples=samples,
    )
    print(
        f"Device: {args.device}; updates: {start_step} -> {state.step}; "
        f"loss: {state.losses[0]:.6f} -> {state.losses[-1]:.6f}"
    )
    print(f"Training time: {train_seconds:.1f}s; sampling time: {sample_seconds:.1f}s")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
