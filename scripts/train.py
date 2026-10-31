"""M4 小样本训练入口：先固定目标拟合，再随机采样；核心逻辑在 ddpm/training.py。"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ddpm.data import make_mnist_loader
from ddpm.diffusion import make_linear_schedule
from ddpm.model import TinyDiT
from ddpm.training import (
    TrainConfig,
    TrainingState,
    load_checkpoint,
    run_training,
    save_checkpoint,
)


def write_training_report(
    output_dir: Path,
    config: TrainConfig,
    state: TrainingState,
    *,
    device: str,
    start_step: int,
    elapsed_seconds: float,
) -> None:
    """功能：保存累计 loss 曲线与本次短运行的配置、进度和耗时。

    参数：
        output_dir：输出目录；checkpoint 由 main 另行保存。
        config：此次实验配置，与 checkpoint 一致。
        state：包含累计 loss 记录的已完成进度；至少有一条记录。
        device：本次实际使用的 cpu 或 mps。
        start_step：本次训练前已完成的步数，恢复时可以大于零。
        elapsed_seconds：仅本次追加训练的耗时；MPS 计时结束前需同步。
    返回：
        None；生成 loss.png 与 report.json，不包含生成图片或采样质量结论。
    """
    # 绘图在训练成功后才导入，TODO 未完成时不创建空曲线。
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
        "elapsed_seconds": elapsed_seconds,
        "loss_first_recorded": state.losses[0],
        "loss_last_recorded": state.losses[-1],
        "loss_last_10_mean": sum(recent) / len(recent),
        "losses": state.losses,
        "checkpoint": "checkpoint.pt",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, state.step + 1), state.losses, linewidth=1)
    ax.set(xlabel="Optimizer update", ylabel="Noise MSE", title=f"M4: {config.mode}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """功能：读取本地 MNIST 小子集，调用学习者实现的训练与 checkpoint 函数。

    参数：
        无函数入参；命令行参数配置实验模式、额外步数、设备及模型规模。
        --resume 时仍需传入与原实验相同的配置；--steps 表示额外更新次数。
    返回：
        None；完成后生成 checkpoint.pt、report.json、loss.png。
        TODO 未完成时退出码为 2；不自动下载数据，不启动完整 MNIST 训练。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("overfit", "random"),
        default="overfit",
        help="固定目标拟合或每步随机采样，默认 overfit",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        default="cpu",
        help="计算设备，默认 cpu；mps 必须实际可用",
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="本次额外更新次数，不是扩散日程 T，默认 100"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="每次更新图片数 B，默认 8")
    parser.add_argument(
        "--num-samples", type=int, default=64, help="读取训练集前多少张图片，默认 64，最多 60000"
    )
    parser.add_argument("--seed", type=int, default=0, help="新实验随机种子，默认 0")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="AdamW 学习率，默认 0.001"
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW 权重衰减，默认 0")
    parser.add_argument(
        "--num-steps", type=int, default=1000, help="扩散日程长度 T，不是训练更新数，默认 1000"
    )
    parser.add_argument("--patch-size", type=int, default=4, help="patch 边长 P，默认 4")
    parser.add_argument("--hidden-size", type=int, default=128, help="隐藏维度 D，默认 128")
    parser.add_argument("--depth", type=int, default=4, help="block 数量，默认 4")
    parser.add_argument("--num-heads", type=int, default=4, help="注意力头数，默认 4")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="M1 已下载的 MNIST 所在目录，默认 data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m4"),
        help="本次产物目录，默认 outputs/m4；不同实验建议分目录",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="恢复的 checkpoint 路径；配置必须与原实验相同"
    )
    args = parser.parse_args()
    if args.steps <= 0 or not 0 < args.num_samples <= 60000:
        parser.error("steps must be positive; num-samples must be in [1,60000]")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; check the environment or explicitly use --device cpu")

    try:
        config = TrainConfig(
            mode=args.mode,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            seed=args.seed,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_steps=args.num_steps,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
        )
        # 只在新进程初始化时设种子；load_checkpoint 会在构造之后恢复保存的 RNG。
        torch.manual_seed(config.seed)
        loader = make_mnist_loader(
            args.data_dir,
            batch_size=config.num_samples,
            train=True,
            shuffle=False,
            seed=config.seed,
            download=False,
        )
        images, _labels = next(iter(loader))  # M4 无类别条件，保留但不使用数字标签。
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
        # 本阶段始终在 CPU 生成训练任务，然后由 train_step 搬到计算设备。
        schedule = make_linear_schedule(config.num_steps, config.beta_start, config.beta_end)
        state = TrainingState()
        if args.resume is not None:
            state = load_checkpoint(args.resume, model, optimizer, config)
        start_step = state.step
        if args.device == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        state = run_training(
            model,
            optimizer,
            images,
            schedule,
            config,
            steps=args.steps,
            device=args.device,
            state=state,
        )
        if args.device == "mps":
            torch.mps.synchronize()
        elapsed_seconds = time.perf_counter() - started
        save_checkpoint(args.output_dir / "checkpoint.pt", model, optimizer, config, state)
    except NotImplementedError as error:
        parser.exit(2, f"练习尚未完成：{error}。请填写 ddpm/training.py。\n")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    write_training_report(
        args.output_dir,
        config,
        state,
        device=args.device,
        start_step=start_step,
        elapsed_seconds=elapsed_seconds,
    )
    print(f"Device: {args.device}; mode: {config.mode}; updates: {start_step} -> {state.step}")
    print(f"Loss: {state.losses[0]:.6f} -> {state.losses[-1]:.6f}; time: {elapsed_seconds:.2f}s")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
