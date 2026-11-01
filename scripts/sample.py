"""M5：从训练好的 checkpoint 反向采样，保存固定种子的样本网格与去噪过程图。"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor

from ddpm.data import denormalize
from ddpm.diffusion import DiffusionSchedule, make_linear_schedule
from ddpm.model import TinyDiT
from ddpm.sampling import p_sample_loop
from ddpm.training import TrainConfig


def load_model_for_sampling(
    checkpoint_path: Path,
    device: str,
) -> tuple[TinyDiT, TrainConfig, int]:
    """功能：从本项目 checkpoint 恢复网络权重与模型配置；不需要优化器状态。

    参数：
        checkpoint_path：M4/M5 保存的 checkpoint.pt，格式版本必须为 1。
        device：模型加载后移动到的计算设备；不做隐式回退。
    返回：
        (model, config, saved_step)：model 已 eval() 并位于 device；saved_step 是
        checkpoint 中记录的已完成优化器更新数。格式或配置不合法时抛出 ValueError。
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format_version')}")
    saved_config = payload.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("checkpoint config is missing or malformed")
    try:
        config = TrainConfig(**saved_config)
    except TypeError as error:
        raise ValueError("checkpoint config is missing or malformed") from error
    saved_step = payload.get("step")
    if not isinstance(saved_step, int) or isinstance(saved_step, bool) or saved_step < 0:
        raise ValueError("checkpoint step must be a nonnegative int")

    model = TinyDiT(
        image_size=config.image_size,
        in_channels=config.in_channels,
        patch_size=config.patch_size,
        hidden_size=config.hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, config, saved_step


def select_snapshot_steps(total_steps: int, count: int) -> list[int]:
    """功能：在 [0,T-1] 上均匀选取若干反向步骤，降序返回。

    参数：
        total_steps：扩散日程长度 T，至少为 2。
        count：需要的快照数量；0 表示不保存过程图，1 表示只取最接近纯噪声的 t。
    返回：
        去重后的整数列表，从大到小；包含 T-1 和（count>1 时）0。
    """
    if count <= 0:
        return []
    if count == 1:
        return [total_steps - 1]
    fractions = {round(index * (total_steps - 1) / (count - 1)) for index in range(count)}
    return sorted(fractions, reverse=True)


def save_image_grid(path: Path, images: Tensor, title: str) -> None:
    """功能：把 [-1,1] 图片排成网格保存；只在显示时裁剪到 [0,1]。

    参数：
        path：PNG 输出路径。
        images：CPU 或 MPS 上的 [N,1,S,S] 张量；原始数值不被修改。
        title：图标题，用于记录 seed、步数等实验信息。
    返回：
        None；写入图片文件。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = images.shape[0]
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
    for index in range(rows * columns):
        axis = axes[index // columns, index % columns]
        axis.axis("off")
        if index < count:
            pixels = denormalize(images[index, 0].detach().cpu()).clamp(0, 1)
            axis.imshow(pixels.numpy(), cmap="gray", vmin=0, vmax=1)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def save_trajectory(
    path: Path,
    snapshots: dict[int, Tensor],
    steps: list[int],
    final_images: Tensor,
    title: str,
    max_rows: int = 4,
) -> None:
    """功能：每一行展示一张图片的反向去噪过程，最后一列为最终结果。

    参数：
        path：PNG 输出路径。
        snapshots：{t: 执行第 t 步之前的 x_t CPU 副本}。
        steps：要展示的 t，通常为降序；缺键时跳过该列。
        final_images：采样循环返回的最终图片，位于任意设备。
        title：图标题。
        max_rows：最多展示的图片数，避免过程图过大。
    返回：
        None；写入图片文件。显示时裁剪到 [0,1]，不改变采样结果。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [step for step in steps if step in snapshots]
    rows = min(max_rows, final_images.shape[0])
    figure, axes = plt.subplots(
        rows, len(available) + 1, figsize=(2 * (len(available) + 1), 2 * rows), squeeze=False
    )
    for row in range(rows):
        for column, step in enumerate(available):
            pixels = denormalize(snapshots[step][row, 0]).clamp(0, 1)
            axes[row, column].imshow(pixels.numpy(), cmap="gray", vmin=0, vmax=1)
            axes[row, column].set_title(f"t={step} (before step)")
            axes[row, column].axis("off")
        final_pixels = denormalize(final_images[row, 0].detach().cpu()).clamp(0, 1)
        axes[row, -1].imshow(final_pixels.numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, -1].set_title("final")
        axes[row, -1].axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    """功能：解析采样配置，恢复 checkpoint，运行 M5 反向采样并保存产物。

    参数：
        无函数入参；命令行指定 checkpoint、设备、种子、样本数和快照数。
    返回：
        None；生成 samples.png、trajectory.png 与 report.json。
        M5 核心 TODO 未完成时以状态码 2 退出，不创建空产物。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="要采样的 checkpoint.pt")
    parser.add_argument(
        "--device", choices=("cpu", "mps"), default="cpu", help="采样设备，默认 cpu"
    )
    parser.add_argument("--seed", type=int, default=0, help="初始噪声与每步噪声的 CPU 种子")
    parser.add_argument("--num-images", type=int, default=16, help="生成图片数 N，默认 16")
    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=6,
        help="去噪过程快照数，0 表示不画过程图，默认 6",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m5/sample"),
        help="图片与报告目录，默认 outputs/m5/sample",
    )
    args = parser.parse_args()
    if args.seed < 0 or args.num_images <= 0 or args.num_snapshots < 0:
        parser.error("seed must be nonnegative; num-images positive; num-snapshots >= 0")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; use --device cpu or check the environment")

    try:
        model, config, checkpoint_step = load_model_for_sampling(args.checkpoint, args.device)
        schedule: DiffusionSchedule = make_linear_schedule(
            config.num_steps, config.beta_start, config.beta_end
        )
        snapshot_steps = select_snapshot_steps(config.num_steps, args.num_snapshots)
        shape = (args.num_images, config.in_channels, config.image_size, config.image_size)
        # 固定 CPU 种子即可复现整条轨迹：初始噪声与每步噪声都由 CPU randn 生成。
        torch.manual_seed(args.seed)
        if args.device == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        images, snapshots = p_sample_loop(
            model, shape, schedule, device=args.device, snapshot_steps=snapshot_steps
        )
        if args.device == "mps":
            torch.mps.synchronize()
    except NotImplementedError as error:
        parser.exit(2, f"练习尚未完成：{error}。请填写 ddpm/sampling.py。\n")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    if not torch.isfinite(images).all().item():
        raise RuntimeError("Sampling produced non-finite images; check the checkpoint and schedule")

    elapsed_seconds = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_image_grid(
        args.output_dir / "samples.png",
        images,
        f"M5 samples: {config.mode}, checkpoint step {checkpoint_step}, seed {args.seed}",
    )
    if snapshot_steps:
        save_trajectory(
            args.output_dir / "trajectory.png",
            snapshots,
            snapshot_steps,
            images,
            f"M5 denoising: seed {args.seed}, T={config.num_steps}",
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "config": asdict(config),
        "device": args.device,
        "seed": args.seed,
        "shape": list(shape),
        "num_steps": config.num_steps,
        "snapshot_steps": snapshot_steps,
        "elapsed_seconds": elapsed_seconds,
        "final_min": images.min().item(),
        "final_max": images.max().item(),
        "final_mean": images.mean().item(),
        "display_note": (
            "PNG images are clipped to [0,1] only for display; raw tensors are unclipped."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"Device: {args.device}; checkpoint step: {checkpoint_step}; "
        f"seed: {args.seed}; time: {elapsed_seconds:.2f}s"
    )
    print(f"Final image range: [{report['final_min']:.4f}, {report['final_max']:.4f}]")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
