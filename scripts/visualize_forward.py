"""M2：完成 diffusion.py 的 TODO 后，绘制噪声日程和同一图片的加噪对比。"""

import argparse
import json
from pathlib import Path

import torch

from ddpm.data import denormalize, make_mnist_loader
from ddpm.diffusion import make_linear_schedule, q_sample


def main() -> None:
    """命令行参数决定读取路径、展示规模和随机种子，不执行模型训练。"""
    parser = argparse.ArgumentParser(description=__doc__)
    # type=Path 将输入转换为路径；相对路径以运行命令时所在目录（项目根目录）为基准。
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="MNIST 缓存根目录，默认 data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/m2"),
        help="图片及 JSON 报告保存目录，默认 outputs/m2",
    )
    # num_steps 是日程长度 T，不是训练次数或本次展示的图片列数。
    parser.add_argument(
        "--num-steps", type=int, default=1000, help="总扩散步数 T，至少为 2，默认 1000"
    )
    parser.add_argument(
        "--num-images", type=int, default=4, help="选取的原图数量（网格行数），默认 4"
    )
    # 固定 seed 方便复现噪声；更换 seed 不会改变不打乱时选取的原图。
    parser.add_argument("--seed", type=int, default=0, help="噪声随机种子，默认 0，用于复现实验")
    # store_true：不传选项时为 False，显式传 --download 才为 True；不需要再写 true。
    parser.add_argument(
        "--download", action="store_true", help="允许下载缺失的 MNIST 文件，默认只读缓存"
    )
    args = parser.parse_args()
    if args.num_steps < 2 or args.num_images < 1:
        parser.error("num-steps must be >= 2 and num-images must be >= 1")

    torch.manual_seed(args.seed)
    # 在读取数据和创建输出前检查 TODO 1，避免未完成时下载或留下空产物。
    try:
        schedule = make_linear_schedule(args.num_steps)
    except NotImplementedError as error:
        parser.exit(2, f"练习尚未完成：{error}。请先填写 ddpm/diffusion.py。\n")

    # batch_size 为展示数量；shuffle=False 保持原图顺序；download 控制是否允许联网下载。
    loader = make_mnist_loader(
        args.data_dir, batch_size=args.num_images, shuffle=False, download=args.download
    )
    x_0, labels = next(iter(loader))  # x_0=[B,1,28,28]；labels=[B]，标签仅用来显示。
    # 选择几个代表性下标；set 去重，避免 T 较小时多个比例落在同一下标。
    # T=1000 时依次为 0、50、200、500、999；干净图片会另占第一列。
    times = sorted(
        {0, args.num_steps // 20, args.num_steps // 5, args.num_steps // 2, args.num_steps - 1}
    )
    # 各列复用同一份 epsilon，便于单独观察系数变化；不是逐步模拟的 Markov 轨迹。
    fixed_noise = torch.randn_like(x_0)  # 与图片同形状的标准正态 epsilon。
    snapshots = []  # 各时间步的原始 x_t，每个元素都是 [B,1,28,28]。
    try:
        for step in times:
            # step 是当前列的单个整数；t=[B] 让这一列所有图片使用相同时间步。
            # 这只是对比图的选择，训练时可为每张图片独立采样 t。
            t = torch.full((len(x_0),), step, dtype=torch.int64)
            x_t, _ = q_sample(x_0, t, schedule, noise=fixed_noise)
            snapshots.append(x_t)
    except NotImplementedError as error:
        parser.exit(2, f"练习尚未完成：{error}。请先填写 ddpm/diffusion.py。\n")

    # TODO 尚未完成时无需初始化绘图库或创建绘图缓存。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # 行数=B，列数=时间步数+一列原图；figsize 单位为英寸。
    # squeeze=False 确保即使只展示一张图，axes 仍是可用 [row,column] 索引的二维数组。
    figure, axes = plt.subplots(
        len(x_0), len(times) + 1, figsize=(2 * (len(times) + 1), 2 * len(x_0)), squeeze=False
    )
    for row in range(len(x_0)):
        for column, batch in enumerate([x_0, *snapshots]):
            # 仅用于显示时裁剪；x_t 本身和报告中的数值都不裁剪。
            pixels = denormalize(batch[row, 0]).clamp(0, 1)
            # cmap=gray 使用灰度色图；vmin/vmax 固定显示范围，避免每张图自动缩放。
            axes[row, column].imshow(pixels.numpy(), cmap="gray", vmin=0, vmax=1)
            title = "clean" if column == 0 else f"t={times[column - 1]}"
            axes[row, column].set_title(f"{title}, label={labels[row].item()}")
            axes[row, column].axis("off")
    figure.suptitle("Forward marginals with shared noise (display clipped only)")
    figure.tight_layout()
    figure.savefig(args.output_dir / "forward_noise.png", dpi=120)  # dpi：每英寸输出像素数。
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(schedule.betas.numpy())
    axes[0].set(title="beta", xlabel="code timestep t")
    axes[1].plot(schedule.alpha_bars.numpy())
    axes[1].set(title="alpha_bar (signal variance factor)", xlabel="code timestep t")
    figure.tight_layout()
    figure.savefig(args.output_dir / "schedule.png", dpi=120)
    plt.close(figure)

    # 报告中的 alpha_bar 是累计信号方差系数；raw_min/raw_max 为整个 batch 的原始范围。
    # shape 是实际 x_0 形状；shared_noise_across_columns 记录是否跨时间步复用 epsilon。
    report = {
        "seed": args.seed,
        "num_steps": args.num_steps,
        "shape": list(x_0.shape),
        "shared_noise_across_columns": True,
        "steps": [
            {
                "t": step,
                "alpha_bar": schedule.alpha_bars[step].item(),
                "raw_min": batch.min().item(),
                "raw_max": batch.max().item(),
            }
            for step, batch in zip(times, snapshots, strict=True)
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved forward_noise.png, schedule.png, report.json to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
