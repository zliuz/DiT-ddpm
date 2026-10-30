"""M3：填写 model.py 后，检查参数量、前向、梯度与一次参数更新；不执行完整训练。"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from ddpm.diffusion import make_linear_schedule, q_sample, sample_timesteps
from ddpm.model import TinyDiT


def main() -> None:
    """功能：解析检查配置，在指定设备执行一次小型网络运算并写入报告。

    参数：
        无函数入参；命令行选项控制 device、batch_size、模型规模和输出目录。
        默认用合成图片构造输入，不下载 MNIST，不进行多步训练。
    返回：
        None。成功时保存 outputs/m3/model_check.json；TODO 未完成时以状态码 2 退出。
        检查通过仅说明网络可运行，不表示已经学会去噪或可以生成数字。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cpu", "mps"),
        default="cpu",
        help="检查设备，默认 cpu；显式选 mps 时要求 GPU 可用，不隐式回退",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="合成图片数 B，默认 2")
    parser.add_argument("--patch-size", type=int, default=4, help="patch 边长 P，默认 4")
    parser.add_argument("--hidden-size", type=int, default=128, help="隐藏维度 D，默认 128")
    parser.add_argument("--depth", type=int, default=4, help="Transformer block 层数，默认 4")
    parser.add_argument("--num-heads", type=int, default=4, help="每层注意力头数，默认 4")
    parser.add_argument("--seed", type=int, default=0, help="合成数据和网络初始化随机种子")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/m3"), help="检查报告保存目录"
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this process; use CPU or check the environment")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    # 必须先成功构造模型；TODO 尚未填写时不创建输出目录或伪造检查产物。
    try:
        model = TinyDiT(
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
        ).to(device)
        model.train()
        # x_0=[B,1,28,28] 为 [-1,1] 的合成数据，只用于检查网络接线和数值有效性。
        x_0 = torch.rand(args.batch_size, 1, 28, 28, device=device) * 2 - 1
        schedule = make_linear_schedule(device=device)
        t = sample_timesteps(args.batch_size, len(schedule.betas), device=device)
        x_t, target_noise = q_sample(x_0, t, schedule)
        prediction = model(x_t, t)
    except NotImplementedError as error:
        parser.exit(2, f"练习尚未完成：{error}。请填写 ddpm/model.py。\n")
    except ValueError as error:
        parser.error(str(error))

    if prediction.shape != x_t.shape or not torch.isfinite(prediction).all().item():
        raise RuntimeError("Expected finite predictions with the same shape as x_t")
    # lr 只是一次更新检查的学习率，不是 M4 最终训练配置。
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(prediction, target_noise)
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (
            parameter.grad is None or not torch.isfinite(parameter.grad).all().item()
        ):
            raise RuntimeError(f"Missing or non-finite gradient: {name}")
    optimizer.step()
    changed = any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    if not changed:
        raise RuntimeError("No parameter changed after optimizer.step()")
    if device.type == "mps":
        torch.mps.synchronize()

    # num_parameters 包含所有注册参数；trainable_parameters 仅统计 requires_grad=True。
    report = {
        "device": str(device),
        "seed": args.seed,
        "input_source": "synthetic_uniform_clean_images",
        "input_shape": list(x_t.shape),
        "output_shape": list(prediction.shape),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "timesteps": t.tolist(),
        "loss_before_update": loss.item(),
        "finite_gradients": True,
        "parameters_changed": changed,
        "config": {
            "image_size": 28,
            "in_channels": 1,
            "patch_size": args.patch_size,
            "hidden_size": args.hidden_size,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "mlp_ratio": 4,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "model_check.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Device: {device}; parameters: {report['num_parameters']:,}")
    print(f"Forward, backward, and parameter update: OK; loss={loss.item():.6f}")
    print(f"Report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
