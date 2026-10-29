"""Inspect MNIST batches and save a labeled image grid and a JSON report."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from ddpm.data import denormalize, make_mnist_loader  # noqa: E402


def save_grid(images: torch.Tensor, labels: torch.Tensor, path: Path) -> None:
    """Display up to 64 clean images using a fixed grayscale range."""
    count = min(len(images), 64)
    columns = min(count, 8)
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns, rows), squeeze=False)
    pixels = denormalize(images[:count]).cpu()
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < count:
            axis.imshow(pixels[index, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"label: {labels[index].item()}", fontsize=9)
    figure.tight_layout(pad=0.5)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/m1"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    loader = make_mnist_loader(
        args.data_dir,
        batch_size=args.batch_size,
        train=args.split == "train",
        shuffle=False,  # Canonical dataset order makes images and labels easy to inspect.
        download=not args.no_download,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / f"mnist_{args.split}.png"
    report_path = args.output_dir / f"mnist_{args.split}.json"

    sample_count = 0
    pixel_min, pixel_max = math.inf, -math.inf
    label_counts = torch.zeros(10, dtype=torch.int64)
    first_batch = None
    last_batch_size = 0
    # Scan every batch to check range, dtype, labels, and the final partial batch.
    for images, labels in loader:
        if images.shape != (len(labels), 1, 28, 28) or images.dtype != torch.float32:
            raise RuntimeError("Expected float32 images with shape [B, 1, 28, 28].")
        if labels.ndim != 1 or labels.dtype != torch.int64:
            raise RuntimeError("Expected int64 labels with shape [B].")
        if not torch.isfinite(images).all().item():
            raise RuntimeError("Images contain NaN or infinity.")
        low, high = images.min().item(), images.max().item()
        if low < -1 or high > 1 or labels.min().item() < 0 or labels.max().item() > 9:
            raise RuntimeError("Image range or digit labels are invalid.")
        pixel_min, pixel_max = min(pixel_min, low), max(pixel_max, high)
        label_counts += torch.bincount(labels, minlength=10)
        sample_count += len(images)
        last_batch_size = len(images)
        if first_batch is None:
            first_batch = {
                "image_shape": list(images.shape),
                "label_shape": list(labels.shape),
                "image_dtype": str(images.dtype),
                "label_dtype": str(labels.dtype),
                "labels": labels.tolist(),
            }
            save_grid(images, labels, image_path)

    if first_batch is None or sample_count != len(loader.dataset):
        raise RuntimeError("DataLoader did not visit every sample exactly once.")
    report = {
        "split": args.split,
        "device": "cpu",
        "shuffle": False,
        "num_workers": loader.num_workers,
        "batch_size": args.batch_size,
        "num_samples": sample_count,
        "num_batches": len(loader),
        "last_batch_size": last_batch_size,
        "pixel_range": [pixel_min, pixel_max],
        "label_counts": label_counts.tolist(),
        "first_batch": first_batch,
        "image_path": str(image_path),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Split: {args.split}; samples: {sample_count}; batches: {len(loader)}")
    print(f"Images: {first_batch['image_shape']} {first_batch['image_dtype']}")
    print(f"Labels: {first_batch['label_shape']} {first_batch['label_dtype']}")
    print(f"Pixel range: [{pixel_min}, {pixel_max}]; last batch: {last_batch_size}")
    print(f"Image grid: {image_path.resolve()}")
    print(f"Report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
