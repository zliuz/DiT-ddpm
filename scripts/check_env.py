"""Check dependencies and one training step without implementing the learning exercises."""

import argparse
import platform
from importlib.metadata import version

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from torch import nn  # noqa: E402
from torchvision.transforms import v2  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()

    print(f"Python: {platform.python_version()} ({platform.machine()})")
    for package in ("torch", "torchvision", "matplotlib", "tqdm"):
        print(f"{package}: {version(package)}")

    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    print(f"MPS built: {mps_built}; available: {mps_available}")
    if args.device == "mps" and not mps_available:
        raise SystemExit("MPS was requested but is unavailable in this process.")
    selected = "mps" if args.device == "auto" and mps_available else args.device
    device = torch.device("cpu" if selected == "auto" else selected)

    torch.manual_seed(0)
    # Exercise attention, normalization, and backward on the actual selected device.
    model = nn.TransformerEncoderLayer(
        d_model=32, nhead=4, dim_feedforward=64, dropout=0.0, batch_first=True
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 49, 32, device=device)
    target = torch.randn_like(inputs)
    before = next(model.parameters()).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)
    loss = nn.functional.mse_loss(prediction, target)
    loss.backward()
    if not torch.isfinite(loss).item():
        raise SystemExit("Training check failed: loss is not finite.")
    for parameter in model.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise SystemExit("Training check failed: missing or non-finite gradients.")
    optimizer.step()
    if torch.equal(before, next(model.parameters()).detach()):
        raise SystemExit("Training check failed: parameters did not change.")
    if device.type == "mps":
        torch.mps.synchronize()

    # Validate torchvision transforms and plotting without downloading a dataset.
    transform = v2.Normalize(mean=(0.5,), std=(0.5,))
    normalized = transform(torch.zeros(1, 28, 28))
    if normalized.shape != (1, 28, 28) or not torch.all(normalized == -1).item():
        raise SystemExit("torchvision transform check failed.")
    figure, axes = plt.subplots()
    axes.imshow(normalized[0].numpy(), cmap="gray")
    figure.canvas.draw()
    plt.close(figure)

    print(f"Device: {device}")
    print(f"Attention training step: OK (loss={loss.item():.6f})")
    print(f"torchvision {torchvision.__version__} transforms and matplotlib: OK")
    print("Environment ready. Next milestone: M1 (MNIST data and visualization).")


if __name__ == "__main__":
    main()
