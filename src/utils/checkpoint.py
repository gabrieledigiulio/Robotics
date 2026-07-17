import torch
from pathlib import Path


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    val_loss: float,
                    path) -> None:
    """
    Saves the complete training state to a .pt file.

    Args:
        model:     The PyTorch model to save.
        optimizer: The optimizer (includes momentum, lr, etc.).
        epoch:     The current epoch (0-indexed).
        val_loss:  The current validation loss (for logging).
        path:      Destination path of the .pt file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch":                epoch,
        "val_loss":             val_loss,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(path,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    device: str = "cpu"):
    """
    Loads a checkpoint and restores the model (and optionally the optimizer).

    Args:
        path:      Path to the .pt file.
        model:     Model instance (already built with the same architecture).
        optimizer: If provided, restores the optimizer state as well.
        device:    Device to map tensors to ("cpu", "cuda", "mps").

    Returns:
        (model, optimizer, epoch, val_loss)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch    = ckpt.get("epoch",    0)
    val_loss = ckpt.get("val_loss", float("inf"))

    print(f"  [Checkpoint] ← {path.name}  "
          f"(epoch={epoch + 1}, val_loss={val_loss:.6f})")
    return model, optimizer, epoch, val_loss
