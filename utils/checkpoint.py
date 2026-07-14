"""
checkpoint.py — Salvataggio e ripristino dei checkpoint di training.

Salva: pesi del modello + stato dell'optimizer + epoca + val_loss.
Caricare l'optimizer state è fondamentale per preservare il momentum di Adam
e riprendere il training esattamente da dove era rimasto.
"""
import torch
from pathlib import Path


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    val_loss: float,
                    path) -> None:
    """
    Salva lo stato completo del training in un file .pt.

    Args:
        model:     Il modello PyTorch da salvare.
        optimizer: L'optimizer (include momentum, lr, ecc.).
        epoch:     L'epoca corrente (0-indexed).
        val_loss:  La validation loss corrente (per il logging).
        path:      Percorso di destinazione del file .pt.
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
    Carica un checkpoint e ripristina il modello (e opzionalmente l'optimizer).

    Args:
        path:      Percorso del file .pt.
        model:     Istanza del modello (già costruita con la stessa architettura).
        optimizer: Se fornito, ripristina anche lo stato dell'optimizer.
        device:    Device su cui mappare i tensori ("cpu", "cuda", "mps").

    Returns:
        (model, optimizer, epoch, val_loss)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch    = ckpt.get("epoch",    0)
    val_loss = ckpt.get("val_loss", float("inf"))

    print(f"  [Checkpoint] ← {path.name}  "
          f"(epoch={epoch + 1}, val_loss={val_loss:.6f})")
    return model, optimizer, epoch, val_loss
