
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import config
from models.world_model import WorldModel
from utils.losses import WorldModelLoss
from utils.checkpoint import save_checkpoint
from utils.early_stopping import EarlyStopping
from utils.data_utils import load_and_concat_datasets, WorldModelDataset
from utils.plot import plot_losses_train_val


torch.manual_seed(config.SEED)
np.random.seed(config.SEED)



def run_epoch(model, loader, criterion, optimizer, device,
              is_training: bool = True, use_gen_dyn: bool = False):
    model.train(is_training)
    total_loss = 0.0
    total_mu_std = 0.0
    n_batches  = 0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for img_t, img_t1, act_t in loader:
            img_t  = img_t.to(device)
            img_t1 = img_t1.to(device)
            act_t  = act_t.to(device)

            if use_gen_dyn:
                img_pred_t, mu, logvar, pred, target = model(
                    img_t, act_t, img_t1
                )
                dyn_loss   = F.mse_loss(pred, target)
                recon_loss = F.mse_loss(img_pred_t, img_t, reduction="mean")
                kl_loss    = -0.5 * torch.mean(
                    torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                )
                loss = dyn_loss + recon_loss + (config.BETA_KL * kl_loss)
            else:
                img_pred, mu, logvar, _, _ = model(img_t, act_t)
                loss, _                    = criterion(img_pred, img_t1, mu, logvar)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
            total_mu_std += mu.std().item()
            n_batches  += 1

    return total_loss / max(n_batches, 1), total_mu_std / max(n_batches, 1)



def train():
    print("=" * 65)
    print("  World Model — Training")
    print(f"  Device   : {config.DEVICE}")
    print(f"  Dataset  : {len(config.DATASETS)} file(s) → " +
          ", ".join(f"trial_{t}_{c}" for t, c in config.DATASETS))
    print(f"  Max epochs: {config.MAX_EPOCHS} | Batch: {config.BATCH_SIZE}")
    print(f"  Beta KL   : {config.BETA_KL} | LR: {config.LEARNING_RATE}")
    print("=" * 65)

    for d in [config.MODELS_DIR, config.PLOTS_DIR, config.VIDEOS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Loading and concatenating datasets...")
    splits, action_scaler, _ = load_and_concat_datasets(
        config.DATASET_DIR, config.DATASETS, config.SPLIT_RATIO
    )

    scaler_path = config.MODELS_DIR / "action_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(action_scaler, f)
    print(f"  [Data] Action scaler saved → {scaler_path}")

    action_dim = (
        splits["train"][2].shape[1]
        if config.ACTION_DIM is None
        else config.ACTION_DIM
    )
    print(f"  [Data] Detected action dim: {action_dim}")

    num_workers = 0

    train_loader = DataLoader(
        WorldModelDataset(*splits["train"]),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        WorldModelDataset(*splits["val"]),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    print("\n[3/4] Building model...")
    print(f"  [Model] Dynamics: {config.DYNAMICS_TYPE.upper()}" +
          (f" (T={config.DIFFUSION_STEPS} step)"
           if config.DYNAMICS_TYPE == "diffusion" else ""))

    model = WorldModel(
        img_channels        = config.IMG_CHANNELS,
        img_latent_dim      = config.IMG_LATENT_DIM,
        action_dim          = action_dim,
        hidden_dim          = config.HIDDEN_DIM,
        dynamics_type       = config.DYNAMICS_TYPE,
        diffusion_steps     = config.DIFFUSION_STEPS,
        diffusion_beta_start= config.DIFFUSION_BETA_START,
        diffusion_beta_end  = config.DIFFUSION_BETA_END,
    ).to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [Model] Trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = config.LEARNING_RATE,
        weight_decay = config.WEIGHT_DECAY,
    )


    criterion = WorldModelLoss(beta=config.BETA_KL).to(config.DEVICE)

    early_stopper = EarlyStopping(
        patience  = config.ES_PATIENCE,
        min_delta = config.ES_MIN_DELTA,
        verbose   = True,
    )

    best_ckpt_path = config.MODELS_DIR / "best_model.pt"

    history = {"train": [], "val": []}

    print(f"\n[4/4] Starting training...\n")
    header = (f"{'Epoca':>6} | {'Train Loss':>12} | {'Val Loss':>12} | "
              f"{'Patience':>10} | {'mu_std':>8}")
    print(header)
    print("-" * len(header))

    for epoch in range(config.MAX_EPOCHS):

        use_gen_dyn = config.DYNAMICS_TYPE in ["diffusion", "flow_matching"]
        train_loss, train_mu_std = run_epoch(
            model, train_loader, criterion, optimizer,
            config.DEVICE, is_training=True, use_gen_dyn=use_gen_dyn
        )

        val_loss, val_mu_std = run_epoch(
            model, val_loader, criterion, optimizer,
            config.DEVICE, is_training=False, use_gen_dyn=use_gen_dyn
        )

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(f"{epoch + 1:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
              f"{early_stopper.counter:>4}/{config.ES_PATIENCE} | {train_mu_std:.4f}")

        early_stopper(val_loss)

        if early_stopper.save_checkpoint:
            save_checkpoint(model, optimizer, epoch, val_loss, best_ckpt_path)

        if early_stopper.early_stop:
            print(f"\n  [Training] Early stopping at epoch {epoch + 1}.")
            break

    plot_path = config.PLOTS_DIR / "loss_curves.png"
    plot_losses_train_val(
        history["train"], history["val"],
        save_path=str(plot_path)
    )

    print("\n" + "=" * 65)
    print("  Training completed.")
    print(f"  Best model      → {best_ckpt_path}")
    print(f"  Loss curves     → {plot_path}")
    print("=" * 65)



if __name__ == "__main__":
    train()
