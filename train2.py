
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from pathlib import Path

import config
from models.world_model import WorldModel
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.early_stopping import EarlyStopping
from utils.data_utils import load_and_concat_datasets, WorldModelDataset
from utils.plot import plot_losses_train_val


torch.manual_seed(config.SEED)
np.random.seed(config.SEED)




def resolve_model_path(model_name: str) -> Path:
    path = Path(model_name)
    if path.suffix == "":
        path = path.with_suffix(".pt")
    if not path.is_absolute():
        path = config.MODELS_DIR / path
    return path

def run_epoch(model, loader, optimizer, device, phase: str,
              is_training: bool = True):
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

            if phase == "vae":
                
                z_t, mu, logvar = model.visual_encoder(img_t)
                
                img_pred_t = model.visual_decoder(z_t)
                
                n_pixels = img_t.shape[1] * img_t.shape[2] * img_t.shape[3]
                recon_loss = F.mse_loss(img_pred_t, img_t, reduction="mean") * n_pixels
                kl_loss    = -0.5 * torch.mean(
                    torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                )
                
                loss = recon_loss + (config.BETA_KL * kl_loss)
                mu_std = mu.std().item()

            elif phase == "dynamics":
                
                with torch.no_grad():
                    _, mu_t, _  = model.visual_encoder(img_t)
                    _, mu_t1, _ = model.visual_encoder(img_t1)
                
                if config.DYNAMICS_TYPE == "mlp":
                    z_next_pred = model.dynamics(mu_t, act_t)
                    loss = F.mse_loss(z_next_pred, mu_t1)
                else:
                    pred, target = model.dynamics(mu_t, act_t, z_next=mu_t1)
                    loss = F.mse_loss(pred, target)
                
                mu_std = mu_t.std().item()

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
            total_mu_std += mu_std
            n_batches  += 1

    return total_loss / max(n_batches, 1), total_mu_std / max(n_batches, 1)



def train(phase: str, resume: bool = False,
          vae_best_name: str = config.VAE_BEST_MODEL_NAME,
          dynamics_best_name: str = config.DYNAMICS_BEST_MODEL_NAME):
    print("=" * 65)
    print(f"  World Model — Training (Phase: {phase.upper()})")
    print(f"  Device   : {config.DEVICE}")
    print(f"  Dataset  : {len(config.DATASETS)} file(s) → " +
          ", ".join(f"trial_{t}_{c}" for t, c in config.DATASETS))
    print(f"  Max epochs: {config.MAX_EPOCHS} | Batch: {config.BATCH_SIZE}")
    print(f"  LR        : {config.LEARNING_RATE}")
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

    action_dim = (
        splits["train"][2].shape[1]
        if config.ACTION_DIM is None
        else config.ACTION_DIM
    )

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

    print(f"\n[2/4] Costruzione modello per fase {phase.upper()}...")

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

    if phase == "vae":
        for p in model.dynamics.parameters():
            p.requires_grad = False
        
        best_ckpt_path = resolve_model_path(vae_best_name)
        plot_path      = config.PLOTS_DIR / "loss_curves_vae.png"

    elif phase == "dynamics":
        vae_ckpt = resolve_model_path(vae_best_name)
        if not vae_ckpt.exists():
            print(f"\n[FATAL ERROR] Missing base VAE model: {vae_ckpt}")
            print("Run first: python train2.py (with TRAIN_PHASE = \"vae\")")
            sys.exit(1)
            
        print(f"  [Model] Loading pretrained VAE weights from {vae_ckpt.name}...")
        ckpt = torch.load(vae_ckpt, map_location=str(config.DEVICE), weights_only=False)
        state_dict = ckpt["model_state_dict"]
        
        vae_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dynamics.")}
        
        model.load_state_dict(vae_state_dict, strict=False)
        print("  [Model] Encoder and decoder weights loaded successfully (dynamics ignored).")

        for p in model.visual_encoder.parameters(): p.requires_grad = False
        for p in model.visual_decoder.parameters(): p.requires_grad = False

        best_ckpt_path = resolve_model_path(dynamics_best_name)
        plot_path      = config.PLOTS_DIR / "loss_curves_dynamics.png"

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  [Model] Trainable parameters in this phase: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.Adam(
        trainable_params,
        lr           = config.LEARNING_RATE,
        weight_decay = config.WEIGHT_DECAY,
    )

    start_epoch = 0
    if resume:
        print(f"\n[3/4] Resuming from the best checkpoint of phase {phase}...")
        model, optimizer, start_epoch, _ = load_checkpoint(
            best_ckpt_path, model, optimizer, device=str(config.DEVICE)
        )
        start_epoch += 1
    else:
        print(f"\n[3/4] No resume requested.")

    early_stopper = EarlyStopping(
        patience  = config.ES_PATIENCE,
        min_delta = config.ES_MIN_DELTA,
        verbose   = True,
    )

    history = {"train": [], "val": []}

    print(f"\n[4/4] Starting training for phase {phase.upper()}...\n")
    header = (f"{'Epoca':>6} | {'Train Loss':>12} | {'Val Loss':>12} | "
              f"{'Patience':>10} | {'mu_std':>8}")
    print(header)
    print("-" * len(header))

    for epoch in range(start_epoch, config.MAX_EPOCHS):

        train_loss, train_mu_std = run_epoch(
            model, train_loader, optimizer, config.DEVICE, phase, is_training=True
        )

        val_loss, val_mu_std = run_epoch(
            model, val_loader, optimizer, config.DEVICE, phase, is_training=False
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

    plot_losses_train_val(history["train"], history["val"], save_path=str(plot_path))

    print("\n" + "=" * 65)
    print(f"  Training {phase.upper()} completed.")
    print(f"  Saved model → {best_ckpt_path}")
    print("=" * 65)


if __name__ == "__main__":
    train(
        phase=config.TRAIN_PHASE,
        resume=config.TRAIN_RESUME,
    )