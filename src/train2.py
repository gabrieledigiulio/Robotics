
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
    total_recon_img = 0.0
    total_vq_img = 0.0
    total_perplexity = 0.0
    total_recon_tac = 0.0
    total_kl_tac = 0.0
    total_recon_pos = 0.0
    total_kl_pos = 0.0

    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for img_t, img_t1, tac_t, tac_t1, pos_t, pos_t1, act_t in loader:
            img_t  = img_t.to(device)
            img_t1 = img_t1.to(device)
            tac_t  = tac_t.to(device)
            tac_t1 = tac_t1.to(device)
            pos_t  = pos_t.to(device)
            pos_t1 = pos_t1.to(device)
            act_t  = act_t.to(device)

            if phase == "vae":
                
                recon_img, recon_tac, recon_pos, mu_img, logvar_img, mu_tac, logvar_tac, mu_pos, logvar_pos, _, _, vq_loss_img, perplexity_img = model(
                    img_t, tac_t, pos_t, act_t, img_t1, tac_t1, pos_t1
                )
                
                recon_loss_img = F.mse_loss(recon_img, img_t, reduction="mean")
                if config.LATENT_TYPE == "vae":
                    kl_loss_img    = -0.5 * torch.mean(
                        torch.mean(1 + logvar_img - mu_img.pow(2) - logvar_img.exp(), dim=1)
                    )
                else:
                    kl_loss_img = 0.0

                recon_loss_tac = F.smooth_l1_loss(recon_tac, tac_t, reduction="mean")
                kl_loss_tac    = -0.5 * torch.mean(
                    torch.mean(1 + logvar_tac - mu_tac.pow(2) - logvar_tac.exp(), dim=1)
                )
                recon_loss_pos = F.smooth_l1_loss(recon_pos, pos_t, reduction="mean")
                kl_loss_pos    = -0.5 * torch.mean(
                    torch.mean(1 + logvar_pos - mu_pos.pow(2) - logvar_pos.exp(), dim=1)
                )
                
                if config.LATENT_TYPE == "vqvae":
                    loss = (recon_loss_img + vq_loss_img) + \
                           (recon_loss_tac + config.BETA_KL * kl_loss_tac) * config.TAC_LOSS_WEIGHT + \
                           (recon_loss_pos + config.BETA_KL * kl_loss_pos) * config.PROPRIO_LOSS_WEIGHT
                else:
                    loss = (recon_loss_img + config.BETA_KL * kl_loss_img) + \
                           (recon_loss_tac + config.BETA_KL * kl_loss_tac) * config.TAC_LOSS_WEIGHT + \
                           (recon_loss_pos + config.BETA_KL * kl_loss_pos) * config.PROPRIO_LOSS_WEIGHT
                
                mu_std = (mu_img.std().item() + mu_tac.std().item() + mu_pos.std().item()) / 3.0
                
                total_recon_img += recon_loss_img.item()
                if isinstance(vq_loss_img, torch.Tensor):
                    total_vq_img += vq_loss_img.item()
                else:
                    total_vq_img += float(vq_loss_img)
                
                total_recon_tac += recon_loss_tac.item()
                total_kl_tac += kl_loss_tac.item()
                
                total_recon_pos += recon_loss_pos.item()
                total_kl_pos += kl_loss_pos.item()
                
                if isinstance(perplexity_img, torch.Tensor):
                    total_perplexity += perplexity_img.item()
                else:
                    total_perplexity += float(perplexity_img)

            elif phase == "dynamics":
                
                _, _, _, mu_img, _, mu_tac, _, mu_pos, _, dyn_pred, dyn_target, _, _ = model(
                    img_t, tac_t, pos_t, act_t, img_t1, tac_t1, pos_t1
                )
                
                loss = F.mse_loss(dyn_pred, dyn_target)
                mu_t = torch.cat([mu_img, mu_tac, mu_pos], dim=-1)
                mu_std = mu_t.std().item()

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                
                if phase == "vae":
                    img_params = list(model.visual_encoder.parameters()) + list(model.visual_decoder.parameters())
                    other_params = [p for n, p in model.named_parameters() if not n.startswith(("visual_encoder", "visual_decoder"))]
                    
                    clip_grad_norm_(img_params, max_norm=config.GRAD_CLIP)
                    if len(other_params) > 0:
                        clip_grad_norm_(other_params, max_norm=config.GRAD_CLIP)
                else:
                    clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP)
                    
                optimizer.step()

            total_loss += loss.item()
            total_mu_std += mu_std
            n_batches  += 1

    stats = {}
    if phase == "vae":
        stats["recon_img"] = total_recon_img / max(n_batches, 1)
        stats["vq_img"] = total_vq_img / max(n_batches, 1)
        stats["perplexity"] = total_perplexity / max(n_batches, 1)
        stats["recon_tac"] = total_recon_tac / max(n_batches, 1)
        stats["kl_tac"] = total_kl_tac / max(n_batches, 1)
        stats["recon_pos"] = total_recon_pos / max(n_batches, 1)
        stats["kl_pos"] = total_kl_pos / max(n_batches, 1)

    return total_loss / max(n_batches, 1), total_mu_std / max(n_batches, 1), stats



def train(phase: str = config.TRAIN_PHASE,
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
    splits, action_scaler, force_scaler, proprio_scaler, _ = load_and_concat_datasets(
        config.DATASET_DIR, config.DATASETS, config.SPLIT_RATIO
    )

    scaler_path = config.MODELS_DIR / "action_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(action_scaler, f)
        
    force_scaler_path = config.MODELS_DIR / "force_scaler.pkl"
    with open(force_scaler_path, "wb") as f:
        pickle.dump(force_scaler, f)
        
    proprio_scaler_path = config.MODELS_DIR / "proprio_scaler.pkl"
    with open(proprio_scaler_path, "wb") as f:
        pickle.dump(proprio_scaler, f)

    action_dim = (
        splits["train"][6].shape[1]
        if config.ACTION_DIM is None
        else config.ACTION_DIM
    )
    
    tac_features = (
        splits["train"][2].shape[1]
        if config.TAC_FEATURES is None
        else config.TAC_FEATURES
    )
    
    proprio_features = (
        splits["train"][4].shape[1]
        if config.PROPRIO_FEATURES is None
        else config.PROPRIO_FEATURES
    )
    
    img_channels = (
        splits["train"][0].shape[1]
        if config.IMG_CHANNELS is None
        else config.IMG_CHANNELS
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
        img_channels        = img_channels,
        img_latent_dim      = config.IMG_LATENT_DIM,
        tac_features        = tac_features,
        tac_latent_dim      = config.TAC_LATENT_DIM,
        proprio_features    = proprio_features,
        proprio_latent_dim  = config.PROPRIO_LATENT_DIM,
        action_dim          = action_dim,
        hidden_dim          = config.HIDDEN_DIM,
        dynamics_hidden_dim = config.DYNAMICS_HIDDEN_DIM,
        dynamics_type       = config.DYNAMICS_TYPE,
        latent_type         = config.LATENT_TYPE,
        vq_num_embeddings   = config.VQ_NUM_EMBEDDINGS,
        vq_embedding_dim    = config.VQ_EMBEDDING_DIM,
        vq_commitment_cost  = config.VQ_COMMITMENT_COST,
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
        for p in model.tactile_encoder.parameters(): p.requires_grad = False
        for p in model.tactile_decoder.parameters(): p.requires_grad = False
        for p in model.proprio_encoder.parameters(): p.requires_grad = False
        for p in model.proprio_decoder.parameters(): p.requires_grad = False

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
    print(f"\n[3/4] Starting new training from epoch 0.")

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

        train_loss, train_mu_std, train_stats = run_epoch(
            model, train_loader, optimizer, config.DEVICE, phase, is_training=True
        )

        val_loss, val_mu_std, val_stats = run_epoch(
            model, val_loader, optimizer, config.DEVICE, phase, is_training=False
        )

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(f"{epoch + 1:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
              f"{early_stopper.counter:>4}/{config.ES_PATIENCE} | {train_mu_std:.4f}")
        
        if phase == "vae":
            print(f"         ↳ Img Recon (val): {val_stats.get('recon_img', 0):.4f} | Img VQ/KL (val): {val_stats.get('vq_img', 0):.4f} | Perplexity: {val_stats.get('perplexity', 0):.1f}/{config.VQ_NUM_EMBEDDINGS}")
            print(f"         ↳ Tac Recon (val): {val_stats.get('recon_tac', 0):.4f} | Tac KL (val): {val_stats.get('kl_tac', 0):.4f}")
            print(f"         ↳ Pos Recon (val): {val_stats.get('recon_pos', 0):.4f} | Pos KL (val): {val_stats.get('kl_pos', 0):.4f}")

        early_stopper(val_loss)
        if early_stopper.save_checkpoint:
            save_checkpoint(model, optimizer, epoch, val_loss, best_ckpt_path)

        if early_stopper.early_stop:
            print(f"\n  [Training] Early stopping at epoch {epoch + 1}.")
            break

    plot_losses_train_val(history["train"], history["val"], save_path=str(plot_path), start_epoch=1)

    print("\n" + "=" * 65)
    print(f"  Training {phase.upper()} completed.")
    print(f"  Saved model → {best_ckpt_path}")
    print("=" * 65)


if __name__ == "__main__":
    train(
        phase=config.TRAIN_PHASE
    )