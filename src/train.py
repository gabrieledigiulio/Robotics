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
    """
    Resolves the absolute path for a given model filename.
    
    Args:
        model_name: The name or path of the model checkpoint.
        
    Returns:
        A resolved Path object pointing to the model file.
    """
    path = Path(model_name)
    if path.suffix == "":
        path = path.with_suffix(".pt")
    if not path.is_absolute():
        path = config.MODELS_DIR / path
    return path


def run_epoch(model, loader, optimizer, device, phase: str,
              is_training: bool = True):
    """
    Executes a single training or validation epoch.
    
    Args:
        model: The WorldModel instance.
        loader: DataLoader for the dataset.
        optimizer: The optimizer (used only if is_training=True).
        device: The compute device.
        phase: "vae" or "dynamics", indicating the training phase.
        is_training: Boolean flag indicating if it's a training or validation pass.
        
    Returns:
        A tuple containing the average loss, average latent standard deviation, 
        and a dictionary of detailed metrics.
    """
    model.train(is_training)
    
    if phase == "dynamics" and config.LATENT_TYPE == "vqvae":
        model.visual_encoder.eval()
        model.tactile_encoder.eval()
        model.proprio_encoder.eval()
        model.visual_decoder.eval()
        model.tactile_decoder.eval()
        model.proprio_decoder.eval()
        model.scaler.eval()

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
    
    total_dyn_img = 0.0
    total_dyn_tac = 0.0
    total_dyn_pos = 0.0

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
                
                recon_loss_img = F.mse_loss(recon_img, img_t, reduction="none").sum(dim=[1, 2, 3]).mean()
                if config.LATENT_TYPE == "vae":
                    kl_loss_img    = -0.5 * torch.sum(
                        1 + logvar_img - mu_img.pow(2) - logvar_img.exp(), dim=1
                    ).mean()
                else:
                    kl_loss_img = 0.0

                recon_loss_tac = F.smooth_l1_loss(recon_tac, tac_t, reduction="none").sum(dim=1).mean()
                kl_loss_tac    = -0.5 * torch.sum(
                    1 + logvar_tac - mu_tac.pow(2) - logvar_tac.exp(), dim=1
                ).mean()
                recon_loss_pos = F.smooth_l1_loss(recon_pos, pos_t, reduction="none").sum(dim=1).mean()
                kl_loss_pos    = -0.5 * torch.sum(
                    1 + logvar_pos - mu_pos.pow(2) - logvar_pos.exp(), dim=1
                ).mean()
                
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
                
                dim_img = model.img_latent_dim
                dim_tac = model.tac_latent_dim
                
                pred_img = dyn_pred[:, :dim_img]
                pred_tac = dyn_pred[:, dim_img : dim_img + dim_tac]
                pred_pos = dyn_pred[:, dim_img + dim_tac :]
                
                target_img = dyn_target[:, :dim_img]
                target_tac = dyn_target[:, dim_img : dim_img + dim_tac]
                target_pos = dyn_target[:, dim_img + dim_tac :]
                
                loss_img = F.mse_loss(pred_img, target_img)
                loss_tac = F.mse_loss(pred_tac, target_tac)
                loss_pos = F.mse_loss(pred_pos, target_pos)
                
                loss = loss_img * config.IMG_LOSS_WEIGHT + \
                       loss_tac * config.TAC_LOSS_WEIGHT + \
                       loss_pos * config.PROPRIO_LOSS_WEIGHT

                total_dyn_img += loss_img.item()
                total_dyn_tac += loss_tac.item()
                total_dyn_pos += loss_pos.item()

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
    elif phase == "dynamics":
        stats["dyn_img"] = total_dyn_img / max(n_batches, 1)
        stats["dyn_tac"] = total_dyn_tac / max(n_batches, 1)
        stats["dyn_pos"] = total_dyn_pos / max(n_batches, 1)

    return total_loss / max(n_batches, 1), total_mu_std / max(n_batches, 1), stats


def train(phase: str = config.TRAIN_PHASE,
          vae_best_name: str = config.VAE_BEST_MODEL_NAME,
          dynamics_best_name: str = config.DYNAMICS_BEST_MODEL_NAME):
    """
    Main training loop for the World Model.
    
    Handles both the VAE and Dynamics phases, initializes the dataset, 
    loads checkpoints if needed, manages early stopping, and logs epoch stats.
    """
    print(f"Phase: {phase.upper()}")

    for d in [config.MODELS_DIR, config.PLOTS_DIR, config.VIDEOS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

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
            sys.exit(1)
            
        print(f"Model File: {vae_ckpt.name}")
        ckpt = torch.load(vae_ckpt, map_location=str(config.DEVICE), weights_only=False)
        state_dict = ckpt["model_state_dict"]
        
        vae_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dynamics.")}
        
        model.load_state_dict(vae_state_dict, strict=False)

        for p in model.visual_encoder.parameters(): p.requires_grad = False
        for p in model.visual_decoder.parameters(): p.requires_grad = False
        for p in model.tactile_encoder.parameters(): p.requires_grad = False
        for p in model.tactile_decoder.parameters(): p.requires_grad = False
        for p in model.proprio_encoder.parameters(): p.requires_grad = False
        for p in model.proprio_decoder.parameters(): p.requires_grad = False

        with torch.no_grad():
            if config.LATENT_TYPE == "vqvae":
                all_z = []
                for img_t, _, tac_t, _, pos_t, _, _ in train_loader:
                    img_t = img_t.to(config.DEVICE)
                    tac_t = tac_t.to(config.DEVICE)
                    pos_t = pos_t.to(config.DEVICE)
                    
                    _, mu_img_t, _, _ = model.visual_encoder(img_t)
                    _, mu_tac_t, _ = model.tactile_encoder(tac_t)
                    _, mu_pos_t, _ = model.proprio_encoder(pos_t)
                    
                    z_t = torch.cat([mu_img_t, mu_tac_t, mu_pos_t], dim=-1)
                    all_z.append(z_t.cpu())
                    
                all_z = torch.cat(all_z, dim=0)
                model.scaler.mean.copy_(all_z.mean(dim=0).to(config.DEVICE))
                model.scaler.var.copy_(all_z.var(dim=0, unbiased=False).to(config.DEVICE))
                model.scaler.is_fitted = True
            
            else:
                for img_t, _, tac_t, _, pos_t, _, _ in train_loader:
                    img_t = img_t.to(config.DEVICE)
                    tac_t = tac_t.to(config.DEVICE)
                    pos_t = pos_t.to(config.DEVICE)
                    
                    _, z_img_t, _ = model.visual_encoder(img_t)
                    _, z_tac_t, _ = model.tactile_encoder(tac_t)
                    _, z_pos_t, _ = model.proprio_encoder(pos_t)
                    
                    z_t = torch.cat([z_img_t, z_tac_t, z_pos_t], dim=-1)
                    
                    model.scaler.train() 
                    model.scaler(z_t) 
                
        model.scaler.freeze = True
        model.scaler.eval()

        best_ckpt_path = resolve_model_path(dynamics_best_name)
        plot_path      = config.PLOTS_DIR / "loss_curves_dynamics.png"

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.Adam(
        trainable_params,
        lr           = config.LEARNING_RATE,
        weight_decay = config.WEIGHT_DECAY,
    )

    start_epoch = 0

    early_stopper = EarlyStopping(
        patience  = config.ES_PATIENCE,
        min_delta = config.ES_MIN_DELTA,
        verbose   = True,
    )

    history = {"train": [], "val": []}

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
        elif phase == "dynamics":
            print(f"         ↳ Dyn MSE (val): Img: {val_stats.get('dyn_img', 0):.4f} | Tac: {val_stats.get('dyn_tac', 0):.4f} | Pos: {val_stats.get('dyn_pos', 0):.4f}")

        early_stopper(val_loss, epoch=epoch + 1)
        if early_stopper.save_checkpoint:
            save_checkpoint(model, optimizer, epoch, val_loss, best_ckpt_path)

        if early_stopper.early_stop:
            print(f"\n  [Training] Early stopping at epoch {epoch + 1}.")
            break

    plot_losses_train_val(history["train"], history["val"], save_path=str(plot_path), start_epoch=1, best_epoch=early_stopper.best_epoch)

    print(f"\nTraining {phase.upper()} completed.")
    print(f"Saved model → {best_ckpt_path.name}")


if __name__ == "__main__":
    train(
        phase=config.TRAIN_PHASE
    )
