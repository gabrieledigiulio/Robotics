import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

import config
from models.world_model import WorldModel
from utils.checkpoint import load_checkpoint
from utils.data_utils import load_and_concat_datasets, WorldModelDataset
from utils.plot import plot_prediction_comparison, export_comparison_video

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

def resolve_model_path(model_name: str) -> Path:
    path = Path(model_name)
    if path.suffix == "":
        path = path.with_suffix(".pt")
    if not path.is_absolute():
        path = config.MODELS_DIR / path
    return path

def evaluate_vae(vae_checkpoint_name: str = config.VAE_BEST_MODEL_NAME):
    print("=" * 65)
    print("  World Model — VAE Evaluation")
    print(f"  Device  : {config.DEVICE}")
    print("=" * 65)

    for d in [config.MODELS_DIR, config.PLOTS_DIR, config.VIDEOS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Preparing test split...")
    splits, _, per_dataset_splits = load_and_concat_datasets(
        config.DATASET_DIR, config.DATASETS, config.SPLIT_RATIO
    )
    test_dataset = WorldModelDataset(*splits["test"])
    test_loader  = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"  Test samples: {len(test_dataset)}")

    print("\n[2/4] Loading VAE checkpoint...")
    vae_ckpt = resolve_model_path(vae_checkpoint_name)
    if not vae_ckpt.exists():
        print(f"  [Error] VAE checkpoint not found: {vae_ckpt}")
        print("  Run VAE training first.")
        return

    action_dim = splits["test"][2].shape[1]
    
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

    ckpt = torch.load(vae_ckpt, map_location=str(config.DEVICE), weights_only=False)
    state_dict = ckpt["model_state_dict"]
    vae_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dynamics.")}
    model.load_state_dict(vae_state_dict, strict=False)
    
    epoch = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    model.eval()
    print(f"  Model      : VAE (Base)")
    print(f"  Checkpoint : {vae_ckpt.name}")
    print(f"  Loaded epoch {epoch + 1} | val_loss={val_loss:.6f}")

    print("\n[3/4] VAE Reconstruction Metrics")
    print("-" * 65)

    total_mse = 0.0
    n_batches = 0
    with torch.no_grad():
        for img_t, _, _ in test_loader:
            img_t = img_t.to(config.DEVICE)
            
            z_t, _, _ = model.visual_encoder(img_t)
            img_pred = model.visual_decoder(z_t)
            
            mse = F.mse_loss(img_pred, img_t).item()
            total_mse += mse
            n_batches += 1

    mean_mse_norm = total_mse / max(n_batches, 1)
    mean_mse_255 = mean_mse_norm * (255.0 ** 2)

    print(f"  VAE Reconstruction MSE : {mean_mse_norm:.6f}  [norm]")
    print(f"  VAE Reconstruction MSE : {mean_mse_255:.2f}  [0-255]")
    print("-" * 65)

    print("\n[4/4] Generating Visualizations per Dataset")
    print("-" * 65)
    
    for ds_info in per_dataset_splits:
        trial = ds_info["trial"]
        condition = ds_info["condition"]
        print(f"\n  Processing dataset: Trial {trial} - {condition}")
        
        ds_test = WorldModelDataset(*ds_info["test"])
        visual_frames = min(50, len(ds_test))
        
        with torch.no_grad():
            img_t = ds_test.X_t[:visual_frames].to(config.DEVICE)
            z_t, _, _ = model.visual_encoder(img_t)
            img_pred = model.visual_decoder(z_t)
            
            real_list = ds_test.X_t[:visual_frames]
            pred_list = img_pred.cpu()

        prefix = f"test_vae_recon_t{trial}_{condition}"
        vae_plot_path = config.PLOTS_DIR / f"{prefix}.png"
        plot_prediction_comparison(
            real_list,
            pred_list,
            num_frames=min(config.EVAL_VISUAL_FRAMES, visual_frames),
            title=f"VAE Recon: Trial {trial} {condition}",
            save_path=str(vae_plot_path)
        )
        print(f"  Saved VAE plot  : {vae_plot_path.name}")

        vae_video_path = config.VIDEOS_DIR / f"{prefix}.mp4"
        export_comparison_video(
            real_list,
            pred_list,
            filename=str(vae_video_path),
            fps=config.VIDEO_FPS,
        )
        print(f"  Saved VAE video : {vae_video_path.name}")

    print("\n  VAE Evaluation complete.")

if __name__ == "__main__":
    evaluate_vae()
