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
from utils.plot import plot_prediction_comparison, export_comparison_video, plot_sensor_rollout

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)


def resolve_model_path(model_name: str) -> Path:
    """
    Resolves the absolute path for a given model filename.
    Appends the '.pt' extension if missing and prepends the models 
    directory if a relative path is provided.
    
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


def evaluate_vae(vae_checkpoint_name: str = config.VAE_BEST_MODEL_NAME):
    """
    Evaluates the trained VAE model on the test dataset.
    
    This function loads the test splits, initializes the WorldModel,
    loads the pre-trained VAE weights, and calculates the Mean Squared Error (MSE) 
    for image, tactile, and proprioceptive reconstructions. 
    It also generates and saves visual comparison plots and videos.
    """
    for d in [config.MODELS_DIR, config.PLOTS_DIR, config.VIDEOS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    splits, action_scaler, force_scaler, proprio_scaler, per_dataset_splits = load_and_concat_datasets(
        config.DATASET_DIR, config.DATASETS, config.SPLIT_RATIO
    )
    test_dataset = WorldModelDataset(*splits["test"])
    test_loader  = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    vae_ckpt = resolve_model_path(vae_checkpoint_name)
    if not vae_ckpt.exists():
        return

    action_dim = splits["test"][6].shape[1]
    
    tac_features = (
        splits["test"][2].shape[1]
        if config.TAC_FEATURES is None
        else config.TAC_FEATURES
    )
    
    proprio_features = (
        splits["test"][4].shape[1]
        if config.PROPRIO_FEATURES is None
        else config.PROPRIO_FEATURES
    )
    
    img_channels = (
        splits["test"][0].shape[1]
        if config.IMG_CHANNELS is None
        else config.IMG_CHANNELS
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
        dynamics_type       = config.DYNAMICS_TYPE,
        latent_type         = config.LATENT_TYPE,
        vq_num_embeddings   = config.VQ_NUM_EMBEDDINGS,
        vq_embedding_dim    = config.VQ_EMBEDDING_DIM,
        vq_commitment_cost  = config.VQ_COMMITMENT_COST,
    ).to(config.DEVICE)

    ckpt = torch.load(vae_ckpt, map_location=str(config.DEVICE), weights_only=False)
    state_dict = ckpt["model_state_dict"]
    vae_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dynamics.")}
    model.load_state_dict(vae_state_dict, strict=False)
    
    model.eval()

    print(f"Model File: {vae_ckpt.name}")

    total_mse_img = 0.0
    total_mse_tac = 0.0
    total_mse_pos = 0.0
    n_batches = 0
    with torch.no_grad():
        for img_t, _, tac_t, _, pos_t, _, _ in test_loader:
            img_t = img_t.to(config.DEVICE)
            tac_t = tac_t.to(config.DEVICE)
            pos_t = pos_t.to(config.DEVICE)
            
            if config.LATENT_TYPE == "vqvae":
                z_img_t, _, _, _ = model.visual_encoder(img_t)
            else:
                _, z_img_t, _ = model.visual_encoder(img_t)
            img_pred = model.visual_decoder(z_img_t)
            
            _, mu_tac_t, _ = model.tactile_encoder(tac_t)
            tac_pred = model.tactile_decoder(mu_tac_t)
            
            _, mu_pos_t, _ = model.proprio_encoder(pos_t)
            pos_pred = model.proprio_decoder(mu_pos_t)
            
            mse_img = F.mse_loss(img_pred, img_t).item()
            mse_tac = F.mse_loss(tac_pred, tac_t).item()
            mse_pos = F.mse_loss(pos_pred, pos_t).item()
            
            total_mse_img += mse_img
            total_mse_tac += mse_tac
            total_mse_pos += mse_pos
            n_batches += 1

    mean_mse_img_norm = total_mse_img / max(n_batches, 1)
    mean_mse_img_255 = mean_mse_img_norm * (255.0 ** 2)
    mean_mse_tac_norm = total_mse_tac / max(n_batches, 1)
    mean_mse_pos_norm = total_mse_pos / max(n_batches, 1)

    print(f"VAE Image Recon MSE   : {mean_mse_img_norm:.6f} [norm] / {mean_mse_img_255:.2f} [0-255]")
    print(f"VAE Tactile Recon MSE : {mean_mse_tac_norm:.6f} [norm]")
    print(f"VAE Proprio Recon MSE : {mean_mse_pos_norm:.6f} [norm]")

    for ds_info in per_dataset_splits[:1]:
        trial = ds_info["trial"]
        condition = ds_info["condition"]
        
        ds_test = WorldModelDataset(*ds_info["test"])
        visual_frames = min(50, len(ds_test))
        
        with torch.no_grad():
            img_t = ds_test.X_t[:visual_frames].to(config.DEVICE)
            tac_t = ds_test.S_t[:visual_frames].to(config.DEVICE)
            pos_t = ds_test.P_t[:visual_frames].to(config.DEVICE)
            
            if config.LATENT_TYPE == "vqvae":
                z_img_t, _, _, _ = model.visual_encoder(img_t)
            else:
                _, z_img_t, _ = model.visual_encoder(img_t)
            img_pred = model.visual_decoder(z_img_t)
            
            _, mu_tac_t, _ = model.tactile_encoder(tac_t)
            tac_pred = model.tactile_decoder(mu_tac_t)
            
            _, mu_pos_t, _ = model.proprio_encoder(pos_t)
            pos_pred = model.proprio_decoder(mu_pos_t)
            
            real_list = ds_test.X_t[:visual_frames]
            pred_list = img_pred.cpu()
            
            real_tac_list = ds_test.S_t[:visual_frames]
            pred_tac_list = tac_pred.cpu()
            
            real_pos_list = ds_test.P_t[:visual_frames]
            pred_pos_list = pos_pred.cpu()

        prefix = f"test_vae_recon_t{trial}_{condition}"
        
        vae_plot_path = config.PLOTS_DIR / f"{prefix}_img.png"
        plot_prediction_comparison(
            real_list,
            pred_list,
            num_frames=min(config.EVAL_VISUAL_FRAMES, visual_frames),
            title=f"VAE Recon: Trial {trial} {condition}",
            save_path=str(vae_plot_path)
        )
        
        tac_plot_path = config.PLOTS_DIR / f"{prefix}_tac.png"
        plot_sensor_rollout(
            real_tac_list,
            pred_tac_list,
            save_path=str(tac_plot_path),
            title=f"VAE Tactile Recon: Trial {trial} {condition}"
        )

        pos_plot_path = config.PLOTS_DIR / f"{prefix}_pos.png"
        plot_sensor_rollout(
            real_pos_list,
            pred_pos_list,
            save_path=str(pos_plot_path),
            title=f"VAE Proprio Recon: Trial {trial} {condition}"
        )

        vae_video_path = config.VIDEOS_DIR / f"{prefix}.mp4"
        export_comparison_video(
            real_list,
            pred_list,
            filename=str(vae_video_path),
            fps=config.VIDEO_FPS,
        )

if __name__ == "__main__":
    evaluate_vae()