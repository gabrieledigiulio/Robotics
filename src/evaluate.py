import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

import config
from models.world_model import WorldModel
from utils.checkpoint import load_checkpoint
from utils.data_utils import load_and_concat_datasets, WorldModelDataset
from utils.plot import (
    plot_prediction_comparison,
    plot_video_rollout,
    export_comparison_video,
    export_triple_comparison_video,
    plot_sensor_rollout,
)

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


def _spaced_start_indices(max_start: int, num_starts: int) -> list[int]:
    """
    Generates a list of evenly spaced starting indices for rollout evaluation.
    
    Args:
        max_start: The maximum possible starting index.
        num_starts: The desired number of starting points.
        
    Returns:
        A sorted list of unique integer indices.
    """
    if max_start <= 0:
        return [0]
    num_starts = max(1, min(num_starts, max_start + 1))
    starts = np.linspace(0, max_start, num=num_starts, dtype=int)
    return np.unique(starts).tolist()


def compute_mse_1step(model, loader, device, num_samples: int = 1):
    """
    Computes the 1-step prediction Mean Squared Error (MSE) for image, 
    tactile, and proprioceptive modalities over the entire dataset.
    
    Args:
        model: The trained World Model.
        loader: DataLoader containing the test data.
        device: The compute device (CPU or GPU).
        num_samples: Number of stochastic forward passes to average.
        
    Returns:
        A tuple of (mean, std) for image, tactile, and proprioceptive MSEs.
    """
    model.eval()
    sample_mses_img = []
    sample_mses_tac = []
    sample_mses_pos = []

    with torch.no_grad():
        for img_t, img_t1, tac_t, tac_t1, pos_t, pos_t1, act_t in loader:
            img_t  = img_t.to(device)
            img_t1 = img_t1.to(device)
            tac_t  = tac_t.to(device)
            tac_t1 = tac_t1.to(device)
            pos_t  = pos_t.to(device)
            pos_t1 = pos_t1.to(device)
            act_t  = act_t.to(device)

            batch_samples_img = []
            batch_samples_tac = []
            batch_samples_pos = []
            for _ in range(max(1, num_samples)):
                img_pred, tac_pred, pos_pred, _, _, _, _, _, _, _, _, _, _ = model(img_t, tac_t, pos_t, act_t)
                batch_samples_img.append(F.mse_loss(img_pred, img_t1, reduction="mean").item())
                batch_samples_tac.append(F.mse_loss(tac_pred, tac_t1, reduction="mean").item())
                batch_samples_pos.append(F.mse_loss(pos_pred, pos_t1, reduction="mean").item())

            sample_mses_img.extend(batch_samples_img)
            sample_mses_tac.extend(batch_samples_tac)
            sample_mses_pos.extend(batch_samples_pos)

    mse_img_norm = float(np.mean(sample_mses_img)) if sample_mses_img else float("nan")
    mse_img_std  = float(np.std(sample_mses_img)) if sample_mses_img else float("nan")
    mse_img_255  = mse_img_norm * (255.0 ** 2)
    mse_img_255_std = mse_img_std * (255.0 ** 2)
    
    mse_tac_norm = float(np.mean(sample_mses_tac)) if sample_mses_tac else float("nan")
    mse_tac_std = float(np.std(sample_mses_tac)) if sample_mses_tac else float("nan")
    
    mse_pos_norm = float(np.mean(sample_mses_pos)) if sample_mses_pos else float("nan")
    mse_pos_std = float(np.std(sample_mses_pos)) if sample_mses_pos else float("nan")
    
    return mse_img_norm, mse_img_std, mse_img_255, mse_img_255_std, mse_tac_norm, mse_tac_std, mse_pos_norm, mse_pos_std


def compute_mse_rollout(model, test_dataset, device, n_steps: int,
                        start_idx: int = 0, num_samples: int = 1):
    """
    Computes the multi-step rollout Mean Squared Error (MSE) over a sequence 
    of a specified length, returning the error curves and predicted frames.
    
    Args:
        model: The trained World Model.
        test_dataset: The test dataset.
        device: The compute device.
        n_steps: Number of future steps to predict.
        start_idx: The index to start the rollout from.
        num_samples: Number of stochastic rollouts to average.
        
    Returns:
        Mean and std of MSE curves for each modality, along with raw real/predicted sequences.
    """
    model.eval()
    max_start = len(test_dataset) - n_steps

    if max_start < 0:
        n_steps = len(test_dataset) - 1
        max_start = 0

    img_0   = test_dataset.X_t[start_idx].unsqueeze(0).to(device)
    tac_0   = test_dataset.S_t[start_idx].unsqueeze(0).to(device)
    pos_0   = test_dataset.P_t[start_idx].unsqueeze(0).to(device)
    actions = test_dataset.A_t[start_idx:start_idx + n_steps].unsqueeze(0).to(device)
    real_frames = test_dataset.X_t1[start_idx:start_idx + n_steps]
    real_tacs = test_dataset.S_t1[start_idx:start_idx + n_steps]
    real_pos = test_dataset.P_t1[start_idx:start_idx + n_steps]

    sample_curves_img = []
    sample_curves_tac = []
    sample_curves_pos = []
    pred_frames_out = None
    pred_tacs_out = None
    pred_pos_out = None

    with torch.no_grad():
        for _ in range(max(1, num_samples)):
            pred_sequence, pred_tac_sequence, pred_pos_sequence = model.rollout(img_0, tac_0, pos_0, actions)
            pred_frames = pred_sequence.squeeze(0).cpu()
            pred_tacs = pred_tac_sequence.squeeze(0).cpu()
            pred_pos = pred_pos_sequence.squeeze(0).cpu()

            mse_per_step_img = []
            mse_per_step_tac = []
            mse_per_step_pos = []
            for t in range(n_steps):
                mse_img_t = F.mse_loss(pred_frames[t], real_frames[t]).item()
                mse_tac_t = F.mse_loss(pred_tacs[t], real_tacs[t]).item()
                mse_pos_t = F.mse_loss(pred_pos[t], real_pos[t]).item()
                mse_per_step_img.append(mse_img_t)
                mse_per_step_tac.append(mse_tac_t)
                mse_per_step_pos.append(mse_pos_t)
            sample_curves_img.append(mse_per_step_img)
            sample_curves_tac.append(mse_per_step_tac)
            sample_curves_pos.append(mse_per_step_pos)

            if pred_frames_out is None:
                pred_frames_out = pred_frames
                pred_tacs_out = pred_tacs
                pred_pos_out = pred_pos

    mse_mean_img = np.mean(sample_curves_img, axis=0).tolist()
    mse_std_img = np.std(sample_curves_img, axis=0).tolist()
    mse_mean_tac = np.mean(sample_curves_tac, axis=0).tolist()
    mse_std_tac = np.std(sample_curves_tac, axis=0).tolist()
    mse_mean_pos = np.mean(sample_curves_pos, axis=0).tolist()
    mse_std_pos = np.std(sample_curves_pos, axis=0).tolist()
    
    return mse_mean_img, mse_std_img, mse_mean_tac, mse_std_tac, mse_mean_pos, mse_std_pos, real_frames, pred_frames_out, real_tacs, pred_tacs_out, real_pos, pred_pos_out


def compute_one_step_sequence(model, test_dataset, device, start_idx: int,
                              n_frames: int, num_samples: int = 1):
    """
    Generates a sequence of 1-step predictions starting from a specific index 
    to be used for visual or metric comparison against the ground truth.
    
    Args:
        model: The trained World Model.
        test_dataset: The test dataset.
        device: The compute device.
        start_idx: The starting index.
        n_frames: The sequence length.
        num_samples: Number of stochastic predictions to average per step.
        
    Returns:
        Real and predicted sequence tensors for all modalities.
    """
    model.eval()
    real_frames = []
    pred_frames = []
    real_tacs = []
    pred_tacs_out = []
    real_pos_out = []
    pred_pos_out = []

    with torch.no_grad():
        for offset in range(n_frames):
            idx = start_idx + offset
            img_t = test_dataset.X_t[idx].unsqueeze(0).to(device)
            tac_t = test_dataset.S_t[idx].unsqueeze(0).to(device)
            pos_t = test_dataset.P_t[idx].unsqueeze(0).to(device)
            act_t = test_dataset.A_t[idx].unsqueeze(0).to(device)
            real_frames.append(test_dataset.X_t1[idx])
            real_tacs.append(test_dataset.S_t1[idx])
            real_pos_out.append(test_dataset.P_t1[idx])

            pred_samples_img = []
            pred_samples_tac = []
            pred_samples_pos = []
            for _ in range(max(1, num_samples)):
                img_pred, tac_pred, pos_pred, _, _, _, _, _, _, _, _, _, _ = model(img_t, tac_t, pos_t, act_t)
                pred_samples_img.append(img_pred.squeeze(0).cpu())
                pred_samples_tac.append(tac_pred.squeeze(0).cpu())
                pred_samples_pos.append(pos_pred.squeeze(0).cpu())

            pred_frames.append(torch.stack(pred_samples_img, dim=0).mean(dim=0))
            pred_tacs_out.append(torch.stack(pred_samples_tac, dim=0).mean(dim=0))
            pred_pos_out.append(torch.stack(pred_samples_pos, dim=0).mean(dim=0))

    return torch.stack(real_frames, dim=0), torch.stack(pred_frames, dim=0), torch.stack(real_tacs, dim=0), torch.stack(pred_tacs_out, dim=0), torch.stack(real_pos_out, dim=0), torch.stack(pred_pos_out, dim=0)


def evaluate(dynamics_checkpoint_name: str = config.DYNAMICS_BEST_MODEL_NAME):
    """
    Main evaluation routine. Loads the trained World Model, computes 1-step 
    and rollout metrics on the test dataset, prints the final numerical results, 
    and exports visualization plots and comparison videos.
    """
    dynamics_label = config.DYNAMICS_TYPE.upper()
        
    print(f"Model: World Model | Dynamics: {dynamics_label}")

    for d in [config.MODELS_DIR, config.PLOTS_DIR, config.VIDEOS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    splits, action_scaler, force_scaler, proprio_scaler, per_dataset_splits = load_and_concat_datasets(
        config.DATASET_DIR, config.DATASETS, config.SPLIT_RATIO
    )

    test_dataset = WorldModelDataset(*splits["test"])
    test_loader  = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    action_dim = splits["test"][6].shape[1]
    tac_features = splits["test"][2].shape[1] if config.TAC_FEATURES is None else config.TAC_FEATURES
    proprio_features = splits["test"][4].shape[1] if config.PROPRIO_FEATURES is None else config.PROPRIO_FEATURES
    img_channels = splits["test"][0].shape[1] if config.IMG_CHANNELS is None else config.IMG_CHANNELS

    dynamics_ckpt = resolve_model_path(dynamics_checkpoint_name)
    if not dynamics_ckpt.exists():
        sys.exit(f"Error: Dynamics checkpoint not found: {dynamics_ckpt}")

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
        vq_commitment_cost  = config.VQ_COMMITMENT_COST
    ).to(config.DEVICE)

    model, _, _, _ = load_checkpoint(dynamics_ckpt, model, optimizer=None, device=str(config.DEVICE))
    model.eval()

    mse_img_norm, mse_img_std, mse_img_255, mse_img_255_std, mse_tac_norm, mse_tac_std, mse_pos_norm, mse_pos_std = compute_mse_1step(
        model, test_loader, config.DEVICE,
        num_samples=config.EVAL_ONE_STEP_SAMPLES
    )

    n_rollout = min(config.ROLLOUT_STEPS, len(test_dataset) - 1)
    max_start = len(test_dataset) - n_rollout
    
    if max_start >= 0:
        rollout_starts = _spaced_start_indices(max_start, config.EVAL_ROLLOUT_STARTS)
        per_start_curves_img = []
        per_start_curves_tac = []
        per_start_curves_pos = []
        
        for start in rollout_starts:
            mse_mean_img, _, mse_mean_tac, _, mse_mean_pos, _, _, _, _, _, _, _ = compute_mse_rollout(
                model,
                test_dataset,
                config.DEVICE,
                n_steps=n_rollout,
                start_idx=start,
                num_samples=config.EVAL_ROLLOUT_SAMPLES,
            )
            per_start_curves_img.append(mse_mean_img)
            per_start_curves_tac.append(mse_mean_tac)
            per_start_curves_pos.append(mse_mean_pos)

        rollout_curve_img_mean = np.mean(per_start_curves_img, axis=0)
        rollout_curve_tac_mean = np.mean(per_start_curves_tac, axis=0)
        rollout_curve_pos_mean = np.mean(per_start_curves_pos, axis=0)
        
        rollout_img_mean = float(np.mean(rollout_curve_img_mean))
        rollout_img_final = float(rollout_curve_img_mean[-1])
        rollout_img_auc = float(np.trapezoid(rollout_curve_img_mean, dx=1.0) / max(1, n_rollout - 1))
        
        rollout_tac_mean = float(np.mean(rollout_curve_tac_mean))
        rollout_tac_final = float(rollout_curve_tac_mean[-1])

        rollout_pos_mean = float(np.mean(rollout_curve_pos_mean))
        rollout_pos_final = float(rollout_curve_pos_mean[-1])

        step10_idx = min(9, n_rollout - 1)
        step25_idx = min(24, n_rollout - 1)
        
        img_step10 = float(rollout_curve_img_mean[step10_idx])
        img_step25 = float(rollout_curve_img_mean[step25_idx])
        
        tac_step10 = float(rollout_curve_tac_mean[step10_idx])
        tac_step25 = float(rollout_curve_tac_mean[step25_idx])

        pos_step10 = float(rollout_curve_pos_mean[step10_idx])
        pos_step25 = float(rollout_curve_pos_mean[step25_idx])
        
        print("\n--- 1-Step MSE ---")
        print(f"Vision (Norm)   : {mse_img_norm:.6f} ± {mse_img_std:.6f}")
        print(f"Vision (255)    : {mse_img_255:.2f} ± {mse_img_255_std:.2f}")
        print(f"Tactile (Norm)  : {mse_tac_norm:.6f} ± {mse_tac_std:.6f}")
        print(f"Proprio (Norm)  : {mse_pos_norm:.6f} ± {mse_pos_std:.6f}")
        
        print(f"\n--- Rollout MSE Progression ({n_rollout} steps total) ---")
        print(f"Img MSE @ step 10 : {img_step10:.6f}")
        print(f"Img MSE @ step 25 : {img_step25:.6f}")
        print(f"Img MSE final     : {rollout_img_final:.6f}")
        print(f"Img mean MSE      : {rollout_img_mean:.6f}")
        print(f"Img AUC           : {rollout_img_auc:.6f}\n")
        
        print(f"Tac MSE @ step 10 : {tac_step10:.6f}")
        print(f"Tac MSE @ step 25 : {tac_step25:.6f}")
        print(f"Tac MSE final     : {rollout_tac_final:.6f}")
        print(f"Tac mean MSE      : {rollout_tac_mean:.6f}\n")
        
        print(f"Pos MSE @ step 10 : {pos_step10:.6f}")
        print(f"Pos MSE @ step 25 : {pos_step25:.6f}")
        print(f"Pos MSE final     : {rollout_pos_final:.6f}")
        print(f"Pos mean MSE      : {rollout_pos_mean:.6f}\n")

        metrics_file = config.OUTPUTS_DIR / f"eval_metrics_{config.DYNAMICS_TYPE}.txt"
        with open(metrics_file, "w") as f:
            f.write("1-Step MSE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Vision (Norm)   : {mse_img_norm:.6f} ± {mse_img_std:.6f}\n")
            f.write(f"Vision (255)    : {mse_img_255:.2f} ± {mse_img_255_std:.2f}\n")
            f.write(f"Tactile (Norm)  : {mse_tac_norm:.6f} ± {mse_tac_std:.6f}\n")
            f.write(f"Proprio (Norm)  : {mse_pos_norm:.6f} ± {mse_pos_std:.6f}\n\n")
            f.write(f"Rollout MSE Progression ({n_rollout} steps total)\n")
            f.write("-" * 30 + "\n")
            f.write(f"Img MSE @ step 10 : {img_step10:.6f}\n")
            f.write(f"Img MSE @ step 25 : {img_step25:.6f}\n")
            f.write(f"Img MSE final     : {rollout_img_final:.6f}\n")
            f.write(f"Img mean MSE      : {rollout_img_mean:.6f}\n")
            f.write(f"Img AUC           : {rollout_img_auc:.6f}\n\n")
            f.write(f"Tac MSE @ step 10 : {tac_step10:.6f}\n")
            f.write(f"Tac MSE @ step 25 : {tac_step25:.6f}\n")
            f.write(f"Tac MSE final     : {rollout_tac_final:.6f}\n")
            f.write(f"Tac mean MSE      : {rollout_tac_mean:.6f}\n\n")
            f.write(f"Pos MSE @ step 10 : {pos_step10:.6f}\n")
            f.write(f"Pos MSE @ step 25 : {pos_step25:.6f}\n")
            f.write(f"Pos MSE final     : {rollout_pos_final:.6f}\n")
            f.write(f"Pos mean MSE      : {rollout_pos_mean:.6f}\n")

    visual_start = max(0, min(config.EVAL_VISUAL_START, len(test_dataset) - n_rollout - 1))

    one_step_real_img, one_step_pred_img, one_step_real_tac, one_step_pred_tac, one_step_real_pos, one_step_pred_pos = compute_one_step_sequence(
        model, test_dataset, config.DEVICE, start_idx=visual_start, n_frames=n_rollout, num_samples=config.EVAL_ONE_STEP_SAMPLES
    )
    
    plot_prediction_comparison(
        one_step_real_img, one_step_pred_img, num_frames=min(config.EVAL_VISUAL_FRAMES, len(one_step_real_img)),
        title="1-step prediction: Ground Truth vs Prediction", save_path=str(config.PLOTS_DIR / "test_one_step_img.png")
    )
    plot_sensor_rollout(one_step_real_tac, one_step_pred_tac, save_path=str(config.PLOTS_DIR / "test_one_step_tac.png"), title="1-step prediction: Tactile")
    plot_sensor_rollout(one_step_real_pos, one_step_pred_pos, save_path=str(config.PLOTS_DIR / "test_one_step_pos.png"), title="1-step prediction: Proprioception")
    export_comparison_video(one_step_real_img, one_step_pred_img, filename=str(config.VIDEOS_DIR / "test_one_step.mp4"), fps=config.VIDEO_FPS)

    _, _, _, _, _, _, rollout_real_img, rollout_pred_img, rollout_real_tac, rollout_pred_tac, rollout_real_pos, rollout_pred_pos = compute_mse_rollout(
        model, test_dataset, config.DEVICE, n_steps=n_rollout, start_idx=visual_start, num_samples=config.EVAL_ROLLOUT_SAMPLES
    )
    
    plot_video_rollout(rollout_real_img, rollout_pred_img, num_frames=min(config.EVAL_VISUAL_FRAMES, len(rollout_real_img)), save_path=str(config.PLOTS_DIR / "test_rollout_img.png"))
    plot_sensor_rollout(rollout_real_tac, rollout_pred_tac, save_path=str(config.PLOTS_DIR / "test_rollout_tac.png"), title="Rollout prediction: Tactile")
    plot_sensor_rollout(rollout_real_pos, rollout_pred_pos, save_path=str(config.PLOTS_DIR / "test_rollout_pos.png"), title="Rollout prediction: Proprioception")
    export_comparison_video(rollout_real_img, rollout_pred_img, filename=str(config.VIDEOS_DIR / "test_rollout.mp4"), fps=config.VIDEO_FPS)
    export_triple_comparison_video(rollout_real_img, one_step_pred_img, rollout_pred_img, filename=str(config.VIDEOS_DIR / "test_triple_comparison.mp4"), fps=config.VIDEO_FPS)

    for ds_info in per_dataset_splits[:1]:
        trial = ds_info["trial"]
        condition = ds_info["condition"]
        ds_test = WorldModelDataset(*ds_info["test"])
        ds_n_rollout = min(config.ROLLOUT_STEPS, len(ds_test) - 1)
        
        if ds_n_rollout > 0:
            ds_visual_start = max(0, min(config.EVAL_VISUAL_START, len(ds_test) - ds_n_rollout - 1))
            ds_one_step_real_img, ds_one_step_pred_img, _, _, _, _ = compute_one_step_sequence(
                model, ds_test, config.DEVICE, start_idx=ds_visual_start, n_frames=ds_n_rollout, num_samples=config.EVAL_ONE_STEP_SAMPLES
            )
            _, _, _, _, _, _, ds_rollout_real_img, ds_rollout_pred_img, ds_rollout_real_tac, ds_rollout_pred_tac, _, _ = compute_mse_rollout(
                model, ds_test, config.DEVICE, n_steps=ds_n_rollout, start_idx=ds_visual_start, num_samples=config.EVAL_ROLLOUT_SAMPLES
            )
            
            prefix = f"test_triple_t{trial}_{condition}"
            export_triple_comparison_video(
                ds_rollout_real_img, ds_one_step_pred_img, ds_rollout_pred_img, filename=str(config.VIDEOS_DIR / f"{prefix}.mp4"), fps=config.VIDEO_FPS
            )
            plot_sensor_rollout(ds_rollout_real_tac, ds_rollout_pred_tac, save_path=str(config.PLOTS_DIR / f"{prefix}_tac.png"), title=f"Rollout prediction: Tactile (Trial {trial})")


if __name__ == "__main__":
    evaluate()