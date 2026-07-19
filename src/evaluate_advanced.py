import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import config
from models.world_model import WorldModel
from utils.checkpoint import load_checkpoint
from utils.data_utils import load_and_concat_datasets, WorldModelDataset
from utils.plot import (
    plot_rollout_quality_summary,
    plot_results,
    plot_divergence_test_results
)

try:
    from torchmetrics.functional.image import structural_similarity_index_measure as torch_ssim
except Exception:
    torch_ssim = None

try:
    from torchmetrics.functional.image import learned_perceptual_image_patch_similarity as torch_lpips
except Exception:
    torch_lpips = None

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)

def resolve_model_path(model_name: str) -> Path:
    path = Path(model_name)
    if path.suffix == "":
        path = path.with_suffix(".pt")
    if not path.is_absolute():
        path = config.MODELS_DIR / path
    return path

def build_action_sequences(real_actions: torch.Tensor, n_steps: int, action_dim: int) -> dict:
    """Builds two opposing constant action sequences based on data std.

    The sequences are constant over time and are exact opposites across
    all action dimensions (push_pos == - push_neg).
    """
    action_std = real_actions.std(dim=0)
    amplitude = 2.0 * action_std  

    seq_pos = amplitude.unsqueeze(0).repeat(n_steps, 1).clone()
    seq_neg = (-amplitude).unsqueeze(0).repeat(n_steps, 1).clone()

    return {"push_pos": seq_pos, "push_neg": seq_neg}


def build_half_rollout_action_sequences(real_actions: torch.Tensor, n_steps: int, action_dim: int, switch_step: int) -> dict:
    """Builds two sequences that are identical until switch_step, then opposite.

    push_pos stays constant to +amplitude for all rollout steps.
    push_neg matches push_pos up to switch_step-1 and then flips sign.
    """
    action_std = real_actions.std(dim=0)
    amplitude = 2.0 * action_std

    seq_pos = amplitude.unsqueeze(0).repeat(n_steps, 1).clone()
    seq_neg = seq_pos.clone()

    split = max(0, min(int(switch_step), n_steps))
    if split < n_steps:
        seq_neg[split:] = -amplitude

    return {"push_pos": seq_pos, "push_neg": seq_neg}


def build_gradual_split_action_sequences(real_actions: torch.Tensor, n_steps: int, action_dim: int, split_step: int) -> dict:
    """Builds two sequences identical up to split_step, then smoothly diverging.

    From split_step onward, both sequences move linearly away from the shared
    action at step split_step - 1 toward opposite targets.
    """
    action_std = real_actions.std(dim=0)
    base_action = 2.0 * action_std
    pos_target = 1.5 * base_action
    neg_target = -base_action

    seq_pos = base_action.unsqueeze(0).repeat(n_steps, 1).clone()
    seq_neg = base_action.unsqueeze(0).repeat(n_steps, 1).clone()

    split = max(0, min(int(split_step), n_steps - 1))
    ramp_len = n_steps - split
    if ramp_len > 0:
        ramp = torch.linspace(0.0, 1.0, ramp_len, device=seq_pos.device).unsqueeze(1)
        start_action = seq_pos[split - 1] if split > 0 else base_action
        seq_pos[split:] = start_action * (1.0 - ramp) + pos_target * ramp
        seq_neg[split:] = start_action * (1.0 - ramp) + neg_target * ramp

    return {"push_pos": seq_pos, "push_neg": seq_neg}


def build_gradual_from_zero_action_sequences(real_actions: torch.Tensor, n_steps: int, action_dim: int) -> dict:
    """Builds two sequences that diverge gradually from t=0.

    push_pos ramps from 0 to +amplitude, while push_neg ramps from 0 to -amplitude.
    This makes the action difference and the resulting latent divergence grow smoothly
    from the beginning of the rollout.
    """
    action_std = real_actions.std(dim=0)
    amplitude = 2.0 * action_std

    ramp = torch.linspace(0.0, 1.0, n_steps, device=real_actions.device).unsqueeze(1)
    seq_pos = amplitude.unsqueeze(0) * ramp
    seq_neg = (-amplitude).unsqueeze(0) * ramp

    return {"push_pos": seq_pos.clone(), "push_neg": seq_neg.clone()}


def _tagged_name(stem: str, tag: str) -> str:
    return f"{stem}_{tag}.png" if tag else f"{stem}.png"


def _spaced_start_indices(max_start: int, num_starts: int) -> list[int]:
    if max_start <= 0:
        return [0]
    num_starts = max(1, min(num_starts, max_start + 1))
    starts = np.linspace(0, max_start, num=num_starts, dtype=int)
    return np.unique(starts).tolist()


def _safe_ssim(pred_img: torch.Tensor, real_img: torch.Tensor) -> float:
    if torch_ssim is None:
        return float("nan")
    try:
        return float(torch_ssim(pred_img.unsqueeze(0), real_img.unsqueeze(0), data_range=1.0))
    except Exception:
        return float("nan")


def _safe_lpips(pred_img: torch.Tensor, real_img: torch.Tensor) -> float:
    if torch_lpips is None:
        return float("nan")
    try:
        pred_lpips = pred_img.unsqueeze(0) * 2.0 - 1.0
        real_lpips = real_img.unsqueeze(0) * 2.0 - 1.0
        return float(torch_lpips(pred_lpips, real_lpips, normalize=False))
    except Exception:
        return float("nan")


def _mean_feature_correlation(pred_seq: torch.Tensor, real_seq: torch.Tensor) -> float:
    pred_np = pred_seq.detach().cpu().numpy()
    real_np = real_seq.detach().cpu().numpy()
    corrs = []
    for idx in range(real_np.shape[-1]):
        pred_vals = pred_np[:, idx]
        real_vals = real_np[:, idx]
        if np.std(pred_vals) < 1e-8 or np.std(real_vals) < 1e-8:
            continue
        corr = np.corrcoef(pred_vals, real_vals)[0, 1]
        if np.isfinite(corr):
            corrs.append(float(corr))
    return float(np.mean(corrs)) if corrs else float("nan")


def _cumulative_derivative_error_curve(pred_seq: torch.Tensor, real_seq: torch.Tensor) -> np.ndarray:
    n_steps = pred_seq.shape[0]
    curve = np.zeros(n_steps, dtype=np.float32)
    if n_steps < 2:
        return curve
    for t in range(1, n_steps):
        pred_delta = pred_seq[1:t + 1] - pred_seq[:t]
        real_delta = real_seq[1:t + 1] - real_seq[:t]
        curve[t] = float(F.mse_loss(pred_delta, real_delta).item())
    return curve

def run_rollouts(model, img_0, tac_0, pos_0, action_sequences: dict, n_samples: int, device):
    """Runs rollouts. Samples with the same k share the same seed (noise)."""
    model.eval()
    results = {}

    with torch.no_grad():
        for name, seq in action_sequences.items():
            actions = seq.unsqueeze(0).to(device)
            imgs_samples, tacs_samples, pos_samples = [], [], []
            
            for k in range(n_samples):
                torch.manual_seed(config.SEED + k) 
                imgs, tacs, pos = model.rollout(img_0, tac_0, pos_0, actions)
                imgs_samples.append(imgs.squeeze(0).cpu())
                tacs_samples.append(tacs.squeeze(0).cpu())
                pos_samples.append(pos.squeeze(0).cpu())

            results[name] = {
                "imgs": torch.stack(imgs_samples, dim=0),
                "tacs": torch.stack(tacs_samples, dim=0),
                "pos":  torch.stack(pos_samples, dim=0),
            }
    return results


def encode_sequence_to_latent(model, imgs: torch.Tensor, tacs: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Encodes a rollout sequence into concatenated mean latents.

    Args:
        model: WorldModel used for the encoders.
        imgs: Tensor of shape [T, C, H, W].
        tacs: Tensor of shape [T, tac_dim].
        pos: Tensor of shape [T, proprio_dim].

    Returns:
        Tensor of shape [T, latent_dim_total] containing concatenated mean latents.
    """
    model.eval()
    with torch.no_grad():
        if imgs.dim() == 3:
            imgs = imgs.unsqueeze(0)
        if tacs.dim() == 1:
            tacs = tacs.unsqueeze(0)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)

        if config.LATENT_TYPE == "vqvae":
            _, mu_img, _, _ = model.visual_encoder(imgs.to(config.DEVICE))
        else:
            _, mu_img, _ = model.visual_encoder(imgs.to(config.DEVICE))
        _, mu_tac, _ = model.tactile_encoder(tacs.to(config.DEVICE))
        _, mu_pos, _ = model.proprio_encoder(pos.to(config.DEVICE))

        latent = torch.cat([mu_img, mu_tac, mu_pos], dim=-1)
    return latent.cpu()


def temporal_latent_variation(model, imgs: torch.Tensor, tacs: torch.Tensor, pos: torch.Tensor) -> np.ndarray:
    """Computes normalized temporal latent variation for a rollout sequence.

    Returns the per-step quantity ||z_{t+1} - z_t|| / sqrt(D) for t in [0, T-2].
    """
    latent_seq = encode_sequence_to_latent(model, imgs, tacs, pos)
    diffs = latent_seq[1:] - latent_seq[:-1]
    latent_dim = latent_seq.shape[-1]
    return (diffs.norm(dim=-1) / np.sqrt(latent_dim)).numpy()


def latent_divergence_between_actions(model, imgs_pos: torch.Tensor, tacs_pos: torch.Tensor, pos_pos: torch.Tensor,
                                      imgs_neg: torch.Tensor, tacs_neg: torch.Tensor, pos_neg: torch.Tensor) -> np.ndarray:
    """Computes normalized latent divergence between push_pos and push_neg at the same step.

    Returns the per-step quantity ||z_t^{pos} - z_t^{neg}|| / sqrt(D).
    """
    latent_pos = encode_sequence_to_latent(model, imgs_pos, tacs_pos, pos_pos)
    latent_neg = encode_sequence_to_latent(model, imgs_neg, tacs_neg, pos_neg)

    steps = min(latent_pos.shape[0], latent_neg.shape[0])
    latent_pos = latent_pos[:steps]
    latent_neg = latent_neg[:steps]

    latent_dim = latent_pos.shape[-1]
    return ((latent_pos - latent_neg).norm(dim=-1) / np.sqrt(latent_dim)).numpy()


def compute_rollout_quality_metrics(model, test_dataset, device, n_steps: int,
                                    start_idx: int = 0, num_samples: int = 1):
    """Computes rollout metrics that penalize flat predictions.

    Returns per-step curves for image MSE / SSIM / LPIPS and for tactile/proprio
    MSE plus cumulative trajectory-shape error, together with scalar correlations.
    """
    model.eval()
    max_start = len(test_dataset) - n_steps - 1
    if max_start < 0:
        n_steps = len(test_dataset) - 1
        max_start = 0

    start_idx = max(0, min(int(start_idx), max_start))

    img_0 = test_dataset.X_t[start_idx].unsqueeze(0).to(device)
    tac_0 = test_dataset.S_t[start_idx].unsqueeze(0).to(device)
    pos_0 = test_dataset.P_t[start_idx].unsqueeze(0).to(device)
    actions = test_dataset.A_t[start_idx:start_idx + n_steps].unsqueeze(0).to(device)

    real_imgs = test_dataset.X_t1[start_idx:start_idx + n_steps]
    real_tacs = test_dataset.S_t1[start_idx:start_idx + n_steps]
    real_pos = test_dataset.P_t1[start_idx:start_idx + n_steps]

    img_mse_samples = []
    img_ssim_samples = []
    img_lpips_samples = []
    tac_mse_samples = []
    tac_shape_samples = []
    pos_mse_samples = []
    pos_shape_samples = []

    pred_img_out = None
    pred_tac_out = None
    pred_pos_out = None

    with torch.no_grad():
        for k in range(max(1, num_samples)):
            torch.manual_seed(config.SEED + k)
            pred_imgs, pred_tacs, pred_pos = model.rollout(img_0, tac_0, pos_0, actions)
            pred_imgs = pred_imgs.squeeze(0).cpu()
            pred_tacs = pred_tacs.squeeze(0).cpu()
            pred_pos = pred_pos.squeeze(0).cpu()

            img_mse_curve = []
            img_ssim_curve = []
            img_lpips_curve = []
            tac_mse_curve = []
            pos_mse_curve = []

            for t in range(n_steps):
                real_img_t = real_imgs[t]
                pred_img_t = pred_imgs[t]
                img_mse_curve.append(float(F.mse_loss(pred_img_t, real_img_t).item()))
                ssim_val = _safe_ssim(pred_img_t, real_img_t)
                img_ssim_curve.append(1.0 - ssim_val if np.isfinite(ssim_val) else float("nan"))
                img_lpips_curve.append(_safe_lpips(pred_img_t, real_img_t))

                tac_mse_curve.append(float(F.mse_loss(pred_tacs[t], real_tacs[t]).item()))
                pos_mse_curve.append(float(F.mse_loss(pred_pos[t], real_pos[t]).item()))

            tac_shape_curve = _cumulative_derivative_error_curve(pred_tacs, real_tacs)
            pos_shape_curve = _cumulative_derivative_error_curve(pred_pos, real_pos)

            img_mse_samples.append(img_mse_curve)
            img_ssim_samples.append(img_ssim_curve)
            img_lpips_samples.append(img_lpips_curve)
            tac_mse_samples.append(tac_mse_curve)
            tac_shape_samples.append(tac_shape_curve.tolist())
            pos_mse_samples.append(pos_mse_curve)
            pos_shape_samples.append(pos_shape_curve.tolist())

            if pred_img_out is None:
                pred_img_out = pred_imgs
                pred_tac_out = pred_tacs
                pred_pos_out = pred_pos

    metrics = {
        "real_imgs": real_imgs,
        "pred_imgs": pred_img_out,
        "real_tacs": real_tacs,
        "pred_tacs": pred_tac_out,
        "real_pos": real_pos,
        "pred_pos": pred_pos_out,
        "img_mse_mean": np.nanmean(np.asarray(img_mse_samples), axis=0),
        "img_ssim_dist_mean": np.nanmean(np.asarray(img_ssim_samples), axis=0),
        "img_lpips_mean": np.nanmean(np.asarray(img_lpips_samples), axis=0),
        "tac_mse_mean": np.nanmean(np.asarray(tac_mse_samples), axis=0),
        "tac_shape_mean": np.nanmean(np.asarray(tac_shape_samples), axis=0),
        "pos_mse_mean": np.nanmean(np.asarray(pos_mse_samples), axis=0),
        "pos_shape_mean": np.nanmean(np.asarray(pos_shape_samples), axis=0),
        "tac_corr": _mean_feature_correlation(pred_tac_out, real_tacs),
        "pos_corr": _mean_feature_correlation(pred_pos_out, real_pos),
    }
    return metrics


def evaluate_rollout_quality_test():
    """Runs a real rollout test and pairs MSE with anti-flat metrics.

    The rollout is computed as in evaluate.py, but the plots also show SSIM/LPIPS
    for images and trajectory-shape error for tactile/proprioception.
    """
    print("--- Rollout quality test: MSE + SSIM/LPIPS + trajectory shape ---")

    _, _, _, _, per_dataset_splits = load_and_concat_datasets(config.DATASET_DIR, config.DATASETS, config.SPLIT_RATIO)
    if len(per_dataset_splits) == 0:
        print("No per-dataset splits found.")
        return

    ds_info = per_dataset_splits[0]
    ds_test = WorldModelDataset(*ds_info["test"])

    img_channels = ds_test.X_t.shape[1] if config.IMG_CHANNELS is None else config.IMG_CHANNELS
    tac_features = ds_test.S_t.shape[1] if config.TAC_FEATURES is None else config.TAC_FEATURES
    proprio_features = ds_test.P_t.shape[1] if config.PROPRIO_FEATURES is None else config.PROPRIO_FEATURES
    action_dim = ds_test.A_t.shape[1]

    model = WorldModel(
        img_channels=img_channels, img_latent_dim=config.IMG_LATENT_DIM,
        tac_features=tac_features, tac_latent_dim=config.TAC_LATENT_DIM,
        proprio_features=proprio_features, proprio_latent_dim=config.PROPRIO_LATENT_DIM,
        action_dim=action_dim, hidden_dim=config.HIDDEN_DIM,
        dynamics_hidden_dim=config.DYNAMICS_HIDDEN_DIM, dynamics_type=config.DYNAMICS_TYPE,
        latent_type=config.LATENT_TYPE, vq_num_embeddings=config.VQ_NUM_EMBEDDINGS,
        vq_embedding_dim=config.VQ_EMBEDDING_DIM, vq_commitment_cost=config.VQ_COMMITMENT_COST
    ).to(config.DEVICE)

    load_checkpoint(resolve_model_path(config.DYNAMICS_BEST_MODEL_NAME), model, None, device=str(config.DEVICE))

    n_steps = min(config.ROLLOUT_STEPS, len(ds_test) - 1)
    max_start = len(ds_test) - n_steps - 1
    rollout_starts = _spaced_start_indices(max_start, config.EVAL_ROLLOUT_STARTS)

    per_start_metrics = []
    for start_idx in rollout_starts:
        per_start_metrics.append(
            compute_rollout_quality_metrics(
                model, ds_test, config.DEVICE, n_steps=n_steps, start_idx=start_idx, num_samples=config.EVAL_ROLLOUT_SAMPLES
            )
        )

    def _avg_curve(key: str):
        return np.nanmean(np.stack([m[key] for m in per_start_metrics], axis=0), axis=0)

    img_mse = _avg_curve("img_mse_mean")
    img_ssim_dist = _avg_curve("img_ssim_dist_mean")
    img_lpips = _avg_curve("img_lpips_mean")
    tac_mse = _avg_curve("tac_mse_mean")
    tac_shape = _avg_curve("tac_shape_mean")
    pos_mse = _avg_curve("pos_mse_mean")
    pos_shape = _avg_curve("pos_shape_mean")

    tac_corr = float(np.nanmean([m["tac_corr"] for m in per_start_metrics]))
    pos_corr = float(np.nanmean([m["pos_corr"] for m in per_start_metrics]))

    steps = np.arange(n_steps)
    out_path = config.PLOTS_DIR_EXP / "rollout_quality_summary.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_rollout_quality_summary(steps, img_mse, img_ssim_dist, img_lpips,
                                 tac_mse, tac_shape, pos_mse, pos_shape,
                                 tac_corr, pos_corr, out_path)

    image_ssim_mean = float(1.0 - np.nanmean(img_ssim_dist))
    image_lpips_mean = float(np.nanmean(img_lpips)) if np.isfinite(img_lpips).any() else float("nan")

    print("--- Rollout quality summary ---")
    print(f"Image MSE mean          : {float(np.nanmean(img_mse)):.6f}")
    print(f"Image SSIM mean         : {image_ssim_mean:.6f}")
    print(f"Image LPIPS mean        : {image_lpips_mean:.6f}" if np.isfinite(image_lpips_mean) else "Image LPIPS mean        : n/a")
    print(f"Tactile MSE mean        : {float(np.nanmean(tac_mse)):.6f}")
    print(f"Tactile shape error mean: {float(np.nanmean(tac_shape)):.6f}")
    print(f"Tactile traj corr       : {tac_corr:.6f}")
    print(f"Proprio MSE mean        : {float(np.nanmean(pos_mse)):.6f}")
    print(f"Proprio shape error mean: {float(np.nanmean(pos_shape)):.6f}")
    print(f"Proprio traj corr       : {pos_corr:.6f}")
    print(f"Saved rollout quality plot to: {out_path}")

def evaluate_action_sensitivity():
    print("--- World Model Action Sensitivity Test ---")
    
    splits, _, _, _, _ = load_and_concat_datasets(config.DATASET_DIR, config.DATASETS, config.SPLIT_RATIO)
    test_dataset = WorldModelDataset(*splits["test"])
    
    img_channels = splits["test"][0].shape[1] if config.IMG_CHANNELS is None else config.IMG_CHANNELS
    tac_features = splits["test"][2].shape[1] if config.TAC_FEATURES is None else config.TAC_FEATURES
    proprio_features = splits["test"][4].shape[1] if config.PROPRIO_FEATURES is None else config.PROPRIO_FEATURES
    action_dim = splits["test"][6].shape[1]

    model = WorldModel(
        img_channels=img_channels, img_latent_dim=config.IMG_LATENT_DIM,
        tac_features=tac_features, tac_latent_dim=config.TAC_LATENT_DIM,
        proprio_features=proprio_features, proprio_latent_dim=config.PROPRIO_LATENT_DIM,
        action_dim=action_dim, hidden_dim=config.HIDDEN_DIM,
        dynamics_hidden_dim=config.DYNAMICS_HIDDEN_DIM, dynamics_type=config.DYNAMICS_TYPE,
        latent_type=config.LATENT_TYPE, vq_num_embeddings=config.VQ_NUM_EMBEDDINGS,
        vq_embedding_dim=config.VQ_EMBEDDING_DIM, vq_commitment_cost=config.VQ_COMMITMENT_COST
    ).to(config.DEVICE)

    load_checkpoint(resolve_model_path(config.DYNAMICS_BEST_MODEL_NAME), model, None, device=str(config.DEVICE))

    start_idx = min(config.EVAL_VISUAL_START, len(test_dataset) - 30)
    n_steps = 25
    
    img_0 = test_dataset.X_t[start_idx].unsqueeze(0).to(config.DEVICE)
    tac_0 = test_dataset.S_t[start_idx].unsqueeze(0).to(config.DEVICE)
    pos_0 = test_dataset.P_t[start_idx].unsqueeze(0).to(config.DEVICE)
    
    action_sequences = build_action_sequences(test_dataset.A_t, n_steps, action_dim)
    results = run_rollouts(model, img_0, tac_0, pos_0, action_sequences, config.EVAL_ROLLOUT_SAMPLES, config.DEVICE)

    plot_results(results, str(config.PLOTS_DIR_EXP / "action_sensitivity_summary.png"))

def calculate_mse_between_rollouts(results, key: str, sensor_idx: int):
    """Computes MSE over time between the mean trajectories of two rollouts for a given sensor index."""
    # Expect keys 'push_pos' and 'push_neg' (or first two entries)
    if "push_pos" in results and "push_neg" in results:
        pos = results["push_pos"][key]
        neg = results["push_neg"][key]
    else:
        keys = list(results.keys())
        pos = results[keys[0]][key]
        neg = results[keys[1]][key]

    mean_pos = pos.mean(dim=0)  # [n_steps, dim]
    mean_neg = neg.mean(dim=0)

    n_steps = mean_pos.shape[0]
    idx = max(0, min(sensor_idx, mean_pos.shape[1] - 1))

    mse_per_step = ((mean_pos[:, idx] - mean_neg[:, idx]) ** 2).cpu().numpy()
    return mse_per_step, mean_pos[:, idx].cpu().numpy(), mean_neg[:, idx].cpu().numpy()


def evaluate_divergence_test(action_mode: str = "full_opposite", switch_step: int = None, output_tag: str = ""):
    print("--- Divergence test: Trial 1 no_obj, start index 0 ---")

    _, _, _, _, per_dataset_splits = load_and_concat_datasets(config.DATASET_DIR, config.DATASETS, config.SPLIT_RATIO)

    if len(per_dataset_splits) == 0:
        print("No per-dataset splits found.")
        return

    ds_info = per_dataset_splits[0]

    ds_test = WorldModelDataset(*ds_info["test"])


    img_channels = ds_test.X_t.shape[1] if config.IMG_CHANNELS is None else config.IMG_CHANNELS
    tac_features = ds_test.S_t.shape[1] if config.TAC_FEATURES is None else config.TAC_FEATURES
    proprio_features = ds_test.P_t.shape[1] if config.PROPRIO_FEATURES is None else config.PROPRIO_FEATURES
    action_dim = ds_test.A_t.shape[1]

    model = WorldModel(
        img_channels=img_channels, img_latent_dim=config.IMG_LATENT_DIM,
        tac_features=tac_features, tac_latent_dim=config.TAC_LATENT_DIM,
        proprio_features=proprio_features, proprio_latent_dim=config.PROPRIO_LATENT_DIM,
        action_dim=action_dim, hidden_dim=config.HIDDEN_DIM,
        dynamics_hidden_dim=config.DYNAMICS_HIDDEN_DIM, dynamics_type=config.DYNAMICS_TYPE,
        latent_type=config.LATENT_TYPE, vq_num_embeddings=config.VQ_NUM_EMBEDDINGS,
        vq_embedding_dim=config.VQ_EMBEDDING_DIM, vq_commitment_cost=config.VQ_COMMITMENT_COST
    ).to(config.DEVICE)

    load_checkpoint(resolve_model_path(config.DYNAMICS_BEST_MODEL_NAME), model, None, device=str(config.DEVICE))

    start_idx = 0
    n_steps = config.ROLLOUT_STEPS

    img_0 = ds_test.X_t[start_idx].unsqueeze(0).to(config.DEVICE)
    tac_0 = ds_test.S_t[start_idx].unsqueeze(0).to(config.DEVICE)
    pos_0 = ds_test.P_t[start_idx].unsqueeze(0).to(config.DEVICE)

    if action_mode == "half_opposite":
        half_step = n_steps // 2 if switch_step is None else switch_step
        action_sequences = build_half_rollout_action_sequences(ds_test.A_t, n_steps, action_dim, half_step)
        print(f"Using half-rollout perturbation: same actions until step {half_step}, opposite afterwards.")
        switch_step = half_step
    elif action_mode == "gradual_split":
        gradual_step = 15 if switch_step is None else switch_step
        action_sequences = build_gradual_split_action_sequences(ds_test.A_t, n_steps, action_dim, gradual_step)
        print(f"Using gradual split perturbation: same actions until step {gradual_step}, then linear divergence.")
        switch_step = gradual_step
    elif action_mode == "gradual_from_zero":
        action_sequences = build_gradual_from_zero_action_sequences(ds_test.A_t, n_steps, action_dim)
        print("Using gradual-from-zero perturbation: push_pos and push_neg ramp apart from t=0.")
        switch_step = 0
    else:
        action_sequences = build_action_sequences(ds_test.A_t, n_steps, action_dim)

    results = run_rollouts(model, img_0, tac_0, pos_0, action_sequences, config.EVAL_ROLLOUT_SAMPLES, config.DEVICE)

    tac_idx = 18
    pos_idx = 18

    imgs_pos = results["push_pos"]["imgs"]  # [n_samples, n_steps, C, H, W]
    imgs_neg = results["push_neg"]["imgs"]
    tacs_pos = results["push_pos"]["tacs"]  # [n_samples, n_steps, tac_dim]
    tacs_neg = results["push_neg"]["tacs"]
    pos_pos = results["push_pos"]["pos"]
    pos_neg = results["push_neg"]["pos"]

    mean_imgs_pos = imgs_pos.mean(dim=0)
    mean_imgs_neg = imgs_neg.mean(dim=0)

    img_mse = np.array([((mean_imgs_pos[t] - mean_imgs_neg[t]) ** 2).mean().item() for t in range(mean_imgs_pos.shape[0])])

    mean_tac_pos = tacs_pos.mean(dim=0)  # [n_steps, tac_dim]
    mean_tac_neg = tacs_neg.mean(dim=0)

    mean_pos_pos = pos_pos.mean(dim=0)
    mean_pos_neg = pos_neg.mean(dim=0)

    latent_var_pos = temporal_latent_variation(
        model,
        mean_imgs_pos,
        results["push_pos"]["tacs"].mean(dim=0),
        results["push_pos"]["pos"].mean(dim=0),
    )
    latent_var_neg = temporal_latent_variation(
        model,
        mean_imgs_neg,
        results["push_neg"]["tacs"].mean(dim=0),
        results["push_neg"]["pos"].mean(dim=0),
    )

    latent_div_pos_neg = latent_divergence_between_actions(
        model,
        mean_imgs_pos,
        results["push_pos"]["tacs"].mean(dim=0),
        results["push_pos"]["pos"].mean(dim=0),
        mean_imgs_neg,
        results["push_neg"]["tacs"].mean(dim=0),
        results["push_neg"]["pos"].mean(dim=0),
    )

    steps = np.arange(mean_imgs_pos.shape[0])

    plot_divergence_test_results(
        steps, img_mse, mean_imgs_pos, mean_imgs_neg,
        latent_var_pos, latent_var_neg, latent_div_pos_neg,
        mean_tac_pos, mean_tac_neg, mean_pos_pos, mean_pos_neg,
        tac_idx, pos_idx, switch_step, output_tag
    )


def evaluate_half_rollout_divergence_test():
    n_steps = config.ROLLOUT_STEPS
    switch_step = n_steps // 2
    evaluate_divergence_test(
        action_mode="half_opposite",
        switch_step=switch_step,
        output_tag=f"half_switch{switch_step}",
    )


def evaluate_gradual_split_divergence_test():
    """Runs the gradual split test: same actions until step 15, then smooth divergence."""
    print("--- Gradual split test: same actions until step 15, then smooth divergence ---")
    evaluate_divergence_test(
        action_mode="gradual_split",
        switch_step=15,
        output_tag="gradual_split15",
    )


def evaluate_gradual_from_zero_divergence_test():
    """Runs a gradual divergence test starting from t=0.

    The two action sequences separate smoothly from the first rollout step, so the
    latent divergence between push_pos and push_neg grows gradually from the start.
    """
    print("--- Gradual-from-zero test: actions diverge smoothly from t=0 ---")
    evaluate_divergence_test(
        action_mode="gradual_from_zero",
        switch_step=0,
        output_tag="gradual_from_zero",
    )


if __name__ == "__main__":
    evaluate_divergence_test()
    evaluate_half_rollout_divergence_test()
    evaluate_gradual_split_divergence_test()
    evaluate_gradual_from_zero_divergence_test()
    evaluate_rollout_quality_test()
