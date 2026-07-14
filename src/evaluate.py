
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
)

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)


def resolve_model_path(model_name: str) -> Path:
    path = Path(model_name)
    if path.suffix == "":
        path = path.with_suffix(".pt")
    if not path.is_absolute():
        path = config.MODELS_DIR / path
    return path


def _spaced_start_indices(max_start: int, num_starts: int) -> list[int]:
    if max_start <= 0:
        return [0]
    num_starts = max(1, min(num_starts, max_start + 1))
    starts = np.linspace(0, max_start, num=num_starts, dtype=int)
    return np.unique(starts).tolist()

def compute_mse_1step(model, loader, device, num_samples: int = 1):
    model.eval()
    sample_mses = []

    with torch.no_grad():
        for img_t, img_t1, act_t in loader:
            img_t  = img_t.to(device)
            img_t1 = img_t1.to(device)
            act_t  = act_t.to(device)

            batch_samples = []
            for _ in range(max(1, num_samples)):
                img_pred, _, _, _, _ = model(img_t, act_t)
                batch_samples.append(F.mse_loss(img_pred, img_t1, reduction="mean").item())

            sample_mses.extend(batch_samples)

    mse_norm = float(np.mean(sample_mses)) if sample_mses else float("nan")
    mse_std  = float(np.std(sample_mses)) if sample_mses else float("nan")
    mse_255  = mse_norm * (255.0 ** 2)
    mse_255_std = mse_std * (255.0 ** 2)
    return mse_norm, mse_std, mse_255, mse_255_std


def compute_mse_rollout(model, test_dataset, device, n_steps: int,
                        start_idx: int = 0, num_samples: int = 1):
    model.eval()
    max_start = len(test_dataset) - n_steps

    if max_start < 0:
        print(
            f"  [Warn] Test set too small for {n_steps} steps. "
            f"Using {len(test_dataset) - 1} steps."
        )
        n_steps = len(test_dataset) - 1
        max_start = 0

    img_0   = test_dataset.X_t[start_idx].unsqueeze(0).to(device)
    actions = test_dataset.A_t[start_idx:start_idx + n_steps].unsqueeze(0).to(device)
    real_frames = test_dataset.X_t1[start_idx:start_idx + n_steps]

    sample_curves = []
    pred_frames_out = None

    with torch.no_grad():
        for _ in range(max(1, num_samples)):
            pred_sequence = model.rollout(img_0, actions)
            pred_frames = pred_sequence.squeeze(0).cpu()

            mse_per_step = []
            for t in range(n_steps):
                mse_t = F.mse_loss(pred_frames[t], real_frames[t]).item()
                mse_per_step.append(mse_t)
            sample_curves.append(mse_per_step)

            if pred_frames_out is None:
                pred_frames_out = pred_frames

    mse_mean = np.mean(sample_curves, axis=0).tolist()
    mse_std = np.std(sample_curves, axis=0).tolist()
    return mse_mean, mse_std, real_frames, pred_frames_out


def compute_one_step_sequence(model, test_dataset, device, start_idx: int,
                              n_frames: int, num_samples: int = 1):
    model.eval()
    real_frames = []
    pred_frames = []

    with torch.no_grad():
        for offset in range(n_frames):
            idx = start_idx + offset
            img_t = test_dataset.X_t[idx].unsqueeze(0).to(device)
            act_t = test_dataset.A_t[idx].unsqueeze(0).to(device)
            real_frames.append(test_dataset.X_t1[idx])

            pred_samples = []
            for _ in range(max(1, num_samples)):
                img_pred, _, _, _, _ = model(img_t, act_t)
                pred_samples.append(img_pred.squeeze(0).cpu())

            pred_frames.append(torch.stack(pred_samples, dim=0).mean(dim=0))

    return torch.stack(real_frames, dim=0), torch.stack(pred_frames, dim=0)



def evaluate(dynamics_checkpoint_name: str = config.DYNAMICS_BEST_MODEL_NAME):
    print("=" * 65)
    print("  World Model — Evaluation")
    print(f"  Device  : {config.DEVICE}")
    print(f"  Dataset : {len(config.DATASETS)} trial(s)")
    print("=" * 65)

    for d in [config.MODELS_DIR, config.PLOTS_DIR, config.VIDEOS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Preparing test split...")
    splits, _, per_dataset_splits = load_and_concat_datasets(
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

    action_dim = splits["test"][2].shape[1]
    print(f"  Test samples: {len(test_dataset)}")

    print("\n[2/4] Loading checkpoint...")

    dynamics_ckpt = resolve_model_path(dynamics_checkpoint_name)

    if not dynamics_ckpt.exists():
        print(f"\n  [Error] Dynamics checkpoint not found: {dynamics_ckpt}")
        print("  Run dynamics training first.")
        return

    dynamics_label = config.DYNAMICS_TYPE.upper()
    if config.DYNAMICS_TYPE == "diffusion":
        dynamics_label += f" (T={config.DIFFUSION_STEPS})"
    print(f"  Model      : {dynamics_label}")
    print(f"  Checkpoint : {dynamics_ckpt.name}")

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

    model, _, dynamics_epoch, dynamics_val_loss = load_checkpoint(
        dynamics_ckpt, model, optimizer=None, device=str(config.DEVICE)
    )

    model.eval()
    print(f"  Loaded epoch {dynamics_epoch + 1} | val_loss={dynamics_val_loss:.6f}")


    print("\n[3/4] Metrics")
    print("-" * 65)

    mse_norm, mse_std, mse_255, mse_255_std = compute_mse_1step(
        model, test_loader, config.DEVICE,
        num_samples=config.EVAL_ONE_STEP_SAMPLES
    )
    print(f"  1-step MSE        : {mse_norm:.6f} ± {mse_std:.6f}  [norm]")
    print(f"  1-step MSE        : {mse_255:.2f} ± {mse_255_std:.2f}  [0-255]")

    n_rollout = min(config.ROLLOUT_STEPS, len(test_dataset) - 1)
    max_start = len(test_dataset) - n_rollout
    if max_start < 0:
        print("  [Error] Cannot compute rollout (test set too short)")
        return

    rollout_starts = _spaced_start_indices(max_start, config.EVAL_ROLLOUT_STARTS)
    per_start_curves = []
    for start in rollout_starts:
        mse_mean, mse_std, _, _ = compute_mse_rollout(
            model,
            test_dataset,
            config.DEVICE,
            n_steps=n_rollout,
            start_idx=start,
            num_samples=config.EVAL_ROLLOUT_SAMPLES,
        )
        per_start_curves.append(mse_mean)

    rollout_curve_mean = np.mean(per_start_curves, axis=0)
    rollout_curve_std  = np.std(per_start_curves, axis=0)
    rollout_mean = float(np.mean(rollout_curve_mean))
    rollout_final = float(rollout_curve_mean[-1])
    rollout_auc = float(np.trapezoid(rollout_curve_mean, dx=1.0) / max(1, n_rollout - 1))

    print(f"  Rollout mean MSE  : {rollout_mean:.6f}")
    print(f"  Rollout final MSE : {rollout_final:.6f}")
    print(f"  Rollout AUC       : {rollout_auc:.6f}")
    print(f"  Rollout starts    : {len(rollout_starts)}")
    print("-" * 65)


    visual_start = min(config.EVAL_VISUAL_START, len(test_dataset) - n_rollout - 1)
    visual_start = max(0, visual_start)

    one_step_real, one_step_pred = compute_one_step_sequence(
        model,
        test_dataset,
        config.DEVICE,
        start_idx=visual_start,
        n_frames=n_rollout,
        num_samples=config.EVAL_ONE_STEP_SAMPLES,
    )
    one_step_plot = config.PLOTS_DIR / "test_one_step.png"
    plot_prediction_comparison(
        one_step_real,
        one_step_pred,
        num_frames=min(config.EVAL_VISUAL_FRAMES, len(one_step_real)),
        title="1-step prediction: Ground Truth vs Predetto",
        save_path=str(one_step_plot),
    )
    print(f"  Saved 1-step plot  : {one_step_plot.name}")

    one_step_video = config.VIDEOS_DIR / "test_one_step.mp4"
    export_comparison_video(
        one_step_real,
        one_step_pred,
        filename=str(one_step_video),
        fps=config.VIDEO_FPS,
    )
    print(f"  Saved 1-step video : {one_step_video.name}")

    rollout_mse_mean, rollout_mse_std, rollout_real, rollout_pred = compute_mse_rollout(
        model,
        test_dataset,
        config.DEVICE,
        n_steps=n_rollout,
        start_idx=visual_start,
        num_samples=config.EVAL_ROLLOUT_SAMPLES,
    )
    rollout_plot = config.PLOTS_DIR / "test_rollout.png"
    plot_video_rollout(
        rollout_real,
        rollout_pred,
        num_frames=min(config.EVAL_VISUAL_FRAMES, len(rollout_real)),
        save_path=str(rollout_plot),
    )
    print(f"  Saved rollout plot : {rollout_plot.name}")

    rollout_video = config.VIDEOS_DIR / "test_rollout.mp4"
    export_comparison_video(
        rollout_real,
        rollout_pred,
        filename=str(rollout_video),
        fps=config.VIDEO_FPS,
    )
    print(f"  Saved rollout video: {rollout_video.name}")

    triple_video = config.VIDEOS_DIR / "test_triple_comparison.mp4"
    export_triple_comparison_video(
        rollout_real,
        one_step_pred,
        rollout_pred,
        filename=str(triple_video),
        fps=config.VIDEO_FPS,
    )
    print(f"  Saved triple video : {triple_video.name}")

    print("\n[4/4] Generating Videos per Dataset")
    print("-" * 65)

    for ds_info in per_dataset_splits:
        trial = ds_info["trial"]
        condition = ds_info["condition"]
        print(f"\n  Processing dataset: Trial {trial} - {condition}")
        
        ds_test = WorldModelDataset(*ds_info["test"])
        
        ds_n_rollout = min(config.ROLLOUT_STEPS, len(ds_test) - 1)
        if ds_n_rollout <= 0:
            print("  [Warn] Dataset too short for rollout")
            continue
            
        ds_visual_start = min(config.EVAL_VISUAL_START, len(ds_test) - ds_n_rollout - 1)
        ds_visual_start = max(0, ds_visual_start)
        
        ds_one_step_real, ds_one_step_pred = compute_one_step_sequence(
            model,
            ds_test,
            config.DEVICE,
            start_idx=ds_visual_start,
            n_frames=ds_n_rollout,
            num_samples=config.EVAL_ONE_STEP_SAMPLES,
        )
        
        _, _, ds_rollout_real, ds_rollout_pred = compute_mse_rollout(
            model,
            ds_test,
            config.DEVICE,
            n_steps=ds_n_rollout,
            start_idx=ds_visual_start,
            num_samples=config.EVAL_ROLLOUT_SAMPLES,
        )
        
        prefix = f"test_triple_t{trial}_{condition}"
        ds_triple_video = config.VIDEOS_DIR / f"{prefix}.mp4"
        export_triple_comparison_video(
            ds_rollout_real,
            ds_one_step_pred,
            ds_rollout_pred,
            filename=str(ds_triple_video),
            fps=config.VIDEO_FPS,
        )
        print(f"  Saved triple video : {ds_triple_video.name}")

    print("\n  Evaluation complete.")



if __name__ == "__main__":
    evaluate()
