import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def plot_losses_train_val(train_losses: list, val_losses: list,
                          save_path: str = None, start_epoch: int = 1, best_epoch: int = None):
    """
    Plots the training and validation loss curves over epochs and highlights the best epoch.
    
    Args:
        train_losses: List of training loss values.
        val_losses: List of validation loss values.
        save_path: Optional path to save the generated plot image. If None, the plot is displayed.
        start_epoch: The starting epoch number for the x-axis.
    """
    epochs = range(start_epoch, start_epoch + len(train_losses))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, train_losses, label="Train Loss",
            color="#2196F3", linewidth=2)
    ax.plot(epochs, val_losses, label="Val Loss",
            color="#F44336", linewidth=2, linestyle="--")

    if best_epoch is None:
        best_epoch = int(np.argmin(val_losses)) + start_epoch
    
    # Calculate index safely
    idx = max(0, min(best_epoch - start_epoch, len(val_losses) - 1))
    best_val   = val_losses[idx]
    
    ax.axvline(best_epoch, color="#4CAF50", linestyle=":", linewidth=1.5,
               label=f"Best epoch ({best_epoch}) — val={best_val:.5f}")
    ax.scatter(best_epoch, best_val, color="#4CAF50", zorder=5, s=80)

    ax.set_title("Loss Curve — Training vs Validation", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_losses(loss_history: dict, save_path: str = None):
    """
    Plots multiple loss components from a history dictionary over epochs.
    
    Args:
        loss_history: Dictionary where keys are loss names and values are lists of loss values.
        save_path: Optional path to save the generated plot image.
    """
    plt.figure(figsize=(10, 6))

    for loss_name, loss_values in loss_history.items():
        plt.plot(loss_values, label=loss_name, linewidth=2)

    plt.title("World Model Training Losses", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=12)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_video_rollout(real_frames, pred_frames, num_frames: int = 5,
                       save_path: str = None):
    """
    Plots a static side-by-side visual comparison of ground truth and predicted image sequences over time.
    
    Args:
        real_frames: Ground truth sequence of images.
        pred_frames: Predicted sequence of images.
        num_frames: Maximum number of frames to display in the plot.
        save_path: Optional path to save the generated plot image.
    """
    if torch.is_tensor(real_frames):
        real_frames = real_frames.detach().cpu().numpy()
    if torch.is_tensor(pred_frames):
        pred_frames = pred_frames.detach().cpu().numpy()

    T = min(len(real_frames), len(pred_frames), num_frames)

    fig, axes = plt.subplots(2, T, figsize=(3 * T, 6))
    plt.suptitle("Visual Rollout: Ground Truth vs Prediction", fontsize=16)

    for t in range(T):
        img_real = np.transpose(real_frames[t], (1, 2, 0))
        img_real = np.clip(img_real, 0, 1)
        if img_real.shape[-1] == 1:
            axes[0, t].imshow(img_real.squeeze(-1), cmap="gray")
        else:
            axes[0, t].imshow(img_real)
        axes[0, t].set_title(f"Real t={t}", fontsize=9)
        axes[0, t].axis("off")

        img_pred = np.transpose(pred_frames[t], (1, 2, 0))
        img_pred = np.clip(img_pred, 0, 1)
        if img_pred.shape[-1] == 1:
            axes[1, t].imshow(img_pred.squeeze(-1), cmap="gray")
        else:
            axes[1, t].imshow(img_pred)
        axes[1, t].set_title(f"Predicted t={t}", fontsize=9)
        axes[1, t].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_prediction_comparison(real_frames, pred_frames, num_frames: int = 5,
                               title: str = "Prediction Comparison",
                               save_path: str = None):
    """
    Plots a frame-by-frame visual comparison of real and predicted sequences with a customizable title.
    
    Args:
        real_frames: Ground truth sequence of images.
        pred_frames: Predicted sequence of images.
        num_frames: Maximum number of frames to display in the plot.
        title: Title of the generated plot.
        save_path: Optional path to save the generated plot image.
    """
    if torch.is_tensor(real_frames):
        real_frames = real_frames.detach().cpu().numpy()
    if torch.is_tensor(pred_frames):
        pred_frames = pred_frames.detach().cpu().numpy()

    T = min(len(real_frames), len(pred_frames), num_frames)

    fig, axes = plt.subplots(2, T, figsize=(3 * T, 6))
    plt.suptitle(title, fontsize=16)

    for t in range(T):
        img_real = np.transpose(real_frames[t], (1, 2, 0))
        img_real = np.clip(img_real, 0, 1)
        if img_real.shape[-1] == 1:
            axes[0, t].imshow(img_real.squeeze(-1), cmap="gray")
        else:
            axes[0, t].imshow(img_real)
        axes[0, t].set_title(f"Real t={t}", fontsize=9)
        axes[0, t].axis("off")

        img_pred = np.transpose(pred_frames[t], (1, 2, 0))
        img_pred = np.clip(img_pred, 0, 1)
        if img_pred.shape[-1] == 1:
            axes[1, t].imshow(img_pred.squeeze(-1), cmap="gray")
        else:
            axes[1, t].imshow(img_pred)
        axes[1, t].set_title(f"Predicted t={t}", fontsize=9)
        axes[1, t].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def plot_sensor_rollout(real_sensor, pred_sensor, save_path: str = None, title: str = "Sensor Rollout"):
    """
    Plots a grid comparing real and predicted 1D sensor data values over time.
    
    Args:
        real_sensor: Ground truth sequence of sensor readings.
        pred_sensor: Predicted sequence of sensor readings.
        save_path: Optional path to save the generated plot image.
        title: Title of the generated plot.
    """
    if torch.is_tensor(real_sensor):
        real_sensor = real_sensor.detach().cpu().numpy()
    if torch.is_tensor(pred_sensor):
        pred_sensor = pred_sensor.detach().cpu().numpy()

    num_sensors = real_sensor.shape[1]
    
    cols = min(5, num_sensors)
    rows = (num_sensors + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
        
    plt.suptitle(title, fontsize=18)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < num_sensors:
            ax.plot(real_sensor[:, i], label="Real", color="blue", linestyle="-", alpha=0.8)
            ax.plot(pred_sensor[:, i], label="Pred", color="red", linestyle="--", alpha=0.8)
            ax.set_title(f"Sensor {i}", fontsize=10)
            ax.grid(True, alpha=0.5)
            if i == 0:
                ax.legend()
        else:
            ax.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def export_to_video(tensor_frames, filename: str = "rollout.mp4",
                    fps: int = 10, upscale_size: tuple = (512, 512)):
    """
    Exports a sequence of image tensors or arrays to an MP4 video file.
    
    Args:
        tensor_frames: Sequence of images to be converted into video.
        filename: Output filename for the MP4 video.
        fps: Frames per second for the exported video.
        upscale_size: Target resolution width and height to upscale the frames.
    """
    if torch.is_tensor(tensor_frames):
        frames = tensor_frames.detach().cpu().numpy()
    else:
        frames = np.array(tensor_frames)

    frames = np.transpose(frames, (0, 2, 3, 1))
    frames = (frames * 255.0).clip(0, 255).astype(np.uint8)

    N, H, W, C = frames.shape
    out_w, out_h = upscale_size if upscale_size is not None else (W, H)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (out_w, out_h))

    for i in range(N):
        frame = frames[i]
        if upscale_size is not None:
            frame = cv2.resize(frame, (out_w, out_h),
                               interpolation=cv2.INTER_NEAREST)
        if C == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()


def export_comparison_video(real_frames, pred_frames, filename: str = "comparison.mp4",
                            fps: int = 10, upscale_size: tuple = (512, 512)):
    """
    Exports a side-by-side comparison video of real and predicted image sequences.
    
    Args:
        real_frames: Ground truth sequence of images.
        pred_frames: Predicted sequence of images.
        filename: Output filename for the MP4 video.
        fps: Frames per second for the exported video.
        upscale_size: Target resolution width and height to upscale the frames.
    """
    if torch.is_tensor(real_frames):
        real_frames = real_frames.detach().cpu().numpy()
    else:
        real_frames = np.array(real_frames)

    if torch.is_tensor(pred_frames):
        pred_frames = pred_frames.detach().cpu().numpy()
    else:
        pred_frames = np.array(pred_frames)

    T = min(len(real_frames), len(pred_frames))
    if T == 0:
        raise ValueError("No frames available to export comparison video")

    real_frames = np.transpose(real_frames[:T], (0, 2, 3, 1))
    pred_frames = np.transpose(pred_frames[:T], (0, 2, 3, 1))
    real_frames = (real_frames * 255.0).clip(0, 255).astype(np.uint8)
    pred_frames = (pred_frames * 255.0).clip(0, 255).astype(np.uint8)

    N, H, W, C = real_frames.shape
    out_w, out_h = upscale_size if upscale_size is not None else (W, H)
    side_w = out_w * 2 if upscale_size is not None else W * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (side_w, out_h))

    for i in range(N):
        real = real_frames[i]
        pred = pred_frames[i]

        if upscale_size is not None:
            real = cv2.resize(real, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            pred = cv2.resize(pred, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        if C == 3:
            real = cv2.cvtColor(real, cv2.COLOR_RGB2BGR)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

        frame = np.concatenate([real, pred], axis=1)
        out.write(frame)

    out.release()


def export_triple_comparison_video(real_frames, pred1_frames, pred2_frames, filename: str = "comparison_triple.mp4",
                                   fps: int = 10, upscale_size: tuple = (512, 512)):
    """
    Exports a side-by-side-by-side triple comparison video using one real and two predicted image sequences.
    
    Args:
        real_frames: Ground truth sequence of images.
        pred1_frames: First predicted sequence of images.
        pred2_frames: Second predicted sequence of images.
        filename: Output filename for the MP4 video.
        fps: Frames per second for the exported video.
        upscale_size: Target resolution width and height to upscale the frames.
    """
    if torch.is_tensor(real_frames):
        real_frames = real_frames.detach().cpu().numpy()
    else:
        real_frames = np.array(real_frames)

    if torch.is_tensor(pred1_frames):
        pred1_frames = pred1_frames.detach().cpu().numpy()
    else:
        pred1_frames = np.array(pred1_frames)

    if torch.is_tensor(pred2_frames):
        pred2_frames = pred2_frames.detach().cpu().numpy()
    else:
        pred2_frames = np.array(pred2_frames)

    T = min(len(real_frames), len(pred1_frames), len(pred2_frames))
    if T == 0:
        raise ValueError("No frames available to export triple comparison video")

    real_frames = np.transpose(real_frames[:T], (0, 2, 3, 1))
    pred1_frames = np.transpose(pred1_frames[:T], (0, 2, 3, 1))
    pred2_frames = np.transpose(pred2_frames[:T], (0, 2, 3, 1))
    
    real_frames = (real_frames * 255.0).clip(0, 255).astype(np.uint8)
    pred1_frames = (pred1_frames * 255.0).clip(0, 255).astype(np.uint8)
    pred2_frames = (pred2_frames * 255.0).clip(0, 255).astype(np.uint8)

    N, H, W, C = real_frames.shape
    out_w, out_h = upscale_size if upscale_size is not None else (W, H)
    side_w = out_w * 3 if upscale_size is not None else W * 3

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (side_w, out_h))

    for i in range(N):
        real = real_frames[i]
        pred1 = pred1_frames[i]
        pred2 = pred2_frames[i]

        if upscale_size is not None:
            real = cv2.resize(real, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            pred1 = cv2.resize(pred1, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            pred2 = cv2.resize(pred2, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        if C == 3:
            real = cv2.cvtColor(real, cv2.COLOR_RGB2BGR)
            pred1 = cv2.cvtColor(pred1, cv2.COLOR_RGB2BGR)
            pred2 = cv2.cvtColor(pred2, cv2.COLOR_RGB2BGR)

        frame = np.concatenate([real, pred1, pred2], axis=1)
        out.write(frame)

    out.release()


def plot_rollout_quality_summary(steps, img_mse, img_ssim_dist, img_lpips,
                                 tac_mse, tac_shape, pos_mse, pos_shape,
                                 tac_corr, pos_corr, out_path):
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)

    # Image row
    axes[0, 0].plot(steps, img_mse, color="black", linewidth=2, label="MSE")
    axes[0, 0].set_title("Image MSE")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(steps, img_ssim_dist, color="blue", linewidth=2, label="1 - SSIM")
    if np.isfinite(img_lpips).any():
        axes[0, 1].plot(steps, img_lpips, color="red", linestyle="--", linewidth=2, label="LPIPS")
    axes[0, 1].set_title("Image anti-flat metrics")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Tactile row
    axes[1, 0].plot(steps, tac_mse, color="black", linewidth=2, label="MSE")
    axes[1, 0].set_title("Tactile MSE")
    axes[1, 0].set_ylabel("MSE")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(steps, tac_shape, color="purple", linewidth=2, label="Derivative shape error")
    axes[1, 1].set_title(f"Tactile trajectory shape error (corr={tac_corr:.3f})")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    # Proprio row
    axes[2, 0].plot(steps, pos_mse, color="black", linewidth=2, label="MSE")
    axes[2, 0].set_title("Proprioception MSE")
    axes[2, 0].set_xlabel("Rollout Step")
    axes[2, 0].set_ylabel("MSE")
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].legend()

    axes[2, 1].plot(steps, pos_shape, color="purple", linewidth=2, label="Derivative shape error")
    axes[2, 1].set_title(f"Proprio trajectory shape error (corr={pos_corr:.3f})")
    axes[2, 1].set_xlabel("Rollout Step")
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].legend()

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()


def calculate_metrics(data_pos, data_neg, n_samples, n_steps):
    import torch.nn.functional as F
    # 1. Noise floor (MSE between samples of the same action)
    noise_floor = []
    for t in range(n_steps):
        diffs = [F.mse_loss(data_pos[i, t], data_pos[j, t]).item() 
                 for i in range(n_samples) for j in range(i + 1, n_samples)]
        noise_floor.append(np.mean(diffs) if diffs else 0.0)

    # 2. Action Divergence (MSE between sample k of Pos vs sample k of Neg)
    divergence = []
    for t in range(n_steps):
        diffs = [F.mse_loss(data_pos[k, t], data_neg[k, t]).item() for k in range(n_samples)]
        divergence.append(np.mean(diffs))
    
    return noise_floor, divergence


def plot_results(results, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    modalities = ["Image", "Tactile", "Proprioception"]
    keys = ["imgs", "tacs", "pos"]
    
    n_samples = results["push_pos"]["imgs"].shape[0]
    n_steps = results["push_pos"]["imgs"].shape[1]

    for i, (ax, mod, key) in enumerate(zip(axes, modalities, keys)):
        noise, div = calculate_metrics(results["push_pos"][key], results["push_neg"][key], n_samples, n_steps)
        
        ax.plot(range(n_steps), noise, 'k--', label="Noise Floor (same action)")
        ax.plot(range(n_steps), div, 'r-', linewidth=2, label="Action Divergence")
        
        ax.set_title(mod)
        ax.set_xlabel("Rollout Step")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)


def plot_divergence_test_results(steps, img_mse, mean_imgs_pos, mean_imgs_neg,
                                 latent_var_pos, latent_var_neg, latent_div_pos_neg,
                                 mean_tac_pos, mean_tac_neg, mean_pos_pos, mean_pos_neg,
                                 tac_idx, pos_idx, switch_step, output_tag):

    import config

    def _tagged_name(stem: str, tag: str) -> str:
        return f"{stem}_{tag}.png" if tag else f"{stem}.png"

    out_path_img = config.PLOTS_DIR_EXP / _tagged_name("divergence_vision_trial1_point0", output_tag)
    out_path_img_int = config.PLOTS_DIR_EXP / _tagged_name("divergence_vision_intensity_trial1_point0", output_tag)
    out_path_latent = config.PLOTS_DIR_EXP / _tagged_name("latent_temporal_variation_trial1_point0", output_tag)
    out_path_latent_div = config.PLOTS_DIR_EXP / _tagged_name("latent_divergence_push_pos_vs_push_neg_trial1_point0", output_tag)
    out_path_tac = config.PLOTS_DIR_EXP / _tagged_name("divergence_tactile_sensor18_trial1_point0", output_tag)
    out_path_pos = config.PLOTS_DIR_EXP / _tagged_name("divergence_proprio_sensor18_trial1_point0", output_tag)
    out_path_summary = config.PLOTS_DIR_EXP / _tagged_name("divergence_trial1_point0", output_tag)

    out_path_img.parent.mkdir(parents=True, exist_ok=True)
    out_path_img_int.parent.mkdir(parents=True, exist_ok=True)
    out_path_latent.parent.mkdir(parents=True, exist_ok=True)
    out_path_latent_div.parent.mkdir(parents=True, exist_ok=True)
    out_path_tac.parent.mkdir(parents=True, exist_ok=True)
    out_path_pos.parent.mkdir(parents=True, exist_ok=True)
    out_path_summary.parent.mkdir(parents=True, exist_ok=True)

    latent_steps = np.arange(len(latent_var_pos))
    latent_div_steps = np.arange(len(latent_div_pos_neg))
    line_step = None if switch_step is None else max(0, switch_step - 1)

    mean_img_pos_intensity = mean_imgs_pos.view(mean_imgs_pos.shape[0], -1).mean(dim=1).cpu().numpy()
    mean_img_neg_intensity = mean_imgs_neg.view(mean_imgs_neg.shape[0], -1).mean(dim=1).cpu().numpy()

    tac_pos_vals = mean_tac_pos[:, tac_idx].cpu().numpy()
    tac_neg_vals = mean_tac_neg[:, tac_idx].cpu().numpy()

    pos_pos_vals = mean_pos_pos[:, pos_idx].cpu().numpy()
    pos_neg_vals = mean_pos_neg[:, pos_idx].cpu().numpy()


    # 1) Vision MSE plot
    plt.figure(figsize=(8, 4))
    plt.plot(steps, img_mse, color="black", linewidth=2)
    plt.title("Vision MSE between push_pos and push_neg (per step)")
    plt.xlabel("Rollout Step")
    plt.ylabel("MSE (pixels)")
    plt.grid(True, alpha=0.3)
    if line_step is not None:
        plt.axvline(line_step, color="gray", linestyle=":", linewidth=1.5)
    plt.tight_layout()
    plt.savefig(str(out_path_img), dpi=150)
    plt.close()

    # Also plot mean image intensity per step for the two rollouts
    plt.figure(figsize=(8, 4))
    plt.plot(steps, mean_img_pos_intensity, label="push_pos", color="blue")
    plt.plot(steps, mean_img_neg_intensity, label="push_neg", color="red", linestyle="--")
    plt.title("Mean image intensity: push_pos vs push_neg")
    plt.xlabel("Rollout Step")
    plt.ylabel("Mean intensity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if line_step is not None:
        plt.axvline(line_step, color="gray", linestyle=":", linewidth=1.5)
    plt.tight_layout()
    plt.savefig(str(out_path_img_int), dpi=150)
    plt.close()

    # 1b) Temporal latent variation plot
    plt.figure(figsize=(8, 4))
    plt.plot(latent_steps, latent_var_pos, label="push_pos", color="blue")
    plt.plot(latent_steps, latent_var_neg, label="push_neg", color="red", linestyle="--")
    plt.title("Temporal latent variation: ||z(t+1) - z(t)|| / sqrt(D)")
    plt.xlabel("Rollout Step")
    plt.ylabel("Normalized latent change")
    plt.legend()
    plt.grid(True, alpha=0.3)
    latent_min = 0.0
    latent_max = float(max(latent_var_pos.max(), latent_var_neg.max()))
    plt.ylim(latent_min, max(latent_max * 1.1, 1e-4))
    if line_step is not None:
        plt.axvline(line_step, color="gray", linestyle=":", linewidth=1.5)
    plt.tight_layout()
    plt.savefig(str(out_path_latent), dpi=150)
    plt.close()

    # 1c) Same-step latent divergence between the two action sequences
    plt.figure(figsize=(8, 4))
    plt.plot(latent_div_steps, latent_div_pos_neg, color="purple", linewidth=2)
    plt.title("Latent divergence: push_pos vs push_neg at the same step")
    plt.xlabel("Rollout Step")
    plt.ylabel("Normalized latent distance")
    plt.grid(True, alpha=0.3)
    latent_div_max = float(latent_div_pos_neg.max()) if len(latent_div_pos_neg) > 0 else 0.0
    plt.ylim(0.0, max(latent_div_max * 1.1, 1e-4))
    if line_step is not None:
        plt.axvline(line_step, color="gray", linestyle=":", linewidth=1.5)
    plt.tight_layout()
    plt.savefig(str(out_path_latent_div), dpi=150)
    plt.close()

    # 2) Tactile sensor plot
    plt.figure(figsize=(8, 4))
    plt.plot(steps, tac_pos_vals, label="push_pos", color="blue")
    plt.plot(steps, tac_neg_vals, label="push_neg", color="red", linestyle="--")
    plt.title(f"Tactile sensor {tac_idx}: push_pos vs push_neg")
    plt.xlabel("Rollout Step")
    plt.ylabel("Sensor value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if switch_step is not None:
        plt.axvline(switch_step, color="gray", linestyle=":", linewidth=1.5)
    plt.tight_layout()
    plt.savefig(str(out_path_tac), dpi=150)
    plt.close()

    # 3) Proprio sensor plot
    plt.figure(figsize=(8, 4))
    plt.plot(steps, pos_pos_vals, label="push_pos", color="blue")
    plt.plot(steps, pos_neg_vals, label="push_neg", color="red", linestyle="--")
    plt.title(f"Proprio sensor {pos_idx}: push_pos vs push_neg")
    plt.xlabel("Rollout Step")
    plt.ylabel("Sensor value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if switch_step is not None:
        plt.axvline(switch_step, color="gray", linestyle=":", linewidth=1.5)
    plt.tight_layout()
    plt.savefig(str(out_path_pos), dpi=150)
    plt.close()

    # Legacy summary plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    # Vision intensity panel
    axes[0].plot(steps, mean_img_pos_intensity, label="push_pos", color="blue")
    axes[0].plot(steps, mean_img_neg_intensity, label="push_neg", color="red", linestyle="--")
    axes[0].set_title("Mean image intensity: push_pos vs push_neg")
    axes[0].set_xlabel("Rollout Step")
    axes[0].set_ylabel("Mean intensity")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    if line_step is not None:
        axes[0].axvline(line_step, color="gray", linestyle=":", linewidth=1.5)

    # Tactile panel
    axes[1].plot(steps, tac_pos_vals, label="push_pos", color="blue")
    axes[1].plot(steps, tac_neg_vals, label="push_neg", color="red", linestyle="--")
    axes[1].set_title(f"Tactile sensor {tac_idx}: push_pos vs push_neg")
    axes[1].set_xlabel("Rollout Step")
    axes[1].set_ylabel("Sensor value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    if line_step is not None:
        axes[1].axvline(line_step, color="gray", linestyle=":", linewidth=1.5)

    # Proprio panel
    axes[2].plot(steps, pos_pos_vals, label="push_pos", color="blue")
    axes[2].plot(steps, pos_neg_vals, label="push_neg", color="red", linestyle="--")
    axes[2].set_title(f"Proprio sensor {pos_idx}: push_pos vs push_neg")
    axes[2].set_xlabel("Rollout Step")
    axes[2].set_ylabel("Sensor value")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    if line_step is not None:
        axes[2].axvline(line_step, color="gray", linestyle=":", linewidth=1.5)

    plt.tight_layout()
    plt.savefig(str(out_path_summary), dpi=150)
    plt.close()

    print(f"Saved vision plot to: {out_path_img}")
    print(f"Saved tactile plot to: {out_path_tac}")
    print(f"Saved proprio plot to: {out_path_pos}")
    print(f"Saved latent variation plot to: {out_path_latent}")
    print(f"Saved latent divergence plot to: {out_path_latent_div}")
    print(f"Saved legacy summary plot to: {out_path_summary}")


