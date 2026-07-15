import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def plot_losses_train_val(train_losses: list, val_losses: list,
                          save_path: str = None, start_epoch: int = 1):
    epochs = range(start_epoch, start_epoch + len(train_losses))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, train_losses, label="Train Loss",
            color="#2196F3", linewidth=2)
    ax.plot(epochs, val_losses, label="Val Loss",
            color="#F44336", linewidth=2, linestyle="--")

    best_epoch = int(np.argmin(val_losses)) + start_epoch
    best_val   = min(val_losses)
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
        print(f"  [Plot] Loss curves saved → {save_path}")
    else:
        plt.show()


def plot_losses(loss_history: dict, save_path: str = None):
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
    if torch.is_tensor(real_sensor):
        real_sensor = real_sensor.detach().cpu().numpy()
    if torch.is_tensor(pred_sensor):
        pred_sensor = pred_sensor.detach().cpu().numpy()

    num_sensors = real_sensor.shape[1]
    
    # Calculate grid size dynamically
    cols = min(5, num_sensors)
    rows = (num_sensors + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    # Handle the case where axes is not a 2D array
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

    print(f"  [Video] Exporting {N} frames → {filename}")

    for i in range(N):
        frame = frames[i]
        if upscale_size is not None:
            frame = cv2.resize(frame, (out_w, out_h),
                               interpolation=cv2.INTER_NEAREST)
        if C == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print(f"  [Video] Saved successfully.")


def export_comparison_video(real_frames, pred_frames, filename: str = "comparison.mp4",
                            fps: int = 10, upscale_size: tuple = (512, 512)):
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

    print(f"  [Video] Exporting comparison of {N} frames → {filename}")

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
    print(f"  [Video] Saved successfully.")


def export_triple_comparison_video(real_frames, pred1_frames, pred2_frames, filename: str = "comparison_triple.mp4",
                                   fps: int = 10, upscale_size: tuple = (512, 512)):
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

    print(f"  [Video] Exporting triple comparison of {N} frames → {filename}")

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
    print(f"  [Video] Saved triple comparison successfully.")