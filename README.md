# Multimodal World Model

This project provides a framework for building, training, and evaluating a multimodal World Model from scratch for robotics applications. It aims to learn a robust representation of the environment by combining visual, tactile, and proprioceptive data, predicting future states conditioned on robotic actions.

In particular, it implements:
* **Multimodal Architecture**: integrates Vision (Images), Tactile sensing, and Proprioception into a unified latent space.
* **Representation Learning**: features both VAE (Variational Autoencoder) and VQ-VAE (Vector Quantized-Variational Autoencoder) implementations for visual encoding and decoding.
* **Latent Dynamics Modeling**: highly customizable transition models supporting both deterministic (MLP) and continuous-time generative dynamics (Flow Matching).
* **Modular Codebase**: easily configurable architecture through a centralized configuration file, fully customizable network depths, latent dimensions, and training hyperparameters.
* **Optimization Techniques**: custom loss scaling for multi-objective optimization, gradient clipping, and robust weight initialization.
* **Training and Validation tools**: Early Stopping, automatic model checkpointing, dataset splitting, and data normalization.
* **Advanced Evaluation & Visualization**: computes 1-step and multi-step (rollout) MSE. Supports generation of prediction comparisons and video rollouts.

---

## Project Structure

| File / Directory                | Description                                                                           |
|:--------------------------------|:--------------------------------------------------------------------------------------|
| `src/train.py`                  | Core script managing the main training loops for the models (VAE and Dynamics).       |
| `src/evaluate.py`               | Main evaluation routine computing multi-step metrics and generating visualization plots and videos. |
| `src/evaluate_vae.py`           | Specific evaluation script for the standalone visual VAE/VQ-VAE model.                |
| `src/config.py`                 | Centralized configuration file for hyperparameters, paths, and model architecture.    |
| `src/models/world_model.py`     | Core class defining the full multimodal World Model architecture.                     |
| `src/models/dynamics.py`        | Contains the latent dynamics logic (e.g., `DynamicsMLP` and `FlowMatchingDynamics`).  |
| `src/models/encoders.py`        | Implementation of encoders for visual, tactile, and proprioceptive modalities.        |
| `src/models/decoders.py`        | Implementation of decoders mapping latents back to raw sensory data.                  |
| `src/models/vq.py`              | Implementation of the Vector Quantization layer for the VQ-VAE.                       |
| `src/utils/data_utils.py`       | Classes and utilities for dataset loading, normalization, and tensor concatenation.   |
| `src/utils/plot.py`             | Utilities for plotting learning curves, sensor rollout charts, and comparison videos. |
| `src/utils/early_stopping.py`   | Logic for patience-based Early Stopping to prevent overfitting.                       |
| `src/utils/checkpoint.py`       | Utilities to save and load model checkpoints during and after training.               |
| `src/utils/losses.py`           | Implementation of custom loss functions for multi-modal reconstruction.               |
| `src/utils/weights_init.py`     | Contains weight initialization methods across different modules.                      |

---

## Usage

### 1. Requirements
Ensure you have the necessary libraries installed. Install the project dependencies explicitly with pip:
```bash
pip install torch torchvision torchaudio numpy matplotlib opencv-python-headless scikit-learn torchmetrics lpips
```

### 2. Configuration
Before running, you can adjust hyperparameters, choose the `LATENT_TYPE` (`vae` or `vqvae`), and select the `DYNAMICS_TYPE` (`mlp` or `flow_matching`) inside `src/config.py`. 

### 3. Run Training
To start training the world model:
```bash
cd src
python3 train.py
```

### 4. Run Evaluation
After training, you can evaluate the models and generate plots/videos for 1-step and multi-step rollouts:
```bash
# Evaluate the full dynamics and world model
python3 evaluate.py

# Evaluate the standalone VAE visually
python3 evaluate_vae.py
```
Outputs, including generated videos and metric text files, will be saved into the `src/outputs` folder by default.
