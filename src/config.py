import torch
from pathlib import Path

ROOT        = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR  = OUTPUTS_DIR / "models"
PLOTS_DIR   = OUTPUTS_DIR / "plots"
VIDEOS_DIR  = OUTPUTS_DIR / "videos"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()         else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

DATASETS = [
    (1, "no_obj"),
    (2, "no_obj"),
    #(1, "obj"),
]
SPLIT_RATIO = (0.70, 0.15, 0.15)

IMG_CHANNELS   = 3
IMG_LATENT_DIM = 64
ACTION_DIM     = None  
HIDDEN_DIM     = 256

DYNAMICS_TYPE        = "diffusion"
DIFFUSION_STEPS      = 500
DIFFUSION_BETA_START = 1e-4
DIFFUSION_BETA_END   = 0.02
FLOW_MATCHING_STEPS  = 50

SEED          = 42
MAX_EPOCHS    = 400
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5
GRAD_CLIP     = 1.0
TRAIN_PHASE   = "dynamics"
TRAIN_RESUME  = False

BETA_KL = 0.2

ES_PATIENCE  = 40
ES_MIN_DELTA = 1e-4

ROLLOUT_STEPS = 50
VIDEO_FPS     = 10
EVAL_ONE_STEP_SAMPLES   = 3
EVAL_ROLLOUT_STARTS     = 8
EVAL_ROLLOUT_SAMPLES    = 3
EVAL_VISUAL_FRAMES      = 8
EVAL_VISUAL_START       = 0

VAE_BEST_MODEL_NAME      = "best_vae.pt"
DYNAMICS_BEST_MODEL_NAME = "best_dynamics_diffusion_2.pt"
