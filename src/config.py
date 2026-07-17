import torch
from pathlib import Path

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PLOTS_DIR_EXP = OUTPUTS_DIR / "plots/exp"
VIDEOS_DIR = OUTPUTS_DIR / "videos"

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

IMG_CHANNELS = None
IMG_LATENT_DIM = 64
TAC_FEATURES = None
TAC_LATENT_DIM = 8
PROPRIO_FEATURES = None
PROPRIO_LATENT_DIM = 8
ACTION_DIM = None  
HIDDEN_DIM = 256
DYNAMICS_HIDDEN_DIM = 512 #1024

LATENT_TYPE = "vae" # "vae" | "vqvae"
VQ_NUM_EMBEDDINGS = 1024
VQ_EMBEDDING_DIM = 8
VQ_COMMITMENT_COST = 0.25

DYNAMICS_TYPE = "mlp" # "mlp" | "flow_matching"
FLOW_MATCHING_STEPS  = 10

SEED = 42
MAX_EPOCHS = 400
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5
GRAD_CLIP = 1.0
TRAIN_PHASE = "dynamics"

BETA_KL = 0.2
IMG_LOSS_WEIGHT = 2.0 
TAC_LOSS_WEIGHT = 1.0  
PROPRIO_LOSS_WEIGHT = 1.0   

ES_PATIENCE = 40
ES_MIN_DELTA = 1e-4

ROLLOUT_STEPS = 50
VIDEO_FPS = 10
EVAL_ONE_STEP_SAMPLES = 3
EVAL_ROLLOUT_STARTS = 8
EVAL_ROLLOUT_SAMPLES = 3
EVAL_VISUAL_FRAMES = 8
EVAL_VISUAL_START = 0

VAE_BEST_MODEL_NAME = "best_vq_vae_complete.pt"
DYNAMICS_BEST_MODEL_NAME = "best_dynamics_vae_flow_complete.pt"
