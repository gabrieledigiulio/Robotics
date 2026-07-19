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

# Number of channels in the input image
IMG_CHANNELS = None
# Size of the compressed latent representation for images
IMG_LATENT_DIM = 64
# Number of raw tactile features/sensors
TAC_FEATURES = None
# Size of the compressed latent representation for tactile data
TAC_LATENT_DIM = 8
# Number of raw proprioceptive features 
PROPRIO_FEATURES = None
# Size of the compressed latent representation for proprioception
PROPRIO_LATENT_DIM = 8
# Dimensionality of the control action space
ACTION_DIM = None  
# Number of hidden units for standard neural network layers
HIDDEN_DIM = 256
# Number of hidden units specifically for the dynamics model
DYNAMICS_HIDDEN_DIM = 256 #1024


# Autoencoder architecture
LATENT_TYPE = "vae" # "vae" | "vqvae"

# Size of the discrete dictionary/codebook 
VQ_NUM_EMBEDDINGS = 1024
# Dimensionality of each discrete codebook vector
VQ_EMBEDDING_DIM = 8
# Weight for the commitment loss in VQ-VAE training
VQ_COMMITMENT_COST = 0.20

# Dynamics architecture
DYNAMICS_TYPE = "mlp" # "mlp" | "flow_matching"
# Number of integration steps
FLOW_MATCHING_STEPS  = 50


# Random seed for reproducibility
SEED = 42
# Maximum number of training epochs
MAX_EPOCHS = 1000
# Number of samples per training batch
BATCH_SIZE = 64
# Step size for the optimizer
LEARNING_RATE = 2e-4 
# L2 regularization penalty
WEIGHT_DECAY  = 1e-5
# Maximum norm for gradients
GRAD_CLIP = 1.0
# Current training phase
TRAIN_PHASE = "dynamics"
# Weight for the KL-divergence loss term 
BETA_KL = 0.1
# Importance weight for the image reconstruction loss
IMG_LOSS_WEIGHT = 1.0 
# Importance weight for the tactile reconstruction loss
TAC_LOSS_WEIGHT = 1.0  
# Importance weight for the proprioception reconstruction loss
PROPRIO_LOSS_WEIGHT = 1.0   
#Number of epochs to wait for validation loss improvement before stopping
ES_PATIENCE = 50
# Minimum change in loss required to qualify as an improvement
ES_MIN_DELTA = 1e-5

# Evaluation and Visualization
# Number of future time steps to predict during evaluation
ROLLOUT_STEPS = 50
# Frames per second for generating rollout visualization videos
VIDEO_FPS = 10
# Number of random samples to evaluate for one-step predictions
EVAL_ONE_STEP_SAMPLES = 3
# Number of different initial states to start evaluation rollouts from
EVAL_ROLLOUT_STARTS = 8
# Number of different probabilistic rollouts to sample per starting state
EVAL_ROLLOUT_SAMPLES = 3
# Number of specific frames to plot side-by-side in visualizations
EVAL_VISUAL_FRAMES = 8
# Time step index to start extracting visual frames from
EVAL_VISUAL_START = 0

# Files name
VAE_BEST_MODEL_NAME = "best_vae_completef.pt"
DYNAMICS_BEST_MODEL_NAME = "best_dynamics_vae_mlp_complete.pt"
