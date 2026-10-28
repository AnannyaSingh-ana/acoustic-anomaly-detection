from pathlib import Path

# Folder paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "fan"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# Audio settings (same as the official DCASE 2020 baseline)
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
N_FRAMES = 313          # time frames per 10 s clip
VAL_FRACTION = 0.10     # 10% of train becomes validation
RANDOM_SEED = 42        # fixed seed = same split every time
MODELS_DIR = PROJECT_ROOT / "results" / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
THRESHOLD_PERCENTILE = 95   # threshold = 95th percentile of validation scores
# CNN autoencoder settings
PATCH_FRAMES = 64      # frames per patch (~2 s)
PATCH_HOP = 32         # patches overlap by half
LATENT_DIM = 128       # bottleneck size
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 50
PATIENCE = 5           # early stopping: stop after 5 epochs with no val improvement
# Machine-ID classifier settings
MACHINE_IDS = ["00", "02", "04", "06"]
DROPOUT = 0.3
CLF_MAX_EPOCHS = 30