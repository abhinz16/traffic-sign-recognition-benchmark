"""Project configuration.

The default setup trains on every compatible dataset found on disk. GTSRB is
used as the common 43-class label space. External datasets are added only when
their class folders can be mapped to the GTSRB labels.
"""

SEED = 42

# Dataset control
DATASET_MODE = "combined"       # "combined", "gtsrb", or "demo"
DOWNLOAD_ALL_DATASETS = True
DATA_DIR = "data"
NUM_CLASSES = 43
IMAGE_SIZE = 64
VALID_SPLIT = 0.15

# External datasets
# GTSRB is downloaded through torchvision. BelgiumTS and Mapillary are attempted
# through Kaggle mirrors when kagglehub and Kaggle credentials are available.
BELGIUM_KAGGLE_DATASET = "abhi8923shriv/belgium-ts"
MAPILLARY_KAGGLE_DATASET = "zeuss2k3/mapillary-traffic-sign-dataset"

# Combined training behavior
# External datasets are allowed into the training set only after their folder
# names are mapped to GTSRB's 43 labels. This avoids mixing different label IDs.
USE_EXTERNAL_DATASETS_FOR_TRAINING = True
MIN_EXTERNAL_IMAGES_PER_DATASET = 25

# Training
MODEL_NAME = "resnet18"         # "small_cnn", "resnet18", or "efficientnet_b0"
EPOCHS = 8
BATCH_SIZE = 96
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

# Hardware
# CUDA is used when available. Without CUDA, PyTorch and DataLoader use as many
# CPU cores as possible while leaving this many cores free.
CPU_CORES_TO_LEAVE_FREE = 3
NUM_WORKERS = "auto"
USE_AMP = True

# Robustness test settings
RUN_ROBUSTNESS_TESTS = True
NOISE_LEVELS = [0.0, 0.03, 0.06, 0.10]
BLUR_LEVELS = [0, 3, 5]
BRIGHTNESS_LEVELS = [0.65, 1.0, 1.35]

# External evaluation
RUN_EXTERNAL_DATASET_EVAL = True

# Failure-case export
SAVE_FAILURE_CASES = True
MAX_FAILURE_CASES = 36

# Fast pipeline check. Used only when DATASET_MODE = "demo".
DEMO_SAMPLES = 1500
