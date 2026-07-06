# ============================================================
# config.py — Central Configuration File
# All project settings are stored here in one place.
# Every other file imports from this file.
# ============================================================

import os

# ============================================================
# SECTION 1 — BASE PROJECT PATH
# os.path.dirname(__file__) means: the folder where config.py is saved
# This makes all paths work correctly on any computer
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# SECTION 2 — DATASET PATHS
# Where your HAM10000 images and CSV file are located
# ============================================================

# Path to the raw dataset folder
DATASET_RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")

# Path to HAM10000 images part 1 folder
IMAGE_DIR_PART1 = os.path.join(DATASET_RAW_DIR, "HAM10000_images_part_1")

# Path to HAM10000 images part 2 folder
IMAGE_DIR_PART2 = os.path.join(DATASET_RAW_DIR, "archive", "images")

# Path to the metadata CSV file (contains age, sex, localization, diagnosis)
METADATA_CSV = os.path.join(DATASET_RAW_DIR, "archive", "HAM10000_metadata.csv")

# Path to processed data folder (we will save cleaned data here)
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")

# ============================================================
# SECTION 3 — GOOGLE COLAB PATHS
# When training on Google Colab, use these paths instead
# Mount your Google Drive and put dataset inside this folder
# ============================================================
COLAB_BASE_DIR = "/content/drive/MyDrive/SkinLesionClassifier"
COLAB_DATASET_DIR = "/content/drive/MyDrive/SkinLesionClassifier/dataset/raw"
COLAB_IMAGE_PART1 = "/content/drive/MyDrive/SkinLesionClassifier/dataset/raw/HAM10000_images_part_1"
COLAB_IMAGE_PART2 = "/content/drive/MyDrive/SkinLesionClassifier/dataset/raw/HAM10000_images_part_2"
COLAB_METADATA_CSV = "/content/drive/MyDrive/SkinLesionClassifier/dataset/raw/HAM10000_metadata.csv"

# ============================================================
# SECTION 4 — MODEL SAVE PATHS
# Where the trained model will be saved after training
# ============================================================

# Best model saved here during training (ModelCheckpoint saves here)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "best_model.h5")

# Label encoder for localization column saved here
ENCODER_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "localization_encoder.pkl")

# Training history CSV saved here (used by evaluate.py for plots)
HISTORY_SAVE_PATH = os.path.join(BASE_DIR, "outputs", "training_history.csv")

# Test split CSV saved here (used by evaluate.py)
TEST_SPLIT_PATH = os.path.join(BASE_DIR, "outputs", "test_split.csv")

# ============================================================
# SECTION 5 — OUTPUT PATHS
# Where all plots and reports will be saved
# ============================================================
PLOTS_DIR = os.path.join(BASE_DIR, "outputs", "plots")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs", "reports")
EVALUATION_REPORT_PATH = os.path.join(REPORTS_DIR, "evaluation_report.txt")

# ============================================================
# SECTION 6 — IMAGE SETTINGS
# EfficientNetB0 expects images of exactly 224x224 pixels
# ============================================================

# Input image dimensions (width and height)
IMAGE_SIZE = (224, 224)

# Image size with colour channels (224 wide, 224 tall, 3 colours: Red Green Blue)
INPUT_SHAPE = (224, 224, 3)

# ============================================================
# SECTION 7 — TRAINING SETTINGS
# These numbers control how the model trains
# ============================================================

# Random seed — keeps results reproducible (same result every time you run)
RANDOM_SEED = 42

# Batch size — number of images processed together before updating weights
BATCH_SIZE = 32

# Phase 1 epochs — training with frozen EfficientNet base (faster learning)
PHASE1_EPOCHS = 10

# Phase 2 epochs — fine-tuning with unfrozen last 20 layers (slower, careful)
PHASE2_EPOCHS = 20

# Total epochs = Phase 1 + Phase 2
TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS

# Phase 1 learning rate — larger because we are only training new layers
PHASE1_LR = 1e-3

# Phase 2 learning rate — very small to avoid destroying pretrained weights
PHASE2_LR = 1e-5

# Number of layers to unfreeze in EfficientNetB0 during Phase 2
UNFREEZE_LAYERS = 20

# Dropout rate for the image branch (prevents overfitting)
DROPOUT_IMAGE = 0.4

# Dropout rate for the fusion layer
DROPOUT_FUSION = 0.3

# ============================================================
# SECTION 8 — DATA SPLIT RATIOS
# How to divide dataset into train, validation, test sets
# 70% training, 15% validation, 15% testing
# ============================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# SECTION 9 — CLASS SETTINGS
# The 7 types of skin lesions in HAM10000
# ============================================================

# Number of output classes
NUM_CLASSES = 7

# Short class codes used in the dataset CSV (dx column)
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Full disease names for display in app and reports
CLASS_FULL_NAMES = {
    'akiec': 'Actinic Keratoses',
    'bcc'  : 'Basal Cell Carcinoma',
    'bkl'  : 'Benign Keratosis',
    'df'   : 'Dermatofibroma',
    'mel'  : 'Melanoma',
    'nv'   : 'Melanocytic Nevi',
    'vasc' : 'Vascular Lesions'
}

# ============================================================
# SECTION 10 — METADATA SETTINGS
# The 3 patient information columns used as model input
# ============================================================

# Column name for patient age in metadata CSV
AGE_COLUMN = 'age'

# Column name for patient sex in metadata CSV
SEX_COLUMN = 'sex'

# Column name for lesion body location in metadata CSV
LOC_COLUMN = 'localization'

# Column name for diagnosis (ground truth label)
LABEL_COLUMN = 'dx'

# Column name for image file name
IMAGE_ID_COLUMN = 'image_id'

# All possible localization values in HAM10000
LOCALIZATION_CATEGORIES = [
    'abdomen', 'acral', 'back', 'chest', 'ear', 'face',
    'foot', 'genital', 'hand', 'lower extremity', 'neck',
    'scalp', 'trunk', 'unknown', 'upper extremity'
]

# ============================================================
# SECTION 11 — CALLBACK SETTINGS
# Settings for EarlyStopping, ReduceLROnPlateau
# ============================================================

# EarlyStopping — stop training if val_accuracy doesn't improve for 5 epochs
EARLY_STOP_PATIENCE = 5

# ReduceLROnPlateau — halve learning rate if val_loss doesn't improve for 3 epochs
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

# ============================================================
# SECTION 12 — STREAMLIT APP SETTINGS
# Used by the web application
# ============================================================

# App title shown in browser tab
APP_TITLE = "DermAI — Skin Lesion Classifier"

# App page icon
APP_ICON = "🔬"

# ============================================================
# SECTION 13 — PRINT CONFIRMATION
# When any file imports config.py, this message prints
# confirming config loaded successfully
# ============================================================
print("✅ config.py loaded successfully")
print(f"   Base directory : {BASE_DIR}")
print(f"   Classes        : {CLASS_NAMES}")
print(f"   Image size     : {IMAGE_SIZE}")
print(f"   Total epochs   : {TOTAL_EPOCHS}")