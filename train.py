# ============================================================
# train.py
# PURPOSE: Build EfficientNetB0 + Metadata Fusion model
#          and train it on HAM10000 dataset.
#
# RUN THIS ON GOOGLE COLAB — NOT locally in VS Code
# Steps to run on Colab are at the bottom of this file.
# ============================================================

# ============================================================
# SECTION 1 — COLAB SETUP (Read this before running)
# ============================================================
# STEP 1: Open https://colab.research.google.com
# STEP 2: Click Runtime → Change Runtime Type → T4 GPU → Save
# STEP 3: Mount Google Drive by running this in a Colab cell:
#         from google.colab import drive
#         drive.mount('/content/drive')
# STEP 4: Upload your SkinLesionClassifier folder to Google Drive
# STEP 5: Run this file in Colab:
#         !python train.py
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight

# Import our custom files
import config
from utils.data_loader import load_dataset
from utils.augmentation import SkinLesionDataGenerator

# ============================================================
# SECTION 2 — SETUP
# ============================================================

# Set random seeds so results are reproducible
# Same seed = same random numbers = same results every run
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)

print("\n" + "="*55)
print("  SKIN LESION CLASSIFIER — TRAINING")
print("="*55)

# Check if GPU is available
# Training on GPU is 10-20x faster than CPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU detected: {gpus[0].name}")
    print("   Training will be fast!")
else:
    print("⚠️  No GPU detected — using CPU")
    print("   Training will be slow. Please enable GPU in Colab:")
    print("   Runtime → Change Runtime Type → T4 GPU")

# ============================================================
# SECTION 3 — DETECT IF RUNNING ON COLAB OR LOCALLY
# ============================================================
# This automatically detects if we are on Colab or local machine
# and sets the correct paths
try:
    import google.colab
    ON_COLAB = True
    print("\n📍 Running on Google Colab")
except ImportError:
    ON_COLAB = False
    print("\n📍 Running locally in VS Code")

# ============================================================
# SECTION 4 — LOAD DATASET
# ============================================================
print("\n" + "-"*55)
print("  STEP 1: Loading Dataset")
print("-"*55)

# load_dataset() from data_loader.py does all the work:
# - Loads metadata CSV
# - Finds all image paths
# - Encodes age, sex, localization
# - Splits into train/val/test (70/15/15)
# - Saves test split CSV for evaluate.py
train_df, val_df, test_df = load_dataset(use_colab=ON_COLAB)

print(f"\n✅ Dataset loaded successfully:")
print(f"   Training images   : {len(train_df)}")
print(f"   Validation images : {len(val_df)}")
print(f"   Test images       : {len(test_df)}")

# ============================================================
# SECTION 5 — COMPUTE CLASS WEIGHTS
# ============================================================
# HAM10000 is imbalanced — nv has 6705 images, df has only 115
# class_weight tells the model to penalise mistakes on rare
# classes more heavily, forcing it to learn ALL 7 classes
print("\n" + "-"*55)
print("  STEP 2: Computing Class Weights")
print("-"*55)

# Get all training labels as a list
train_labels = train_df['label'].values

# compute_class_weight calculates the correct weight for each class
# Classes with fewer images get HIGHER weights (more penalty)
class_weights_array = compute_class_weight(
    class_weight='balanced',      # automatically balance all classes
    classes=np.unique(train_labels),
    y=train_labels
)

# Convert array to dictionary format that Keras expects
# {0: weight_for_class0, 1: weight_for_class1, ...}
class_weight_dict = dict(enumerate(class_weights_array))

print("✅ Class weights computed:")
for idx, cls in enumerate(config.CLASS_NAMES):
    print(f"   {cls:6s} (class {idx}): weight = {class_weight_dict[idx]:.3f}")

# ============================================================
# SECTION 6 — CREATE DATA GENERATORS
# ============================================================
# Generators load images in batches during training
# They do NOT load all 10,015 images at once (too much RAM)
# Instead they load 32 images at a time as needed
print("\n" + "-"*55)
print("  STEP 3: Creating Data Generators")
print("-"*55)

# Training generator — augment=True applies random flips,
# rotations, zoom etc. to create more variety
train_generator = SkinLesionDataGenerator(
    dataframe=train_df,
    batch_size=config.BATCH_SIZE,
    augment=True,     # Apply augmentation for training
    shuffle=True      # Shuffle order each epoch
)

# Validation generator — augment=False, no shuffling
# We want consistent validation results each epoch
val_generator = SkinLesionDataGenerator(
    dataframe=val_df,
    batch_size=config.BATCH_SIZE,
    augment=False,    # No augmentation for validation
    shuffle=False     # Keep same order always
)

print("✅ Generators created successfully")

# ============================================================
# SECTION 7 — BUILD THE MODEL
# ============================================================
print("\n" + "-"*55)
print("  STEP 4: Building Model Architecture")
print("-"*55)

# --- IMAGE BRANCH ---
# Input 1: skin lesion image (224 x 224 x 3 pixels)
image_input = keras.Input(
    shape=config.INPUT_SHAPE,
    name='image_input'
)

# Load EfficientNetB0 with ImageNet pretrained weights
# include_top=False removes the original 1000-class output layer
# We will add our own 7-class output layer
base_model = EfficientNetB0(
    weights='imagenet',     # use weights trained on 1.2M ImageNet images
    include_top=False,      # remove original classification head
    input_tensor=image_input
)

# FREEZE all EfficientNetB0 layers for Phase 1
# frozen = weights cannot change during Phase 1 training
# We only train our new layers first
base_model.trainable = False
print(f"✅ EfficientNetB0 loaded: {len(base_model.layers)} layers (all frozen)")

# Add layers on top of EfficientNetB0
# GlobalAveragePooling2D converts the feature map to a 1D vector
# Example: (7, 7, 1280) feature map → (1280,) vector
x = layers.GlobalAveragePooling2D(name='gap')(base_model.output)

# Dense layer learns to combine EfficientNet features
# relu activation: negative values become 0, positive stay same
x = layers.Dense(256, activation='relu', name='dense_image')(x)

# BatchNormalization: normalizes values to speed up training
# Keeps values in a good range for the next layer
x = layers.BatchNormalization(name='bn_image')(x)

# Dropout: randomly turns off 40% of neurons during training
# Forces other neurons to compensate — prevents overfitting
x = layers.Dropout(config.DROPOUT_IMAGE, name='dropout_image')(x)

# --- METADATA BRANCH ---
# Input 2: patient metadata (3 numbers: age, sex, localization)
metadata_input = keras.Input(
    shape=(3,),
    name='metadata_input'
)

# Small neural network to process the 3 metadata values
m = layers.Dense(32, activation='relu', name='dense_meta1')(metadata_input)
m = layers.Dense(64, activation='relu', name='dense_meta2')(m)
m = layers.BatchNormalization(name='bn_meta')(m)

# --- FUSION LAYER ---
# Concatenate: join image features (256 values) and
# metadata features (64 values) into one vector (320 values)
# This is the KEY innovation — combining image + patient data
combined = layers.Concatenate(name='fusion')([x, m])

# Process the combined features
combined = layers.Dense(128, activation='relu', name='dense_fusion')(combined)
combined = layers.Dropout(config.DROPOUT_FUSION, name='dropout_fusion')(combined)

# --- OUTPUT LAYER ---
# Softmax converts 7 numbers into 7 probabilities that sum to 1.0
# Example output: [0.01, 0.02, 0.03, 0.05, 0.87, 0.01, 0.01]
#                  akiec bcc  bkl   df   mel   nv  vasc
# → Predicted class: mel (index 4, probability 0.87 = 87%)
output = layers.Dense(
    config.NUM_CLASSES,
    activation='softmax',
    name='output'
)(combined)

# --- ASSEMBLE MODEL ---
# Create the final model with 2 inputs and 1 output
model = Model(
    inputs=[image_input, metadata_input],
    outputs=output,
    name='SkinLesionClassifier'
)

# Print model summary — shows all layers and parameter counts
model.summary()

print(f"\n✅ Model built successfully")
print(f"   Total parameters     : {model.count_params():,}")
print(f"   Trainable parameters : "
      f"{sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ============================================================
# SECTION 8 — CALLBACKS
# ============================================================
# Callbacks are automatic helpers that run during training

# Create directory for saving model
os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

# EarlyStopping: stop training if val_accuracy doesn't improve
# patience=5 means wait 5 epochs before stopping
# restore_best_weights=True goes back to best epoch's weights
early_stopping = EarlyStopping(
    monitor='val_accuracy',    # watch validation accuracy
    patience=config.EARLY_STOP_PATIENCE,
    restore_best_weights=True, # use best weights, not last weights
    verbose=1                  # print message when triggered
)

# ModelCheckpoint: save model whenever val_accuracy improves
# save_best_only=True only saves when it's a new best score
model_checkpoint = ModelCheckpoint(
    filepath=config.MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,       # only save if better than before
    verbose=1                  # print message when saved
)

# ReduceLROnPlateau: halve learning rate when val_loss stagnates
# patience=3 means wait 3 epochs before reducing
# factor=0.5 means multiply learning rate by 0.5 (halve it)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=config.REDUCE_LR_FACTOR,
    patience=config.REDUCE_LR_PATIENCE,
    min_lr=1e-7,               # never go below this learning rate
    verbose=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]
print("✅ Callbacks configured")

# ============================================================
# SECTION 9 — PHASE 1 TRAINING
# ============================================================
# Train ONLY the new top layers we added
# EfficientNetB0 base layers are frozen (cannot change)
# Learning rate: 1e-3 (fast learning — safe because base is frozen)
print("\n" + "="*55)
print("  PHASE 1 TRAINING — Feature Extraction")
print(f"  Epochs: {config.PHASE1_EPOCHS} | LR: {config.PHASE1_LR}")
print(f"  EfficientNetB0: FROZEN")
print("="*55)

# Compile the model with Phase 1 settings
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=config.PHASE1_LR),
    loss='categorical_crossentropy',  # standard loss for multi-class
    metrics=['accuracy']
)

# Train Phase 1
history1 = model.fit(
    train_generator,
    epochs=config.PHASE1_EPOCHS,
    validation_data=val_generator,
    class_weight=class_weight_dict,   # handle class imbalance
    callbacks=callbacks,
    verbose=1
)

print(f"\n✅ Phase 1 complete!")
print(f"   Best val accuracy: "
      f"{max(history1.history['val_accuracy'])*100:.2f}%")

# ============================================================
# SECTION 10 — PHASE 2 FINE-TUNING
# ============================================================
# Unfreeze the last 20 layers of EfficientNetB0
# Very small learning rate to gently adapt pretrained weights
# to skin lesion images without destroying ImageNet knowledge
print("\n" + "="*55)
print("  PHASE 2 TRAINING — Fine-Tuning")
print(f"  Epochs: {config.PHASE2_EPOCHS} | LR: {config.PHASE2_LR}")
print(f"  EfficientNetB0 last {config.UNFREEZE_LAYERS} layers: UNFROZEN")
print("="*55)

# Unfreeze the whole base model first
base_model.trainable = True

#