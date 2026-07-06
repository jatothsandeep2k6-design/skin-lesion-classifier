# ============================================================
# evaluate.py
# PURPOSE: Load trained model, run predictions on test set,
#          calculate all metrics, generate all plots,
#          save evaluation report.
#
# RUN THIS ON GOOGLE COLAB AFTER TRAINING IS COMPLETE
# Make sure these files exist before running:
#   - saved_models/best_model.h5
#   - saved_models/localization_encoder.pkl
#   - outputs/test_split.csv
#   - outputs/training_history.csv
#   - dataset/raw/HAM10000_metadata.csv
# ============================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# Import our custom files
import config
from utils.metrics import (
    compute_all_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
    plot_f1_per_class,
    save_evaluation_report
)

# ============================================================
# SECTION 1 — SETUP
# ============================================================
print("\n" + "="*55)
print("  SKIN LESION CLASSIFIER — EVALUATION")
print("="*55)

# Detect if running on Colab or locally
try:
    import google.colab
    ON_COLAB = True
    print("📍 Running on Google Colab")
except ImportError:
    ON_COLAB = False
    print("📍 Running locally in VS Code")

# ============================================================
# SECTION 2 — LOAD TRAINED MODEL
# ============================================================
print("\n" + "-"*55)
print("  STEP 1: Loading Trained Model")
print("-"*55)

# Check if model file exists
if not os.path.exists(config.MODEL_SAVE_PATH):
    print(f"❌ Model file not found at: {config.MODEL_SAVE_PATH}")
    print("   Please complete training first (run train.py on Colab)")
    print("   Then copy best_model.h5 to saved_models/ folder")
    sys.exit(1)

# Load the trained model
print(f"📂 Loading model from: {config.MODEL_SAVE_PATH}")
model = tf.keras.models.load_model(config.MODEL_SAVE_PATH)
print("✅ Model loaded successfully")
print(f"   Input shapes : {[inp.shape for inp in model.inputs]}")
print(f"   Output shape : {model.output.shape}")

# ============================================================
# SECTION 3 — LOAD LOCALIZATION ENCODER
# ============================================================
print("\n" + "-"*55)
print("  STEP 2: Loading Localization Encoder")
print("-"*55)

# Check if encoder file exists
if not os.path.exists(config.ENCODER_SAVE_PATH):
    print(f"❌ Encoder not found at: {config.ENCODER_SAVE_PATH}")
    print("   This file is saved automatically during training.")
    sys.exit(1)

# Load the saved encoder
with open(config.ENCODER_SAVE_PATH, 'rb') as f:
    localization_encoder = pickle.load(f)

print(f"✅ Encoder loaded successfully")
print(f"   Known localizations: {list(localization_encoder.classes_)}")

# ============================================================
# SECTION 4 — LOAD TEST DATA
# ============================================================
print("\n" + "-"*55)
print("  STEP 3: Loading Test Dataset")
print("-"*55)

# Check test split CSV exists
if not os.path.exists(config.TEST_SPLIT_PATH):
    print(f"❌ Test split not found at: {config.TEST_SPLIT_PATH}")
    print("   This file is saved automatically during training.")
    sys.exit(1)

# Load test split — created and saved by data_loader.py during training
test_df = pd.read_csv(config.TEST_SPLIT_PATH)
print(f"✅ Test split loaded: {len(test_df)} images")

# Show class distribution in test set
print(f"\n📊 Test set class distribution:")
label_counts = test_df['label'].value_counts().sort_index()
for idx, count in label_counts.items():
    cls = config.CLASS_NAMES[idx]
    print(f"   {cls:6s} (class {idx}): {count} images")

# ============================================================
# SECTION 5 — LOAD AND PREPROCESS TEST IMAGES
# ============================================================
print("\n" + "-"*55)
print("  STEP 4: Loading and Preprocessing Test Images")
print("-"*55)

# Create empty arrays to store all test images and metadata
n_test = len(test_df)

# Array for images: shape (n_test, 224, 224, 3)
test_images = np.zeros(
    (n_test, config.INPUT_SHAPE[0],
     config.INPUT_SHAPE[1], config.INPUT_SHAPE[2]),
    dtype=np.float32
)

# Array for metadata: shape (n_test, 3)
# 3 features: age, sex, localization
test_metadata = np.zeros((n_test, 3), dtype=np.float32)

# Array for true labels
test_labels = test_df['label'].values.astype(int)

# Load each test image one by one
print(f"📂 Loading {n_test} test images...")
failed_count = 0

for i, (_, row) in enumerate(test_df.iterrows()):

    # Show progress every 100 images
    if i % 100 == 0:
        print(f"   Progress: {i}/{n_test} images loaded...")

    # --- LOAD AND PREPROCESS IMAGE ---
    try:
        # Open image file
        img = Image.open(row['image_path'])

        # Convert to RGB (ensures 3 channels always)
        img = img.convert('RGB')

        # Resize to 224x224
        img = img.resize(config.IMAGE_SIZE)

        # Convert to numpy array and normalize to 0-1
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Store in our images array
        test_images[i] = img_array

    except Exception as e:
        # If image fails to load use blank image
        print(f"   ⚠️ Failed to load: {row['image_path']} — {e}")
        test_images[i] = np.zeros(config.INPUT_SHAPE, dtype=np.float32)
        failed_count += 1

    # --- LOAD METADATA ---
    # Age is already normalized (divided by 100) in the CSV
    test_metadata[i, 0] = float(row[config.AGE_COLUMN])

    # Sex is already encoded (0/0.5/1) in the CSV
    test_metadata[i, 1] = float(row[config.SEX_COLUMN])

    # Localization is already encoded (integer) in the CSV
    test_metadata[i, 2] = float(row[config.LOC_COLUMN])

print(f"\n✅ Test images loaded successfully")
print(f"   Total loaded : {n_test - failed_count}")
if failed_count > 0:
    print(f"   Failed       : {failed_count} (replaced with blank)")

# ============================================================
# SECTION 6 — RUN PREDICTIONS
# ============================================================
print("\n" + "-"*55)
print("  STEP 5: Running Model Predictions")
print("-"*55)

print("🔄 Running predictions on test set...")
print("   (This may take 1-2 minutes)")

# model.predict() runs all test images through the model
# Returns probability array of shape (n_test, 7)
# Each row is 7 probabilities summing to 1.0
# Example row: [0.01, 0.02, 0.03, 0.05, 0.87, 0.01, 0.01]
#               akiec bcc  bkl   df   mel   nv  vasc
y_pred_probs = model.predict(
    [test_images, test_metadata],
    batch_size=config.BATCH_SIZE,
    verbose=1   # show progress bar
)

print(f"\n✅ Predictions complete")
print(f"   Prediction array shape: {y_pred_probs.shape}")
print(f"   Sample prediction (first test image):")
print(f"   True class : {config.CLASS_NAMES[test_labels[0]]}")
pred_class = np.argmax(y_pred_probs[0])
print(f"   Predicted  : {config.CLASS_NAMES[pred_class]} "
      f"({y_pred_probs[0][pred_class]*100:.1f}% confidence)")

# ============================================================
# SECTION 7 — CALCULATE ALL METRICS
# ============================================================
print("\n" + "-"*55)
print("  STEP 6: Calculating All Metrics")
print("-"*55)

# compute_all_metrics() from utils/metrics.py calculates:
# accuracy, precision, recall, F1, ROC-AUC for all 7 classes
metrics_dict = compute_all_metrics(
    y_true=test_labels,
    y_pred_probs=y_pred_probs,
    class_names=config.CLASS_NAMES
)

# ============================================================
# SECTION 8 — GENERATE ALL PLOTS
# ============================================================
print("\n" + "-"*55)
print("  STEP 7: Generating All Plots")
print("-"*55)

# Create output directories if they don't exist
os.makedirs(config.PLOTS_DIR, exist_ok=True)
os.makedirs(config.REPORTS_DIR, exist_ok=True)

# --- PLOT 1: CONFUSION MATRIX ---
plot_confusion_matrix(
    y_true=test_labels,
    y_pred=metrics_dict['y_pred'],
    class_names=config.CLASS_NAMES,
    save_path=os.path.join(config.PLOTS_DIR, 'confusion_matrix.png')
)

# --- PLOT 2: ROC CURVES ---
plot_roc_curves(
    y_true_bin=metrics_dict['y_true_bin'],
    y_pred_probs=y_pred_probs,
    class_names=config.CLASS_NAMES,
    save_path=os.path.join(config.PLOTS_DIR, 'roc_curves.png')
)

# --- PLOT 3: TRAINING HISTORY ---
# Only if history CSV exists
if os.path.exists(config.HISTORY_SAVE_PATH):
    plot_training_history(
        history_csv_path=config.HISTORY_SAVE_PATH,
        save_path=os.path.join(config.PLOTS_DIR, 'training_history.png')
    )
else:
    print(f"⚠️  Training history not found — skipping history plot")
    print(f"   Expected at: {config.HISTORY_SAVE_PATH}")

# --- PLOT 4: F1 PER CLASS BAR CHART ---
plot_f1_per_class(
    metrics_dict=metrics_dict,
    class_names=config.CLASS_NAMES,
    save_path=os.path.join(config.PLOTS_DIR, 'f1_per_class.png')
)

print(f"\n✅ All plots saved to: {config.PLOTS_DIR}")

# ============================================================
# SECTION 9 — SAVE EVALUATION REPORT
# ============================================================
print("\n" + "-"*55)
print("  STEP 8: Saving Evaluation Report")
print("-"*55)

save_evaluation_report(
    metrics_dict=metrics_dict,
    class_names=config.CLASS_NAMES,
    save_path=config.EVALUATION_REPORT_PATH
)

# ============================================================
# SECTION 10 — FINAL SUMMARY
# ============================================================
print("\n" + "="*55)
print("  EVALUATION COMPLETE")
print("="*55)
print(f"\n📊 Final Results Summary:")
print(f"   Overall Accuracy  : {metrics_dict['accuracy']*100:.2f}%")
print(f"   Weighted F1-Score : {metrics_dict['weighted_f1']:.4f}")
print(f"   Macro F1-Score    : {metrics_dict['macro_f1']:.4f}")
print(f"   ROC-AUC (macro)   : {metrics_dict['auc_macro']:.4f}")

print(f"\n📁 Files Generated:")
print(f"   outputs/plots/confusion_matrix.png")
print(f"   outputs/plots/roc_curves.png")
print(f"   outputs/plots/training_history.png")
print(f"   outputs/plots/f1_per_class.png")
print(f"   outputs/reports/evaluation_report.txt")

print(f"\n🔜 Next Step:")
print(f"   Download all output files from Colab/Drive to laptop")
print(f"   Then run Session 4 — Streamlit web application")
print("="*55)