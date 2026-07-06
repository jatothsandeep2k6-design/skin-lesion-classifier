# ============================================================
# utils/data_loader.py
# PURPOSE: Load HAM10000 dataset, clean metadata,
#          encode columns, split into train/val/test sets
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Import all settings from config.py
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================
# FUNCTION 1 — find_image_path()
# HAM10000 images are split across two folders (part1 and part2)
# This function searches both folders and returns the correct path
# ============================================================
def find_image_path(image_id, part1_dir, part2_dir):
    """
    Given an image_id (e.g. 'ISIC_0024306'),
    find the full file path in part1 or part2 folder.
    Returns the path if found, None if not found.
    """
    # Build the filename by adding .jpg extension
    filename = image_id + ".jpg"

    # Check in part 1 folder first
    path1 = os.path.join(part1_dir, filename)
    if os.path.exists(path1):
        return path1  # Found in part 1 — return this path

    # If not in part 1, check part 2 folder
    path2 = os.path.join(part2_dir, filename)
    if os.path.exists(path2):
        return path2  # Found in part 2 — return this path

    # Image not found in either folder
    return None


# ============================================================
# FUNCTION 2 — encode_metadata()
# Converts text columns (sex, localization) into numbers
# because neural networks only understand numbers, not text
# ============================================================
def encode_metadata(df, fit_encoder=True, encoder=None):
    """
    Encodes the three metadata columns:
    - age        : divide by 100 to normalize between 0 and 1
    - sex        : male=0, female=1, unknown=0.5
    - localization: text labels converted to integers using LabelEncoder

    fit_encoder=True  → create a NEW encoder (used during training)
    fit_encoder=False → use EXISTING encoder (used during evaluation/app)
    """

    # Make a copy so we don't change the original dataframe
    df = df.copy()

    # --- ENCODE AGE ---
    # Fill missing age values with the mean age of the dataset
    # (some patients in HAM10000 have no age recorded)
    mean_age = df[config.AGE_COLUMN].mean()
    df[config.AGE_COLUMN] = df[config.AGE_COLUMN].fillna(mean_age)

    # Divide age by 100 to get a value between 0 and 1
    # Example: age 45 becomes 0.45
    # Neural networks work better with small numbers (0 to 1 range)
    df[config.AGE_COLUMN] = df[config.AGE_COLUMN] / 100.0

    # --- ENCODE SEX ---
    # Convert text 'male'/'female' to numbers 0/1
    # 'unknown' gets 0.5 (middle value)
    sex_mapping = {
        'male'   : 0.0,
        'female' : 1.0,
        'unknown': 0.5
    }
    # Apply mapping — if value not in mapping, use 0.5 (unknown)
    df[config.SEX_COLUMN] = df[config.SEX_COLUMN].map(sex_mapping).fillna(0.5)

    # --- ENCODE LOCALIZATION ---
    # LabelEncoder converts text categories to integers
    # Example: 'back'=1, 'face'=4, 'scalp'=11
    if fit_encoder:
        # CREATE a new encoder and learn the categories from this data
        encoder = LabelEncoder()
        df[config.LOC_COLUMN] = encoder.fit_transform(
            df[config.LOC_COLUMN].fillna('unknown')
        )

        # Save the encoder to disk so the app can use it later
        # without needing to retrain
        os.makedirs(os.path.dirname(config.ENCODER_SAVE_PATH), exist_ok=True)
        with open(config.ENCODER_SAVE_PATH, 'wb') as f:
            pickle.dump(encoder, f)
        print(f"✅ Localization encoder saved to: {config.ENCODER_SAVE_PATH}")

    else:
        # USE the existing encoder that was saved during training
        if encoder is None:
            raise ValueError("encoder must be provided when fit_encoder=False")

        # Handle any localization values not seen during training
        known_classes = list(encoder.classes_)
        df[config.LOC_COLUMN] = df[config.LOC_COLUMN].fillna('unknown').apply(
            lambda x: x if x in known_classes else 'unknown'
        )
        df[config.LOC_COLUMN] = encoder.transform(df[config.LOC_COLUMN])

    return df, encoder


# ============================================================
# FUNCTION 3 — encode_labels()
# Converts diagnosis text (e.g. 'mel') to integer class index
# Example: 'akiec'=0, 'bcc'=1, 'bkl'=2, 'df'=3,
#          'mel'=4, 'nv'=5, 'vasc'=6
# ============================================================
def encode_labels(df):
    """
    Maps the dx column (diagnosis) to integer class indices
    using the CLASS_NAMES list from config.py
    """
    # Create mapping dictionary from class name to index
    # {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
    label_map = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}

    # Apply mapping to the label column
    df = df.copy()
    df['label'] = df[config.LABEL_COLUMN].map(label_map)

    return df


# ============================================================
# FUNCTION 4 — load_dataset()
# MAIN FUNCTION — loads everything and returns train/val/test splits
# This is the function that train.py will call
# ============================================================
def load_dataset(image_dir_part1=None, image_dir_part2=None,
                 metadata_csv=None, use_colab=False):
    """
    Master function that:
    1. Loads metadata CSV
    2. Finds image paths for all images
    3. Encodes metadata columns
    4. Encodes labels
    5. Splits into train, validation, test sets
    6. Saves test split CSV for evaluate.py

    Returns: train_df, val_df, test_df
    Each DataFrame has columns:
    - image_id, image_path, age, sex, localization, label
    """

    print("\n" + "="*55)
    print("  LOADING HAM10000 DATASET")
    print("="*55)

    # --- SET PATHS ---
    # Use Colab paths if running on Google Colab
    if use_colab:
        part1 = config.COLAB_IMAGE_PART1
        part2 = config.COLAB_IMAGE_PART2
        csv_path = config.COLAB_METADATA_CSV
        print("📍 Using Google Colab paths")
    else:
        # Use local paths (or override with function arguments)
        part1 = image_dir_part1 or config.IMAGE_DIR_PART1
        part2 = image_dir_part2 or config.IMAGE_DIR_PART2
        csv_path = metadata_csv or config.METADATA_CSV
        print("📍 Using local paths")

    # --- STEP 1: LOAD METADATA CSV ---
    print(f"\n📂 Loading metadata from:\n   {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Metadata loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")

    # --- STEP 2: FIND IMAGE PATHS ---
    print(f"\n🔍 Finding image file paths...")
    print(f"   Searching in part 1: {part1}")
    print(f"   Searching in part 2: {part2}")

    # For each image_id in metadata, find where the file is
    df['image_path'] = df[config.IMAGE_ID_COLUMN].apply(
        lambda img_id: find_image_path(img_id, part1, part2)
    )

    # Count how many images were found
    found = df['image_path'].notna().sum()
    missing = df['image_path'].isna().sum()
    print(f"✅ Images found    : {found}")
    if missing > 0:
        print(f"⚠️  Images missing  : {missing} (these rows will be dropped)")

    # Remove rows where image file was not found
    df = df.dropna(subset=['image_path'])
    print(f"✅ Dataset size after cleanup: {len(df)} images")

    # --- STEP 3: SHOW CLASS DISTRIBUTION ---
    print(f"\n📊 Class distribution (before split):")
    class_counts = df[config.LABEL_COLUMN].value_counts()
    for cls, count in class_counts.items():
        bar = "█" * (count // 100)
        print(f"   {cls:6s}: {count:5d} images  {bar}")

    # --- STEP 4: ENCODE LABELS ---
    print(f"\n🏷️  Encoding diagnosis labels...")
    df = encode_labels(df)
    print(f"✅ Labels encoded: {dict(zip(config.CLASS_NAMES, range(config.NUM_CLASSES)))}")

    # --- STEP 5: ENCODE METADATA ---
    print(f"\n🔢 Encoding metadata columns (age, sex, localization)...")
    df, encoder = encode_metadata(df, fit_encoder=True)
    print(f"✅ Metadata encoded successfully")

    # --- STEP 6: SPLIT INTO TRAIN / VAL / TEST ---
    print(f"\n✂️  Splitting dataset...")
    print(f"   Train: {int(config.TRAIN_RATIO*100)}%  |  "
          f"Val: {int(config.VAL_RATIO*100)}%  |  "
          f"Test: {int(config.TEST_RATIO*100)}%")

    # First split: separate test set (15%)
    # stratify=df['label'] ensures all 7 classes appear in test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=config.TEST_RATIO,
        random_state=config.RANDOM_SEED,
        stratify=df['label']
    )

    # Second split: separate validation from training
    # val_size relative to train_val (15/85 = 0.176)
    val_relative_size = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=config.RANDOM_SEED,
        stratify=train_val_df['label']
    )

    print(f"✅ Train set : {len(train_df)} images")
    print(f"✅ Val set   : {len(val_df)} images")
    print(f"✅ Test set  : {len(test_df)} images")

    # --- STEP 7: SAVE TEST SPLIT CSV ---
    # evaluate.py needs to know which images are in the test set
    os.makedirs(os.path.dirname(config.TEST_SPLIT_PATH), exist_ok=True)
    test_df[['image_id', 'image_path',
             'age', 'sex', 'localization', 'label']].to_csv(
        config.TEST_SPLIT_PATH, index=False
    )
    print(f"\n💾 Test split saved to: {config.TEST_SPLIT_PATH}")
    print(f"   (evaluate.py will use this file)")

    print("\n" + "="*55)
    print("  DATASET LOADING COMPLETE ✅")
    print("="*55 + "\n")

    return train_df, val_df, test_df


# ============================================================
# QUICK TEST — runs only when you execute this file directly
# python utils/data_loader.py
# ============================================================
if __name__ == "__main__":
    print("Testing data_loader.py...")
    print("Note: Dataset must be downloaded first.")
    print("If dataset not downloaded yet, this will show path errors.")
    print("That is OK for now — run this again after downloading dataset.")

    # Try loading — will show helpful error if dataset not found
    try:
        train_df, val_df, test_df = load_dataset()
        print("\n✅ data_loader.py is working correctly!")
        print(f"   Sample train row:\n{train_df.iloc[0]}")
    except FileNotFoundError as e:
        print(f"\n⚠️  Dataset not found yet: {e}")
        print("   Download HAM10000 first (Step 8), then test again.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Send this error to your mentor for help.")