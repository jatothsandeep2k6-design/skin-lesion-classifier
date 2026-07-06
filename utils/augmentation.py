# ============================================================
# utils/augmentation.py
# FIXED FOR EFFICIENTNETB0
# ============================================================

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from tensorflow.keras.applications.efficientnet import preprocess_input


def load_and_preprocess_image(image_path, augment=False):
    """
    Load image and preprocess for EfficientNetB0
    """

    img = Image.open(image_path).convert("RGB")
    img = img.resize(config.IMAGE_SIZE)

    # DON'T divide by 255
    img_array = np.array(img, dtype=np.float32)

    if augment:
        img_array = apply_augmentation(img_array)

    # EfficientNet preprocessing
    img_array = preprocess_input(img_array)

    return img_array


def apply_augmentation(img_array):

    img_tensor = tf.convert_to_tensor(img_array)

    img_tensor = tf.expand_dims(img_tensor, 0)

    img_tensor = tf.image.random_flip_left_right(img_tensor)

    img_tensor = tf.image.random_flip_up_down(img_tensor)

    img_tensor = tf.image.random_brightness(
        img_tensor,
        max_delta=25.0
    )

    img_tensor = tf.image.random_contrast(
        img_tensor,
        lower=0.8,
        upper=1.2
    )

    img_tensor = tf.squeeze(img_tensor, axis=0)

    img_tensor = tf.clip_by_value(
        img_tensor,
        0.0,
        255.0
    )

    return img_tensor.numpy()


class SkinLesionDataGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        dataframe,
        batch_size=32,
        augment=False,
        shuffle=True
    ):

        self.df = dataframe.reset_index(drop=True)

        self.batch_size = batch_size

        self.augment = augment

        self.shuffle = shuffle

        self.num_classes = config.NUM_CLASSES

        self.on_epoch_end()

        print(
            f"Generator: {len(self.df)} images | "
            f"batch={batch_size} | augment={augment}"
        )

    def __len__(self):

        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):

        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.df))

        batch_df = self.df.iloc[start:end]

        image_batch = np.zeros(
            (
                len(batch_df),
                config.INPUT_SHAPE[0],
                config.INPUT_SHAPE[1],
                config.INPUT_SHAPE[2]
            ),
            dtype=np.float32
        )

        metadata_batch = np.zeros(
            (
                len(batch_df),
                3
            ),
            dtype=np.float32
        )

        labels = np.zeros(
            (
                len(batch_df),
                config.NUM_CLASSES
            ),
            dtype=np.float32
        )

        for i, (_, row) in enumerate(batch_df.iterrows()):

            image_batch[i] = load_and_preprocess_image(
                row["image_path"],
                augment=self.augment
            )

            metadata_batch[i] = np.array([
                row[config.AGE_COLUMN],
                row[config.SEX_COLUMN],
                row[config.LOC_COLUMN]
            ], dtype=np.float32)

            labels[i] = tf.keras.utils.to_categorical(
                int(row["label"]),
                config.NUM_CLASSES
            )

        return (image_batch, metadata_batch), labels

    def on_epoch_end(self):

        if self.shuffle:
            self.df = self.df.sample(
                frac=1,
                random_state=np.random.randint(100000)
            ).reset_index(drop=True)