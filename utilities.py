import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import os
from skimage import io
from PIL import Image
from tensorflow.keras import backend as K


# ==========================================
# Custom Data Generator
# ==========================================

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, ids, mask, image_dir='./', batch_size=16,
                 img_h=256, img_w=256, shuffle=True):

        self.ids = ids
        self.mask = mask
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch"""

        indexes = self.indexes[
            index * self.batch_size:(index + 1) * self.batch_size
        ]

        list_ids = [self.ids[i] for i in indexes]
        list_mask = [self.mask[i] for i in indexes]

        X, y = self.__data_generation(list_ids, list_mask)

        return X, y

    def on_epoch_end(self):
        """Shuffle indexes after each epoch"""

        self.indexes = np.arange(len(self.ids))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids, list_mask):

        X = np.empty(
            (len(list_ids), self.img_h, self.img_w, 3),
            dtype=np.float32
        )

        y = np.empty(
            (len(list_ids), self.img_h, self.img_w, 1),
            dtype=np.float32
        )

        for i in range(len(list_ids)):

            img_path = './' + str(list_ids[i])
            mask_path = './' + str(list_mask[i])

            img = io.imread(img_path)
            mask = io.imread(mask_path)

            img = cv2.resize(img, (self.img_h, self.img_w))
            mask = cv2.resize(mask, (self.img_h, self.img_w))

            img = img.astype(np.float32)
            mask = mask.astype(np.float32)

            # normalize image
            img -= img.mean()
            img /= (img.std() + 1e-8)

            # normalize mask
            mask = mask / 255.0

            X[i] = img
            y[i] = np.expand_dims(mask, axis=2)

        y = (y > 0).astype(np.float32)

        return X, y


# ==========================================
# Prediction Function
# ==========================================

def prediction(df, model, model_seg):

    image_id = []
    predicted_mask = []
    has_mask = []

    for i in range(len(df)):

        sample = df.iloc[i]["image_path"]
        path = "./" + sample

        img = io.imread(path)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32)

        img -= img.mean()
        img /= (img.std() + 1e-8)

        X = np.expand_dims(img, axis=0)

        # classification
        cls = model.predict(X, verbose=0)
        tumor_class = np.argmax(cls)

        if tumor_class == 0:
            image_id.append(sample)
            predicted_mask.append(
                np.zeros((256, 256), dtype=np.uint8)
            )
            has_mask.append(0)
            continue

        # segmentation
        seg = model_seg.predict(X, verbose=0)[0]
        seg = (seg > 0.5).astype(np.uint8)

        image_id.append(sample)
        predicted_mask.append(seg[:, :, 0])
        has_mask.append(1)

    return image_id, predicted_mask, has_mask


# ==========================================
# Tversky Loss
# ==========================================

def tversky(y_true, y_pred, smooth=1e-6):

    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)

    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)

    alpha = 0.7

    return (true_pos + smooth) / (
        true_pos +
        alpha * false_neg +
        (1 - alpha) * false_pos +
        smooth
    )


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):

    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75

    return K.pow((1 - pt_1), gamma)