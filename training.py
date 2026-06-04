"""
Face Emotion Detection — Training Script
=========================================
Trains a CNN on the FER-2013 dataset to classify 7 facial emotions:
    angry, disgust, fear, happy, sad, surprise, neutral

Dataset: FER-2013 (icml_face_data.csv)
    Download from: https://www.kaggle.com/datasets/deadskull7/fer2013

Architecture:
    Conv2D(64) -> BN -> Dropout
    Conv2D(128) -> BN -> Dropout
    Conv2D(256) -> BN -> MaxPool -> Dropout
    Flatten -> Dense(128) -> BN -> Dropout
    Dense(256) -> BN -> Dropout
    Dense(7, softmax)

Output: model.h5  (saved to current directory)

Usage:
    python training.py --data path/to/icml_face_data.csv
"""

import argparse
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout,
                          Flatten, BatchNormalization)
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE   = 48
NUM_LABELS = 7
BATCH_SIZE = 64
EPOCHS     = 200
EMOTIONS   = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


# ── Data loading ──────────────────────────────────────────────────────────────
def load_fer2013(csv_path):
    """Load and preprocess the FER-2013 CSV dataset.

    Args:
        csv_path: Path to icml_face_data.csv

    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    """
    df = pd.read_csv(csv_path)

    X_train, y_train = [], []
    X_test,  y_test  = [], []

    for _, row in df.iterrows():
        pixels = np.array(row[' pixels'].split(), dtype='float32')
        if 'Training' in row[' Usage']:
            X_train.append(pixels)
            y_train.append(row['emotion'])
        elif 'PublicTest' in row[' Usage']:
            X_test.append(pixels)
            y_test.append(row['emotion'])

    X_train = np.array(X_train, 'float32')
    X_test  = np.array(X_test,  'float32')

    # Normalise
    X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-7)
    X_test  = (X_test  - X_test.mean(axis=0))  / (X_test.std(axis=0)  + 1e-7)

    # Reshape to (N, 48, 48, 1)
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test  = X_test.reshape(-1,  IMG_SIZE, IMG_SIZE, 1)

    y_train = to_categorical(np.array(y_train, 'float32'), NUM_LABELS)
    y_test  = to_categorical(np.array(y_test,  'float32'), NUM_LABELS)

    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    """Build and compile the CNN emotion classifier."""
    model = Sequential([
        Conv2D(64,  (1, 1), padding='same', activation='relu',
               input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        BatchNormalization(),
        Dropout(0.25),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Dropout(0.25),

        Conv2D(256, (5, 5), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),

        Dense(NUM_LABELS, activation='softmax'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train(csv_path, output_path='model.h5'):
    X_train, y_train, X_test, y_test = load_fer2013(csv_path)
    model = build_model()

    callbacks = [EarlyStopping(patience=7, restore_best_weights=True)]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
    )

    model.save(output_path)
    print(f"\nModel saved to: {output_path}")

    val_acc = max(history.history['val_accuracy'])
    print(f"Best validation accuracy: {val_acc:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train face emotion CNN on FER-2013')
    parser.add_argument('--data',   required=True,         help='Path to icml_face_data.csv')
    parser.add_argument('--output', default='model.h5',    help='Output model path (default: model.h5)')
    args = parser.parse_args()
    train(args.data, args.output)
