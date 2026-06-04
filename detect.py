"""
Face Emotion Detection — Real-Time Inference
=============================================
Loads a trained model and runs live emotion detection from webcam.

For each frame:
  1. Convert to grayscale
  2. Detect faces with OpenCV Haar Cascade
  3. Crop and resize each face to 48x48
  4. Predict emotion with the CNN
  5. Overlay the label on the frame

Controls:
    Press  q  to quit

Usage:
    python detect.py --model model.h5
    python detect.py --model model.h5 --camera 0
"""

import argparse
import numpy as np
import cv2
import tensorflow as tf


# ── Constants ─────────────────────────────────────────────────────────────────
EMOTIONS  = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
IMG_SIZE  = 48
DISPLAY_W = 900
DISPLAY_H = 600


# ── Face detection ────────────────────────────────────────────────────────────
def detect_faces(frame):
    """Detect faces in a BGR frame using Haar Cascade.

    Args:
        frame: BGR image (numpy array)

    Returns:
        faces: list of (x, y, w, h) bounding boxes
        gray:  grayscale version of the frame
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.32, minNeighbors=5)
    return faces, gray


# ── Inference loop ────────────────────────────────────────────────────────────
def run(model_path, camera_index=0):
    """Run real-time emotion detection.

    Args:
        model_path:   Path to the trained .h5 model file
        camera_index: OpenCV camera index (default 0)
    """
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    print("Running — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        faces, gray = detect_faces(frame)

        for (x, y, w, h) in faces:
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

            # Preprocess face ROI
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))   # (1, 48, 48, 1)

            # Predict
            predictions   = model.predict(roi, verbose=0)
            emotion_label = EMOTIONS[np.argmax(predictions[0])]
            confidence    = predictions[0].max()

            # Overlay label
            label = f"{emotion_label} ({confidence:.0%})"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        cv2.imshow('Face Emotion Detection  —  press q to quit', display)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time face emotion detection')
    parser.add_argument('--model',  required=True,  help='Path to trained model (.h5)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    args = parser.parse_args()
    run(args.model, args.camera)
