# Face Emotion Detection

Real-time facial emotion detection from a live webcam feed. A CNN is trained on the
[FER-2013](https://www.kaggle.com/datasets/deadskull7/fer2013) dataset and detects
**7 emotions** — angry, disgust, fear, happy, sad, surprise, neutral — in real time
using OpenCV face detection.

---

## Pipeline

```
FER-2013 dataset (CSV)
        ↓
Data loading & normalisation
        ↓
CNN training  (Conv2D × 3 → Dense × 2 → Softmax)
        ↓
model.h5
        ↓
Live webcam feed
        ↓
OpenCV Haar Cascade face detection (per frame)
        ↓
Crop → resize 48×48 → CNN predict
        ↓
Emotion label + confidence overlaid on frame
```

---

## Model Architecture

| Layer | Details |
|-------|---------|
| Conv2D(64, 1×1) + BN + Dropout | Feature extraction |
| Conv2D(128, 3×3) + BN + Dropout | |
| Conv2D(256, 5×5) + BN + MaxPool + Dropout | |
| Dense(128) + BN + Dropout | Classification head |
| Dense(256) + BN + Dropout | |
| Dense(7, softmax) | 7 emotion classes |

- **Optimiser:** Adam (lr=0.0001)
- **Loss:** Categorical cross-entropy
- **Early stopping:** patience=7

---

## Setup

```bash
git clone https://github.com/bazzal99/Face-Emotion-Detection
cd Face-Emotion-Detection
pip install -r requirements.txt
```

---

## Usage

### 1. Train the model

Download the FER-2013 dataset (`icml_face_data.csv`) from
[Kaggle](https://www.kaggle.com/datasets/deadskull7/fer2013), then:

```bash
python training.py --data path/to/icml_face_data.csv
# Saves model.h5 in the current directory
```

### 2. Run real-time detection

```bash
python detect.py --model model.h5
# Press  q  to quit
```

Optional: specify a different camera:
```bash
python detect.py --model model.h5 --camera 1
```

---

## Files

| File | Description |
|------|-------------|
| `training.py` | CNN training on FER-2013 — loads data, builds model, trains, saves `.h5` |
| `detect.py` | Real-time webcam inference — loads model, detects faces, overlays emotion labels |
| `requirements.txt` | Python dependencies |

> **Note:** The trained model file (`model.h5`) is not included in the repository
> due to file size. Train it yourself using `training.py` or download a pre-trained
> FER-2013 model and rename it to `model.h5`.

---

## Emotion Classes

`angry` · `disgust` · `fear` · `happy` · `sad` · `surprise` · `neutral`

---

## License

MIT License. See [LICENSE](LICENSE).
