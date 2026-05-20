# FaceWell AI — Facial Stress Detection System

A deep learning application that detects mental stress levels from facial expressions in real time.

## 🔗 Live Demo

**Try it here → [https://facewell-ai.lovable.app](https://facewell-ai.lovable.app)**

Upload a face image or use your webcam and get your stress score instantly.


## Overview

FaceWell AI uses a custom Convolutional Neural Network (CNN) trained on the FER-2013 dataset to classify facial expressions into 7 emotion categories and map them to a meaningful stress score between 0 and 100.

## How It Works

1. User uploads or captures a face image via the frontend
2. Image is sent to the FastAPI backend
3. Backend preprocesses the image and runs it through the CNN model
4. Model predicts the emotion with confidence score
5. Stress score is computed using psychology based emotion weights
6. Results are returned to the frontend as a JSON response

## Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | 66.12% |
| Dataset | FER-2013 |
| Architecture | Custom CNN (4 Conv blocks) |
| Total Parameters | 7,187,911 |

## Emotion to Stress Mapping

| Emotion | Stress Weight |
|---|---|
| Fear | 0.95 |
| Angry | 0.90 |
| Disgust | 0.70 |
| Sad | 0.60 |
| Surprise | 0.40 |
| Neutral | 0.20 |
| Happy | 0.05 |

## Stress Levels

| Score | Level |
|---|---|
| 0 - 25 | Low |
| 25 - 50 | Moderate |
| 50 - 75 | High |
| 75 - 100 | Severe |

**GET /**
Health check — confirms the API is running

**POST /predict**
Accepts a face image and returns:
```json
{
  "predicted_emotion": "happy",
  "confidence": 48.5,
  "stress_score": 18.78,
  "stress_level": "Low",
  "emotion_breakdown": {
    "angry": 4.02,
    "disgust": 0.0,
    "fear": 1.36,
    "happy": 48.5,
    "neutral": 40.43,
    "sad": 5.4,
    "surprise": 0.28
  },
  "message": "Detected happy with 48.5% confidence"
}
```

## Tech Stack

**Model**
- Python
- TensorFlow / Keras
- NumPy, OpenCV, PIL
- Scikit-learn
- Trained on Kaggle (Tesla T4 GPU)

**Backend**
- FastAPI
- Uvicorn
- Deployed on Render

**Frontend**
- Built with Lovable AI
- React
- Live at [https://facewell-ai.lovable.app](https://facewell-ai.lovable.app)

## Model

The trained CNN model is publicly available on Hugging Face:

**→ [https://huggingface.co/barth019/facewell_model](https://huggingface.co/barth019/facewell_model/tree/main)**

You can download and use the model directly for research or integration into your own applications.

## API

**Base URL**

https://facewell-ai.onrender.com

## Dataset

- **Name** — FER-2013
- **Size** — 35,887 images
- **Classes** — 7 emotions
- **Image size** — 48×48 pixels grayscale
- **Source** — Kaggle

## Author

- Terna Bartholomew