# FaceWell AI — FastAPI Backend
# Serves stress detection predictions via REST API

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
import uvicorn

# Initialise app 
app = FastAPI(
    title       = "FaceWell AI API",
    description = "Stress detection from facial expressions",
    version     = "1.0.0"
)

# CORS — allows frontend to communicate 
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Download and load model from Hugging Face on startup 
print("Downloading FaceWell model from Hugging Face...")

model_path = hf_hub_download(
    repo_id   = "barth019/facewell_model",
    filename  = "facewell_stress_model.keras"
)

print(f"✓ Model downloaded to : {model_path}")

model = tf.keras.models.load_model(model_path, compile=False)
model.compile(
    optimizer = 'adam',
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)
print("✓ Model loaded successfully")

# Load config
with open('facewell_config.json', 'r') as f:
    config = json.load(f)

CLASS_NAMES    = config['class_names']
STRESS_WEIGHTS = config['stress_weights']
print("✓ Config loaded successfully")

# Helper functions 
def preprocess_image(image_bytes):
    """
    Takes raw image bytes from the frontend,
    preprocesses them to match our training format.
    Returns numpy array ready for model prediction.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert('L')           # convert to grayscale
    img = img.resize((48, 48))       # resize to 48x48
    img = np.array(img) / 255.0     # normalise to 0-1
    img = img.reshape(1, 48, 48, 1) # add batch and channel dims
    return img

def compute_stress_score(probabilities):
    """Computes weighted stress score from emotion probabilities."""
    weights = np.array([STRESS_WEIGHTS[cls] for cls in CLASS_NAMES])
    score   = np.dot(probabilities, weights) * 100
    return round(float(score), 2)

def get_stress_level(score):
    """Returns stress level label from numerical score."""
    if score < 25:
        return "Low"
    elif score < 50:
        return "Moderate"
    elif score < 75:
        return "High"
    else:
        return "Severe"

# API endpoints 
@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status"   : "FaceWell AI is running",
        "version"  : "1.0.0",
        "accuracy" : f"{config['model_accuracy']}%"
    }

@app.post("/predict")
async def predict_stress(file: UploadFile = File(...)):
    """
    Main prediction endpoint.
    Accepts an image file and returns stress analysis.
    """
    try:
        # Read image bytes from request
        image_bytes = await file.read()

        # Preprocess image
        processed_img = preprocess_image(image_bytes)

        # Run model prediction
        predictions   = model.predict(processed_img, verbose=0)
        probabilities = predictions[0]

        # Get predicted emotion
        predicted_idx     = int(np.argmax(probabilities))
        predicted_emotion = CLASS_NAMES[predicted_idx]
        confidence        = round(float(probabilities[predicted_idx]) * 100, 2)

        # Compute stress score
        stress_score = compute_stress_score(probabilities)
        stress_level = get_stress_level(stress_score)

        # Build emotion probability breakdown
        emotion_breakdown = {
            cls: round(float(prob) * 100, 2)
            for cls, prob in zip(CLASS_NAMES, probabilities)
        }

        # Return response
        return JSONResponse(content={
            "success"          : True,
            "predicted_emotion": predicted_emotion,
            "confidence"       : confidence,
            "stress_score"     : stress_score,
            "stress_level"     : stress_level,
            "emotion_breakdown": emotion_breakdown,
            "message"          : f"Detected {predicted_emotion} with {confidence}% confidence"
        })

    except Exception as e:
        return JSONResponse(
            status_code = 500,
            content     = {
                "success" : False,
                "error"   : str(e)
            }
        )

# Run server 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)