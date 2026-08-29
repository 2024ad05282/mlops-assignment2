"""
app.py - FastAPI Inference Service with Monitoring & Logging
===========================================================
MLOps Assignment 2 - Model Packaging, Monitoring, and CD
REST API with health check, prediction, feedback, metrics, and performance tracking.
"""

import os
import io
import time
import uuid
import json
import logging
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "cat_dog_cnn.pt"))
IMAGE_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

PREDICTIONS_LOG_PATH = os.path.join(LOGS_DIR, "predictions.jsonl")
FEEDBACK_LOG_PATH = os.path.join(LOGS_DIR, "feedback.jsonl")

# ─── Logging Setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "app.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("inference-service")

# ─── Model Definition ────────────────────────────────────────────────────────

class CatDogCNN(nn.Module):
    """
    Simple baseline CNN for binary classification (Cat vs Dog).
    Architecture: 3 conv blocks + 2 FC layers.
    """

    def __init__(self, image_size=128):
        super(CatDogCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * (image_size // 8) * (image_size // 8), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ─── Image Preprocessing ────────────────────────────────────────────────────

inference_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Load Model ─────────────────────────────────────────────────────────────

def load_model():
    """Load the trained model from checkpoint."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    image_size = checkpoint.get("image_size", IMAGE_SIZE)
    model = CatDogCNN(image_size=image_size).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = checkpoint.get("class_names", ["Cat", "Dog"])
    val_acc = checkpoint.get("val_acc", None)

    logger.info(f"Model loaded successfully from '{MODEL_PATH}'")
    logger.info(f"Using device: {DEVICE}")
    if val_acc:
        logger.info(f"Checkpoint Validation Accuracy: {val_acc:.2f}%")

    return model, class_names


# ─── Pydantic Models for Schema Validation ──────────────────────────────────

class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="The UUID request ID from prediction response.")
    true_label: str = Field(..., description="The ground truth label (Cat or Dog).")


# ─── Initialize App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Cat vs Dog MLOps Inference Service",
    description="Containerized classifier CAT DOG service with logging, metrics, and drift monitoring.",
    version="1.1.0",
)

# Global variables
model = None
class_names = None
model_load_time = None

# Simple In-Memory Metrics Store
metrics_store = {
    "total_requests": 0,
    "predictions": {"Cat": 0, "Dog": 0},
    "status_codes": {},
    "total_latency_seconds": 0.0,
}


@app.on_event("startup")
async def startup_event():
    """Load model at startup."""
    global model, class_names, model_load_time
    start = time.time()
    model, class_names = load_model()
    model_load_time = time.time() - start


# ─── Middleware for Request Logging & Metrics ───────────────────────────────

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """
    Excludes metrics/health from logging, captures request details & latency,
    and updates request counters.
    """
    path = request.url.path
    if path in ["/health", "/metrics", "/performance"]:
        return await call_next(request)

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    # Update in-memory metrics
    status_code = response.status_code
    metrics_store["total_requests"] += 1
    metrics_store["total_latency_seconds"] += duration
    metrics_store["status_codes"][status_code] = metrics_store["status_codes"].get(status_code, 0) + 1

    # Log request summary
    logger.info(
        f"HTTP {request.method} {path} | Status: {status_code} | "
        f"Client: {request.client.host if request.client else 'Unknown'} | Latency: {duration:.4f}s"
    )

    return response


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "model": {
            "path": MODEL_PATH,
            "loaded": model is not None,
            "device": str(DEVICE),
            "classes": class_names,
            "load_time_seconds": round(model_load_time, 3) if model_load_time else None,
        },
        "gpu_available": torch.cuda.is_available(),
    }


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint.
    Accepts image file, returns prediction metadata, and logs prediction to local file.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload JPEG or PNG image.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        inference_time = time.time() - start_time

        # Extract predictions
        probs = probabilities[0].cpu().numpy()
        predicted_idx = int(probs.argmax())
        predicted_label = class_names[predicted_idx]
        confidence = float(probs[predicted_idx])

        # Track request predictions
        metrics_store["predictions"][predicted_label] += 1

        request_id = str(uuid.uuid4())
        class_probabilities = {
            class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))
        }

        # Log prediction to file for performance tracking / feedback loop
        prediction_record = {
            "request_id": request_id,
            "timestamp": time.time(),
            "predicted_label": predicted_label,
            "confidence": confidence,
            "filename": file.filename,
            "filesize_bytes": len(contents),
            "inference_time_seconds": inference_time,
        }
        with open(PREDICTIONS_LOG_PATH, "a") as f:
            f.write(json.dumps(prediction_record) + "\n")

        return {
            "request_id": request_id,
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": class_probabilities,
            "inference_time_seconds": round(inference_time, 4),
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/feedback", tags=["Monitoring & Performance"])
async def submit_feedback(feedback: FeedbackRequest):
    """
    Feedback loop endpoint.
    Accepts true labels for past predictions to track real-world performance.
    """
    # Validate true label value
    if feedback.true_label not in ["Cat", "Dog"]:
        raise HTTPException(status_code=400, detail="true_label must be 'Cat' or 'Dog'.")

    # Verify if request_id exists in predictions.jsonl
    prediction_found = False
    if os.path.exists(PREDICTIONS_LOG_PATH):
        with open(PREDICTIONS_LOG_PATH, "r") as f:
            for line in f:
                record = json.loads(line)
                if record.get("request_id") == feedback.request_id:
                    prediction_found = True
                    break

    if not prediction_found:
        raise HTTPException(
            status_code=404,
            detail=f"request_id '{feedback.request_id}' not found in prediction logs."
        )

    # Save feedback record
    feedback_record = {
        "request_id": feedback.request_id,
        "true_label": feedback.true_label,
        "timestamp": time.time()
    }
    with open(FEEDBACK_LOG_PATH, "a") as f:
        f.write(json.dumps(feedback_record) + "\n")

    logger.info(f"Feedback logged for request {feedback.request_id} -> True Label: {feedback.true_label}")
    return {"status": "success", "message": "Feedback submitted successfully."}


@app.get("/metrics", tags=["Monitoring & Performance"])
async def get_metrics():
    """
    Exposes real-time in-memory monitoring metrics.
    """
    reqs = metrics_store["total_requests"]
    avg_latency = (
        metrics_store["total_latency_seconds"] / reqs if reqs > 0 else 0.0
    )

    return {
        "service": {
            "uptime_model_load_time_seconds": round(model_load_time, 4) if model_load_time else None,
            "device": str(DEVICE),
        },
        "monitoring": {
            "total_requests": reqs,
            "average_latency_seconds": round(avg_latency, 4),
            "status_codes": metrics_store["status_codes"],
            "prediction_counts": metrics_store["predictions"],
        }
    }


@app.get("/performance", tags=["Monitoring & Performance"])
async def get_performance():
    """
    Calculates post-deployment performance metrics (Accuracy) by merging
    the prediction logs with user feedback.
    """
    if not os.path.exists(FEEDBACK_LOG_PATH) or not os.path.exists(PREDICTIONS_LOG_PATH):
        return {
            "status": "insufficient_data",
            "message": "No feedback/prediction history found. Submit predictions and feedback first.",
            "metrics": None
        }

    # Load predictions
    predictions = {}
    with open(PREDICTIONS_LOG_PATH, "r") as f:
        for line in f:
            record = json.loads(line)
            predictions[record["request_id"]] = record["predicted_label"]

    # Load feedback and calculate metrics
    correct = 0
    total = 0
    feedback_collected = []

    with open(FEEDBACK_LOG_PATH, "r") as f:
        for line in f:
            record = json.loads(line)
            req_id = record["request_id"]
            true_lbl = record["true_label"]

            if req_id in predictions:
                pred_lbl = predictions[req_id]
                total += 1
                is_correct = pred_lbl == true_lbl
                if is_correct:
                    correct += 1
                feedback_collected.append({
                    "request_id": req_id,
                    "prediction": pred_lbl,
                    "true_label": true_lbl,
                    "correct": is_correct
                })

    if total == 0:
        return {
            "status": "insufficient_data",
            "message": "No matching predictions found for the submitted feedback.",
            "metrics": None
        }

    accuracy = correct / total
    return {
        "status": "success",
        "sample_count": total,
        "metrics": {
            "post_deployment_accuracy": round(accuracy, 4),
            "correct_predictions": correct,
            "incorrect_predictions": total - correct,
        },
        "samples": feedback_collected
    }


# ─── Run Server ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
