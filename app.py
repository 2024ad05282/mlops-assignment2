"""
app.py - FastAPI Inference Service for Cat vs Dog CNN
=====================================================
MLOps Assignment 2 - Model Packaging & Containerization
REST API with health check and prediction endpoints.
"""

import os
import io
import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("models", "cat_dog_cnn.pt"))
IMAGE_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model Definition (must match train.py) ──────────────────────────────────

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

    print(f"Model loaded from '{MODEL_PATH}'")
    print(f"Device: {DEVICE}")
    print(f"Classes: {class_names}")
    if val_acc:
        print(f"Validation Accuracy: {val_acc:.2f}%")

    return model, class_names


# ─── Initialize App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Cat vs Dog Classifier API",
    description="REST API for Cat vs Dog image classification using a CNN model.",
    version="1.0.0",
)

# Load model at startup
model = None
class_names = None
model_load_time = None


@app.on_event("startup")
async def startup_event():
    """Load model when the server starts."""
    global model, class_names, model_load_time
    start = time.time()
    model, class_names = load_model()
    model_load_time = time.time() - start
    print(f"Model loaded in {model_load_time:.2f}s")


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns service status, model info, and device being used.
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
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint.
    Accepts an image file and returns the predicted class label
    along with class probabilities.

    - **file**: Image file (JPEG/PNG) of a cat or dog.
    - **Returns**: Predicted label, confidence, and per-class probabilities.
    """
    # Validate model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload a JPEG or PNG image.",
        )

    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

        # Run inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        inference_time = time.time() - start_time

        # Extract results
        probs = probabilities[0].cpu().numpy()
        predicted_idx = int(probs.argmax())
        predicted_label = class_names[predicted_idx]
        confidence = float(probs[predicted_idx])

        # Build per-class probability dict
        class_probabilities = {
            class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))
        }

        return {
            "prediction": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": class_probabilities,
            "inference_time_seconds": round(inference_time, 4),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ─── Run Server ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
