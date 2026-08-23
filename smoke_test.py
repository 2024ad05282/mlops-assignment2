"""
smoke_test.py - Post-deployment Smoke Tests for Cat vs Dog Classifier
=====================================================================
MLOps Assignment 2 - CD Pipeline Smoke Tests
Verifies health check and prediction endpoints on a running container.
Exits with 0 on success, 1 on failure.
"""

import os
import sys
import time
import requests
from PIL import Image
import io

# Get base URL from environment or default to localhost
BASE_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
HEALTH_URL = f"{BASE_URL}/health"
PREDICT_URL = f"{BASE_URL}/predict"

MAX_RETRIES = 5
RETRY_DELAY = 5


def wait_for_service():
    """Wait for the service to become active with retries."""
    print(f"[*] Checking connection to service at {HEALTH_URL}...")
    for i in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(HEALTH_URL, timeout=5)
            if response.status_code == 200:
                print(f"[+] Service is up and responding (Attempt {i}/{MAX_RETRIES})")
                return True
        except requests.exceptions.RequestException:
            pass
        print(f"[-] Service not ready. Retrying in {RETRY_DELAY}s... ({i}/{MAX_RETRIES})")
        time.sleep(RETRY_DELAY)
    return False


def test_health_endpoint():
    """Verify GET /health returns standard healthy payload."""
    print("\n[*] Testing GET /health...")
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data.get("status") == "healthy", f"Expected status 'healthy', got '{data.get('status')}'"
        assert "model" in data, "Response missing 'model' metadata"
        assert data["model"].get("loaded") is True, "Model is not loaded"
        
        print("[+] Health check passed!")
        print(f"    - Status: {data['status']}")
        print(f"    - Device: {data['model']['device']}")
        print(f"    - Classes: {data['model']['classes']}")
        return True
    except Exception as e:
        print(f"[x] Health check failed: {str(e)}")
        return False


def test_predict_endpoint():
    """Verify POST /predict returns valid class predictions for an uploaded image."""
    print("\n[*] Testing POST /predict...")
    try:
        # Generate dummy 128x128 RGB image dynamically
        img = Image.new("RGB", (128, 128), color=(255, 0, 0))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

        # Send post request
        files = {"file": ("test_cat.jpg", img_bytes, "image/jpeg")}
        response = requests.post(PREDICT_URL, files=files, timeout=10)
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "prediction" in data, "Response missing 'prediction'"
        assert "confidence" in data, "Response missing 'confidence'"
        assert "probabilities" in data, "Response missing 'probabilities'"
        
        # Verify classes
        prediction = data["prediction"]
        confidence = data["confidence"]
        probs = data["probabilities"]
        
        assert prediction in ["Cat", "Dog"], f"Invalid class predicted: {prediction}"
        assert 0.0 <= confidence <= 1.0, f"Confidence not in range [0, 1]: {confidence}"
        assert "Cat" in probs and "Dog" in probs, "Probabilities missing class keys"
        
        print("[+] Prediction endpoint passed!")
        print(f"    - Predicted: {prediction}")
        print(f"    - Confidence: {confidence:.4f}")
        print(f"    - Probabilities: {probs}")
        return True
    except Exception as e:
        print(f"[x] Prediction test failed: {str(e)}")
        return False


def main():
    print("=" * 60)
    print("Post-Deployment Smoke Tests")
    print("=" * 60)

    # 1. Wait for container to be ready
    if not wait_for_service():
        print("[x] Error: Service failed to start within time limit.")
        sys.exit(1)

    # 2. Run tests
    health_ok = test_health_endpoint()
    predict_ok = test_predict_endpoint()

    print("=" * 60)
    if health_ok and predict_ok:
        print("[+] All smoke tests PASSED successfully!")
        sys.exit(0)
    else:
        print("[x] Some smoke tests FAILED. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
