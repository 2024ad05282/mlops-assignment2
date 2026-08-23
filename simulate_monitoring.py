"""
simulate_monitoring.py - Post-deployment Performance and Metrics Simulation
===========================================================================
MLOps Assignment 2 - Monitoring & Logging
Simulates requests on a running server, submits feedback, and outputs metrics.
"""

import os
import sys
import random
import requests
import json

BASE_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
HEALTH_URL = f"{BASE_URL}/health"
PREDICT_URL = f"{BASE_URL}/predict"
FEEDBACK_URL = f"{BASE_URL}/feedback"
METRICS_URL = f"{BASE_URL}/metrics"
PERFORMANCE_URL = f"{BASE_URL}/performance"


def check_server():
    """Verify that the service is running."""
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def get_sample_images(base_dir, count=5):
    """Retrieve list of actual cat and dog files from dataset."""
    cat_dir = os.path.join(base_dir, "Cat")
    dog_dir = os.path.join(base_dir, "Dog")
    
    if not os.path.exists(cat_dir) or not os.path.exists(dog_dir):
        print(f"[x] Error: Dataset folders not found at '{base_dir}'")
        return [], []
        
    cats = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if f.endswith(".jpg")]
    dogs = [os.path.join(dog_dir, f) for f in os.listdir(dog_dir) if f.endswith(".jpg")]
    
    return random.sample(cats, count), random.sample(dogs, count)


def main():
    print("=" * 60)
    print("Post-Deployment Monitoring & Performance Simulation")
    print("=" * 60)

    # 1. Check server status
    if not check_server():
        print(f"[x] Error: API server is not running at {BASE_URL}.")
        print("    Please start the server first (e.g. 'python app.py').")
        sys.exit(1)
    print("[+] Connected to API server.")

    # 2. Get samples
    dataset_dir = os.path.join("data", "PetImages")
    print(f"[*] Selecting test images from '{dataset_dir}'...")
    cats, dogs = get_sample_images(dataset_dir, count=5)
    
    if not cats or not dogs:
        print("[x] Failed to retrieve sample images.")
        sys.exit(1)

    all_samples = []
    for c in cats:
        all_samples.append((c, "Cat"))
    for d in dogs:
        all_samples.append((d, "Dog"))
        
    # Shuffle requests
    random.shuffle(all_samples)
    print(f"[+] Loaded {len(all_samples)} sample requests.")

    # 3. Simulate requests & submit feedback
    print("\n[*] Sending prediction requests and submitting ground truth feedback...")
    print("-" * 60)
    
    for idx, (img_path, true_label) in enumerate(all_samples, 1):
        print(f"[{idx}/10] Image: {os.path.basename(img_path)} (True: {true_label})")
        
        # Predict
        try:
            with open(img_path, "rb") as img_file:
                files = {"file": (os.path.basename(img_path), img_file, "image/jpeg")}
                res = requests.post(PREDICT_URL, files=files, timeout=15)
            
            if res.status_code != 200:
                print(f"    [x] Prediction failed with status: {res.status_code}")
                continue
                
            pred_data = res.json()
            req_id = pred_data["request_id"]
            pred_label = pred_data["prediction"]
            conf = pred_data["confidence"]
            
            print(f"    -> Predicted: {pred_label} (Conf: {conf:.4f}) | Request ID: {req_id}")
            
            # Submit feedback (ground truth)
            feedback_payload = {
                "request_id": req_id,
                "true_label": true_label
            }
            fb_res = requests.post(FEEDBACK_URL, json=feedback_payload, timeout=5)
            if fb_res.status_code == 200:
                print(f"    -> Feedback logged successfully.")
            else:
                print(f"    [x] Feedback failed: {fb_res.status_code} - {fb_res.text}")
                
        except Exception as e:
            print(f"    [x] Error: {str(e)}")

    print("-" * 60)

    # 4. Fetch metrics and performance results
    print("\n[*] Fetching live monitoring metrics...")
    try:
        metrics = requests.get(METRICS_URL, timeout=5).json()
        print(json.dumps(metrics, indent=4))
    except Exception as e:
        print(f"[x] Failed to get metrics: {str(e)}")

    print("\n[*] Fetching live post-deployment performance tracker...")
    try:
        perf = requests.get(PERFORMANCE_URL, timeout=5).json()
        print(json.dumps(perf, indent=4))
    except Exception as e:
        print(f"[x] Failed to get performance data: {str(e)}")

    print("=" * 60)
    print("[+] Simulation complete!")
    print(f"Logs written to: {os.path.abspath('logs/')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
