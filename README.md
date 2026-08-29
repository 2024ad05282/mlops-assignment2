# End-to-End MLOps Pipeline: Cat vs Dog Classification

Project Demo - https://drive.google.com/file/d/1PuaseXeGaxObtCLykdJCNaRi46NgXT65/view?usp=sharing

Production-grade MLOps pipeline for binary image classification (Cat vs Dog) developed for a pet adoption platform. The project covers data tracking, baseline model training, experiment tracking with MLflow, containerized REST API inference with FastAPI, automated CI/CD using GitHub Actions and Docker Compose, post-deployment smoke testing, and continuous monitoring with a ground-truth feedback loop.

---

## 📑 Table of Contents
- [Project Architecture](#-project-architecture)
- [Repository Structure](#-repository-structure)
- [Module Breakdown](#-module-breakdown)
  - [M1: Model Development & Experiment Tracking](#m1-model-development--experiment-tracking)
  - [M2: Model Packaging & Containerization](#m2-model-packaging--containerization)
  - [M3: Continuous Integration Pipeline](#m3-continuous-integration-pipeline)
  - [M4: Continuous Deployment & Target Infrastructure](#m4-continuous-deployment--target-infrastructure)
  - [M5: Monitoring, Logging & Performance Tracking](#m5-monitoring-logging--performance-tracking)
- [Quickstart & Local Execution](#-quickstart--local-execution)
- [Verification & Testing](#-verification--testing)

---

## 🏗️ Project Architecture

```
[ Kaggle PetImages Dataset ] ──▶ [ Preprocessing & DVC ] ──▶ [ PyTorch CNN (train.py) ]
                                                                       │
                                                   (cat_dog_cnn.pt + MLflow Tracking)
                                                                       │
[ Client / Adoption App ] ◀── [ Docker Compose Deploy ] ◀── [ GitHub Actions CI/CD ]
            │                  (docker-compose.yml)          (Pytest + GHCR Image)
            ▼
[ FastAPI Inference (app.py) ]
    ├── /health       ──▶ Readiness probe
    ├── /predict      ──▶ Returns class label, probabilities & UUID
    ├── /feedback     ──▶ Logs ground-truth label for UUID
    ├── /metrics      ──▶ Real-time request count, latency & status codes
    └── /performance  ──▶ Post-deploy live accuracy & drift metrics
```

---

## 📂 Repository Structure

```text
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD workflow (Test, Build, Deploy)
├── data/                          # Dataset directory (tracked via DVC / Git)
│   └── PetImages/                 # Cat and Dog image folders
├── k8s/                           # Kubernetes infrastructure manifests
│   ├── deployment.yaml            # 2-replica Deployment with resource limits & probes
│   └── service.yaml               # NodePort/ClusterIP service manifest
├── logs/                          # Persistent logging directory
│   ├── app.log                    # Structured access & latency logs
│   ├── predictions.jsonl          # Request IDs and inference records
│   └── feedback.jsonl             # Ground-truth feedback records
├── models/                        # Serialized model checkpoint
│   └── cat_dog_cnn.pt             # Trained PyTorch model weights (97.1 MB)
├── plots/                         # Training evaluation artifacts
│   ├── classification_report.json # Precision, recall, and F1-scores
│   ├── confusion_matrix.png       # Test set confusion matrix plot
│   └── loss_accuracy_curves.png   # Per-epoch loss and accuracy curves
├── tests/                         # Automated unit tests (Pytest)
│   ├── test_preprocessing.py      # Image resize, normalization, corrupt file tests
│   └── test_model.py              # CNN forward pass, tensor shape, inference tests
├── app.py                         # FastAPI service with middleware, metrics & feedback
├── docker-compose.yml             # Docker Compose deployment manifest
├── Dockerfile                     # Multi-stage production container definition
├── local_cd_demo.py               # Local end-to-end CD simulation script
├── requirements.txt               # Pinned project dependencies
├── requirements-inference.txt     # Slim inference-only dependencies
├── simulate_monitoring.py         # Batch production simulation script
├── smoke_test.py                  # Post-deployment health & inference smoke test
├── train.py                       # CNN model training script with MLflow logging
└── README.md                      # Project documentation and report
```

---

## 📦 Module Breakdown

### M1: Model Development & Experiment Tracking
- **Dataset Preprocessing:** Raw Kaggle Cats and Dogs dataset cleaned of truncated/corrupted headers, resized to `128x128` RGB tensors, normalized with ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`), and split into **80% Train (19,998)**, **10% Validation (2,499)**, and **10% Test (2,501)**.
- **Model Architecture:** Custom PyTorch CNN consisting of 3 convolutional blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool2D) and 2 fully connected layers with Dropout (0.5) for regularization.
- **Model Serialization:** Trained checkpoint saved to `models/cat_dog_cnn.pt` containing model weights, architecture hyperparameters, class mappings, and validation metrics.
- **Experiment Tracking (MLflow):**
  - **Run ID:** `e6e24eb243fd453ba3610f33c87dbf98`
  - **Metrics Tracked:** Validation Accuracy (`83.03%`), Test Accuracy (`84.29%`), Test Loss (`0.3583`), Training Loss per epoch.
  - **Artifacts Logged:** `loss_accuracy_curves.png`, `confusion_matrix.png`, `classification_report.json`, model binary.

```bash
# Run training and track experiments
python train.py

# Launch MLflow UI
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

---

### M2: Model Packaging & Containerization
- **REST API (FastAPI):** Implemented in `app.py` with strict input schema validation:
  - `GET /health` — Returns service status, loaded model metadata, device, and uptime.
  - `POST /predict` — Accepts multipart image file (`image/jpeg`, `image/png`), runs forward pass, and returns predicted class (`Cat` or `Dog`), confidence score, probabilities, and a unique `request_id` (UUID).
- **Containerization (`Dockerfile`):**
  - Base Image: `python:3.13-slim`
  - Dependencies installed from pinned `requirements.txt`
  - Non-root application execution on port `8000`

```bash
# Build Docker image
docker build -t ghcr.io/2024ad05282/catdog-classifier:latest .

# Run Docker container
docker run -d --name catdog-inference-container -p 8000:8000 ghcr.io/2024ad05282/catdog-classifier:latest
```

---

### M3: Continuous Integration Pipeline
Configured in `.github/workflows/ci.yml` with automated triggers on push/PR to `main` and `master`:
1. **Job 1 (`test`):** Checks out code, sets up Python 3.13, installs dependencies, and runs automated unit tests with Pytest.
   - `test_preprocessing.py`: Validates transforms, tensor shapes, and corrupted image handling.
   - `test_model.py`: Validates model layer architecture, output shape `(1, 2)`, softmax probability range `[0, 1]`, and inference latency.
2. **Job 2 (`build-ci`):** Triggers only upon successful tests. Builds container image and pushes both `:latest` and `:<commit-sha>` tags to **GitHub Container Registry (GHCR)**.

---

### M4: Continuous Deployment & Target Infrastructure
- **Deployment Manifests:**
  - `docker-compose.yml` defining port mapping `8000:8000`, volume mounts for persistent `./logs:/app/logs`, and container healthchecks.
  - `k8s/deployment.yaml` and `k8s/service.yaml` specifying 2 replicas, resource requests/limits, and liveness/readiness probes.
- **CD Flow (`deploy-cd` job):**
  - Runs on the self-hosted Windows runner.
  - Executes `docker compose pull`, `docker compose down`, and `docker compose up -d` for seamless updates.
- **Post-Deploy Smoke Test:**
  - `smoke_test.py` polls `http://localhost:8000/health` with retry backoff.
  - Sends a synthetic RGB test payload to `http://localhost:8000/predict` and verifies response schema and status 200. Fails the pipeline if verification fails.

```bash
# Run post-deployment smoke tests manually
python smoke_test.py
```

---

### M5: Monitoring, Logging & Performance Tracking
- **Request/Response Logging:** Asynchronous HTTP middleware captures incoming requests, status codes, client IPs, and execution durations in seconds. Excludes raw image pixel data to protect sensitive payloads. Appends structured records to `logs/app.log`.
- **In-App Metrics Endpoint (`GET /metrics`):** Exposes total request counts, average latency, prediction class distribution, and HTTP status code tallies.
- **Ground-Truth Feedback Loop (`POST /feedback`):** Accepts `{ "request_id": "<uuid>", "true_label": "Cat|Dog" }` and stores ground truth in `logs/feedback.jsonl`.
- **Post-Deployment Performance Tracker (`GET /performance`):** Performs an inner-join on `predictions.jsonl` and `feedback.jsonl` by `request_id` to compute live real-world accuracy, confusion metrics, and detect potential model drift.
- **Simulation Script (`simulate_monitoring.py`):** Automatically sends a batch of 10 real sample images from `data/PetImages/`, submits corresponding ground truth feedback, and prints live metrics.

```bash
# Run monitoring and feedback simulation
python simulate_monitoring.py
```

---

## 🚀 Quickstart & Local Execution

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/2024ad05282/mlops-assignment2.git
cd mlops-assignment2

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model & Start MLflow
```bash
python train.py
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
```

### 3. Deploy via Docker Compose
```bash
docker compose pull
docker compose up -d
```

### 4. Verify API & Monitoring
```bash
# Run smoke tests
python smoke_test.py

# Run monitoring simulation
python simulate_monitoring.py

# Access Interactive API Docs
# Open in browser: http://localhost:8000/docs
```

---

## 🧪 Verification & Testing

| Test Suite / Script | Command | Purpose |
| :--- | :--- | :--- |
| **Unit Tests** | `pytest tests/ -v` | Data preprocessing & inference logic tests |
| **Smoke Tests** | `python smoke_test.py` | Container health and prediction response validation |
| **Simulation** | `python simulate_monitoring.py` | Live production traffic, feedback & performance metrics |
| **Local CD Demo** | `python local_cd_demo.py` | Standalone end-to-end build, deploy, update, test demo |

---

## 📊 Summary of Model Performance

| Metric | Training Result |
| :--- | :--- |
| **Architecture** | 3-Block CNN (Conv2D, BatchNorm, ReLU, MaxPool) + 2 FC |
| **Input Resolution** | 128 x 128 x 3 RGB |
| **Total Parameters** | 8,483,074 |
| **Validation Accuracy** | **83.03%** |
| **Test Accuracy** | **84.29%** |
| **Cat Precision / Recall** | 0.81 / 0.90 |
| **Dog Precision / Recall** | 0.89 / 0.79 |
| **Test Loss** | 0.3583 |
