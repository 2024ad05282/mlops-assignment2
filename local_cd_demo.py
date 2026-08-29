"""
local_cd_demo.py - Local Continuous Deployment Demo
====================================================
MLOps Assignment 2 - Demonstrates the CD pipeline locally.

This script simulates what GitHub Actions does in the cloud, but on your
local machine so you can DEMO it live:

  Step 1: Build Docker image (simulates CI build)
  Step 2: Deploy via Docker Compose (simulates CD deployment)
  Step 3: Run smoke tests (health + prediction)
  Step 4: Simulate a code change (version bump)
  Step 5: Rebuild & Redeploy (simulates CD auto-update)
  Step 6: Verify new version is live
  Step 7: Teardown

Pre-requisites:
  - Docker Desktop must be running
  - Run from the project root directory

Usage:
  python local_cd_demo.py
"""

import os
import sys
import time
import json
import subprocess
import requests
from PIL import Image
import io

# ─── Config ──────────────────────────────────────────────────────────────────

IMAGE_NAME = "catdog-classifier"
IMAGE_TAG = f"{IMAGE_NAME}:latest"
COMPOSE_FILE = "docker-compose.yml"
APP_FILE = "app.py"
BASE_URL = "http://localhost:8000"
HEALTH_URL = f"{BASE_URL}/health"
PREDICT_URL = f"{BASE_URL}/predict"
METRICS_URL = f"{BASE_URL}/metrics"

MAX_RETRIES = 12
RETRY_DELAY = 5


# ─── Utilities ───────────────────────────────────────────────────────────────

def print_header(step_num, title):
    print(f"\n{'='*65}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'='*65}\n")


def print_success(msg):
    print(f"  [+] {msg}")


def print_fail(msg):
    print(f"  [x] {msg}")


def print_info(msg):
    print(f"  [*] {msg}")


def run_cmd(cmd, desc=""):
    """Run a shell command and print output."""
    if desc:
        print_info(f"{desc}...")
    print(f"  $ {cmd}\n")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        # Print limited output
        lines = result.stdout.strip().split("\n")
        for line in lines[:30]:
            print(f"    {line}")
        if len(lines) > 30:
            print(f"    ... ({len(lines) - 30} more lines)")
    if result.returncode != 0:
        if result.stderr:
            for line in result.stderr.strip().split("\n")[:10]:
                print(f"    [stderr] {line}")
        return False
    return True


def wait_for_service():
    """Wait for the deployed service to become ready."""
    print_info(f"Waiting for service at {HEALTH_URL}...")
    for i in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(HEALTH_URL, timeout=5)
            if r.status_code == 200:
                print_success(f"Service is UP! (attempt {i}/{MAX_RETRIES})")
                return True
        except requests.exceptions.RequestException:
            pass
        print(f"    Retry {i}/{MAX_RETRIES}... waiting {RETRY_DELAY}s")
        time.sleep(RETRY_DELAY)
    return False


def get_service_version():
    """Get current service version from /health endpoint."""
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        data = r.json()
        return data
    except Exception:
        return None


def test_prediction():
    """Send a test prediction and return the result."""
    img = Image.new("RGB", (128, 128), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    files = {"file": ("demo_test.jpg", img_bytes, "image/jpeg")}
    r = requests.post(PREDICT_URL, files=files, timeout=10)
    return r.json() if r.status_code == 200 else None


def get_current_version():
    """Read current version from app.py."""
    with open(APP_FILE, "r") as f:
        for line in f:
            if 'version=' in line and '"' in line:
                # Extract version string
                start = line.index('"') + 1
                end = line.index('"', start)
                return line[start:end]
    return "unknown"


def bump_version(old_ver, new_ver):
    """Change version in app.py to simulate a code change."""
    with open(APP_FILE, "r") as f:
        content = f.read()

    updated = content.replace(f'version="{old_ver}"', f'version="{new_ver}"')
    with open(APP_FILE, "w") as f:
        f.write(updated)


# ─── Main Demo Flow ─────────────────────────────────────────────────────────

def main():
    print("\n" + "#" * 65)
    print("#" + " " * 63 + "#")
    print("#   LOCAL CONTINUOUS DEPLOYMENT (CD) DEMO                       #")
    print("#   MLOps Assignment 2                                          #")
    print("#" + " " * 63 + "#")
    print("#" * 65)
    print("\nThis script demonstrates the CD pipeline LOCALLY:")
    print("  Build -> Deploy -> Test -> Code Change -> Rebuild -> Redeploy -> Verify\n")

    # ── Pre-check: Docker running? ───────────────────────────────────────
    print_info("Checking if Docker is running...")
    result = subprocess.run("docker info", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print_fail("Docker is not running! Please start Docker Desktop first.")
        sys.exit(1)
    print_success("Docker is running.\n")

    original_version = get_current_version()
    demo_version = "2.0.0-cd-demo"
    print_info(f"Current app version: {original_version}")

    try:
        # ═════════════════════════════════════════════════════════════════
        # STEP 1: Build Docker Image (CI Build Phase)
        # ═════════════════════════════════════════════════════════════════
        print_header(1, "BUILD DOCKER IMAGE (simulates CI build)")
        success = run_cmd(
            f"docker build -t {IMAGE_TAG} .",
            "Building Docker image from Dockerfile"
        )
        if not success:
            print_fail("Docker build failed!")
            sys.exit(1)
        print_success(f"Docker image '{IMAGE_TAG}' built successfully!")

        # Show the built image
        run_cmd(f"docker images {IMAGE_NAME}", "Listing built image")

        # ═════════════════════════════════════════════════════════════════
        # STEP 2: Deploy via Docker Compose (CD Deployment Phase)
        # ═════════════════════════════════════════════════════════════════
        print_header(2, "DEPLOY VIA DOCKER COMPOSE (simulates CD deployment)")

        # Stop any existing deployment first
        run_cmd("docker compose down 2>nul", "Stopping any existing deployment")
        time.sleep(2)

        # Deploy with local image
        os.environ["IMAGE_TAG"] = IMAGE_TAG
        success = run_cmd(
            f"docker compose up -d",
            "Starting service via Docker Compose"
        )
        if not success:
            print_fail("Docker Compose deployment failed!")
            sys.exit(1)

        print_success("Service deployed!")
        run_cmd("docker compose ps", "Checking running containers")

        # ═════════════════════════════════════════════════════════════════
        # STEP 3: Run Smoke Tests (Post-deploy Verification)
        # ═════════════════════════════════════════════════════════════════
        print_header(3, "RUN SMOKE TESTS (post-deploy verification)")

        if not wait_for_service():
            print_fail("Service did not start in time!")
            run_cmd("docker compose logs", "Container logs")
            sys.exit(1)

        # Health Check
        print_info("Testing GET /health ...")
        health = get_service_version()
        if health and health.get("status") == "healthy":
            print_success("Health check PASSED!")
            print(f"    Status : {health['status']}")
            print(f"    Model  : {'loaded' if health['model']['loaded'] else 'NOT loaded'}")
            print(f"    Device : {health['model']['device']}")
            print(f"    Classes: {health['model']['classes']}")
        else:
            print_fail("Health check FAILED!")
            sys.exit(1)

        # Prediction Test
        print_info("\nTesting POST /predict ...")
        pred = test_prediction()
        if pred and "prediction" in pred:
            print_success("Prediction endpoint PASSED!")
            print(f"    Predicted : {pred['prediction']}")
            print(f"    Confidence: {pred['confidence']}")
            print(f"    Probs     : {pred['probabilities']}")
        else:
            print_fail("Prediction test FAILED!")
            sys.exit(1)

        print(f"\n  {'─'*55}")
        print_success("ALL SMOKE TESTS PASSED! Service v{} is live.".format(original_version))
        print(f"  {'─'*55}")

        # ═════════════════════════════════════════════════════════════════
        # STEP 4: Simulate Code Change (Developer pushes a change)
        # ═════════════════════════════════════════════════════════════════
        print_header(4, "SIMULATE CODE CHANGE (version bump)")

        print_info(f"Changing app version: '{original_version}' -> '{demo_version}'")
        print_info("(This simulates a developer pushing a code change to main)\n")
        bump_version(original_version, demo_version)
        print_success(f"app.py updated! New version = '{demo_version}'")

        input("\n  >>> Press ENTER to trigger rebuild & redeploy (simulates CD)... ")

        # ═════════════════════════════════════════════════════════════════
        # STEP 5: Rebuild & Redeploy (CD auto-update)
        # ═════════════════════════════════════════════════════════════════
        print_header(5, "REBUILD & REDEPLOY (simulates CD auto-update on push)")

        print_info("In production, GitHub Actions would automatically:")
        print("    1. Detect the push to main")
        print("    2. Run tests")
        print("    3. Build new Docker image")
        print("    4. Push to container registry (GHCR)")
        print("    5. Pull & deploy the updated image")
        print_info("\nWe simulate this locally now:\n")

        # Rebuild
        success = run_cmd(
            f"docker build -t {IMAGE_TAG} .",
            "[CD] Rebuilding Docker image with new code"
        )
        if not success:
            print_fail("Rebuild failed!")
            sys.exit(1)
        print_success("New image built with updated code!")

        # Redeploy
        run_cmd("docker compose down", "[CD] Stopping old deployment")
        time.sleep(2)

        os.environ["IMAGE_TAG"] = IMAGE_TAG
        success = run_cmd(
            "docker compose up -d",
            "[CD] Deploying updated service"
        )
        if not success:
            print_fail("Redeployment failed!")
            sys.exit(1)

        print_success("Updated service deployed!")

        # ═════════════════════════════════════════════════════════════════
        # STEP 6: Verify New Version is Live
        # ═════════════════════════════════════════════════════════════════
        print_header(6, "VERIFY UPDATED SERVICE IS LIVE")

        if not wait_for_service():
            print_fail("Updated service did not start!")
            sys.exit(1)

        # Check health with new version
        print_info("Verifying updated service...")
        health = get_service_version()
        if health and health.get("status") == "healthy":
            print_success("Updated service is healthy!")
        else:
            print_fail("Health check failed after update!")
            sys.exit(1)

        # Test prediction still works
        pred = test_prediction()
        if pred and "prediction" in pred:
            print_success("Prediction still works after update!")
            print(f"    Predicted : {pred['prediction']}")
            print(f"    Confidence: {pred['confidence']}")
        else:
            print_fail("Prediction broken after update!")
            sys.exit(1)

        # Fetch metrics
        print_info("\nFetching /metrics ...")
        try:
            metrics = requests.get(METRICS_URL, timeout=5).json()
            print(f"    {json.dumps(metrics, indent=4)}")
        except Exception:
            pass

        print(f"\n  {'─'*55}")
        print_success("CD DEMO COMPLETE!")
        print(f"  Old version : {original_version}")
        print(f"  New version : {demo_version}")
        print(f"  API running : {BASE_URL}")
        print(f"  {'─'*55}")

    finally:
        # ═════════════════════════════════════════════════════════════════
        # STEP 7: Cleanup
        # ═════════════════════════════════════════════════════════════════
        print_header(7, "CLEANUP")

        # Restore original version in app.py
        print_info(f"Restoring app.py version back to '{original_version}'...")
        try:
            current = get_current_version()
            if current != original_version:
                bump_version(current, original_version)
                print_success(f"app.py restored to version '{original_version}'")
        except Exception as e:
            print_fail(f"Could not restore version: {e}")
            print_info(f"Please manually set version back to '{original_version}' in app.py")

        # Ask user if they want to keep the container running
        print()
        choice = input("  >>> Keep the container running? (y/n): ").strip().lower()
        if choice != "y":
            run_cmd("docker compose down", "Stopping containers")
            print_success("Containers stopped and removed.")
        else:
            print_success(f"Container is still running at {BASE_URL}")
            print_info("To stop later: docker compose down")

    print(f"\n{'#'*65}")
    print("#  LOCAL CD DEMO FINISHED SUCCESSFULLY!                         #")
    print(f"{'#'*65}\n")


if __name__ == "__main__":
    main()
