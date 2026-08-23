"""
test_model.py - Unit Tests for Model Utility & Inference Functions
==================================================================
Tests model architecture, forward pass, inference, and model loading.
"""

import os
import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train import CatDogCNN, IMAGE_SIZE, val_transform
from app import load_model, inference_transform, CatDogCNN as AppCatDogCNN

MODEL_PATH = os.path.join("models", "cat_dog_cnn.pt")


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def model():
    """Create a fresh CatDogCNN model instance."""
    return CatDogCNN()


@pytest.fixture
def dummy_input():
    """Create a dummy input tensor matching expected input shape."""
    return torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture
def dummy_batch():
    """Create a batch of dummy inputs."""
    return torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture
def sample_image():
    """Create a sample PIL image for end-to-end inference testing."""
    return Image.fromarray(
        np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8), mode="RGB"
    )


# ─── Test: Model Architecture ───────────────────────────────────────────────

class TestModelArchitecture:
    """Tests for the CatDogCNN model structure."""

    def test_model_has_features(self, model):
        """Model should have a features (conv) component."""
        assert hasattr(model, "features")
        assert isinstance(model.features, nn.Sequential)

    def test_model_has_classifier(self, model):
        """Model should have a classifier (FC) component."""
        assert hasattr(model, "classifier")
        assert isinstance(model.classifier, nn.Sequential)

    def test_output_classes(self, model, dummy_input):
        """Model output should have 2 classes (Cat, Dog)."""
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 2)

    def test_batch_output_shape(self, model, dummy_batch):
        """Model should handle batched inputs correctly."""
        model.eval()
        with torch.no_grad():
            output = model(dummy_batch)
        assert output.shape == (4, 2)

    def test_parameter_count(self, model):
        """Model should have a reasonable number of parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
        assert total_params < 50_000_000  # Sanity check: under 50M params


# ─── Test: Forward Pass ─────────────────────────────────────────────────────

class TestForwardPass:
    """Tests for model forward pass behavior."""

    def test_forward_returns_tensor(self, model, dummy_input):
        """Forward pass should return a tensor."""
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert isinstance(output, torch.Tensor)

    def test_forward_output_not_nan(self, model, dummy_input):
        """Forward pass output should not contain NaN values."""
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        assert not torch.isnan(output).any()

    def test_softmax_sums_to_one(self, model, dummy_input):
        """Softmax of output should sum to 1."""
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            probs = torch.nn.functional.softmax(output, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_eval_mode_no_dropout(self, model, dummy_input):
        """In eval mode, same input should give same output (no dropout)."""
        model.eval()
        with torch.no_grad():
            out1 = model(dummy_input)
            out2 = model(dummy_input)
        assert torch.equal(out1, out2)


# ─── Test: Model Loading ────────────────────────────────────────────────────

class TestModelLoading:
    """Tests for loading the saved model checkpoint."""

    @pytest.mark.skipif(
        not os.path.exists(MODEL_PATH),
        reason="Trained model not found. Run train.py first."
    )
    def test_checkpoint_loads(self):
        """Saved checkpoint should load without errors."""
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "class_names" in checkpoint

    @pytest.mark.skipif(
        not os.path.exists(MODEL_PATH),
        reason="Trained model not found. Run train.py first."
    )
    def test_checkpoint_has_required_keys(self):
        """Checkpoint should contain all required keys."""
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        required_keys = ["model_state_dict", "class_names", "val_acc", "image_size"]
        for key in required_keys:
            assert key in checkpoint, f"Missing key: {key}"

    @pytest.mark.skipif(
        not os.path.exists(MODEL_PATH),
        reason="Trained model not found. Run train.py first."
    )
    def test_loaded_model_inference(self):
        """Loaded model should produce valid predictions."""
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model = CatDogCNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            output = model(dummy)
        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()


# ─── Test: End-to-End Inference ──────────────────────────────────────────────

class TestEndToEndInference:
    """Tests for the full inference pipeline (image -> prediction)."""

    @pytest.mark.skipif(
        not os.path.exists(MODEL_PATH),
        reason="Trained model not found. Run train.py first."
    )
    def test_image_to_prediction(self, sample_image):
        """Full pipeline: PIL image -> transform -> model -> class label."""
        # Load model
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model = CatDogCNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        class_names = checkpoint["class_names"]

        # Preprocess
        input_tensor = inference_transform(sample_image).unsqueeze(0)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)

        predicted_idx = probs[0].argmax().item()
        predicted_label = class_names[predicted_idx]
        confidence = probs[0][predicted_idx].item()

        # Validate
        assert predicted_label in ["Cat", "Dog"]
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.skipif(
        not os.path.exists(MODEL_PATH),
        reason="Trained model not found. Run train.py first."
    )
    def test_prediction_probabilities_valid(self, sample_image):
        """Prediction probabilities should sum to 1 and be in [0, 1]."""
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model = CatDogCNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        input_tensor = inference_transform(sample_image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)

        probs_np = probs[0].numpy()
        assert all(0.0 <= p <= 1.0 for p in probs_np)
        assert abs(probs_np.sum() - 1.0) < 1e-5
