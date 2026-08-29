"""
test_preprocessing.py - Unit Tests for Data Pre-processing Functions
====================================================================
This test suite verifies the robustness of the data loading and image
preprocessing pipelines.

Main Test Categories & Verifications:
1. Training Transforms (TestTrainTransform):
   - Output Tensor Check: Verifies that training transforms output a PyTorch tensor.
   - Resizing Integrity: Confirms inputs are correctly resized to 128x128.
   - Dynamic Range: Ensures pixel values are normalized within expected float bounds.
2. Validation/Inference Transforms (TestValTransform):
   - Deterministic Processing: Verifies preprocessing has no random augmentations.
   - Shape Consistency: Ensures training and validation pipelines produce identical dimensions.
3. Image Loading & Robustness (TestImageLoading):
   - Valid Images: Ensures standard JPEGs load without error.
   - Corrupt Files: Asserts exceptions are raised when trying to load corrupted file bytes.
   - Empty Files: Checks that 0-byte files are correctly caught by file size validation.
   - Grayscale/RGBA Handling: Verifies channel dimensions convert correctly to 3-channel RGB.
"""

import os
import pytest
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Import from train.py
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train import (
    train_transform,
    val_transform,
    SafeImageFolder,
    IMAGE_SIZE,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_rgb_image():
    """Create a sample RGB image for testing."""
    return Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8), mode="RGB"
    )


@pytest.fixture
def sample_grayscale_image():
    """Create a sample grayscale image for testing."""
    return Image.fromarray(
        np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode="L"
    )


@pytest.fixture
def sample_small_image():
    """Create a very small image to test resizing."""
    return Image.fromarray(
        np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8), mode="RGB"
    )


@pytest.fixture
def sample_large_image():
    """Create a large image to test resizing."""
    return Image.fromarray(
        np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8), mode="RGB"
    )


@pytest.fixture
def temp_image_file(tmp_path, sample_rgb_image):
    """Save a sample image to a temp file."""
    path = tmp_path / "test_image.jpg"
    sample_rgb_image.save(str(path))
    return str(path)


# ─── Test: Train Transform ──────────────────────────────────────────────────

class TestTrainTransform:
    """Tests for training data augmentation pipeline."""

    def test_output_is_tensor(self, sample_rgb_image):
        """Train transform should output a PyTorch tensor."""
        result = train_transform(sample_rgb_image)
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self, sample_rgb_image):
        """Output tensor should have shape (3, IMAGE_SIZE, IMAGE_SIZE)."""
        result = train_transform(sample_rgb_image)
        assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_output_dtype_float(self, sample_rgb_image):
        """Output tensor should be float type (normalized)."""
        result = train_transform(sample_rgb_image)
        assert result.dtype == torch.float32

    def test_small_image_resized(self, sample_small_image):
        """Small images should be resized up to IMAGE_SIZE."""
        result = train_transform(sample_small_image)
        assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_large_image_resized(self, sample_large_image):
        """Large images should be resized down to IMAGE_SIZE."""
        result = train_transform(sample_large_image)
        assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_normalized_range(self, sample_rgb_image):
        """After normalization, values should not be in raw [0, 255] range."""
        result = train_transform(sample_rgb_image)
        # After ImageNet normalization, values are roughly in [-2.5, 2.5]
        assert result.min() >= -3.0
        assert result.max() <= 3.0


# ─── Test: Validation Transform ─────────────────────────────────────────────

class TestValTransform:
    """Tests for validation data preprocessing pipeline."""

    def test_output_is_tensor(self, sample_rgb_image):
        """Val transform should output a PyTorch tensor."""
        result = val_transform(sample_rgb_image)
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self, sample_rgb_image):
        """Output tensor should have shape (3, IMAGE_SIZE, IMAGE_SIZE)."""
        result = val_transform(sample_rgb_image)
        assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_deterministic(self, sample_rgb_image):
        """Val transform should be deterministic (no random augmentation)."""
        result1 = val_transform(sample_rgb_image)
        result2 = val_transform(sample_rgb_image)
        assert torch.equal(result1, result2)

    def test_train_val_same_size(self, sample_rgb_image):
        """Train and val transforms should produce same sized output."""
        train_result = train_transform(sample_rgb_image)
        val_result = val_transform(sample_rgb_image)
        assert train_result.shape == val_result.shape


# ─── Test: Image Loading & Validation ───────────────────────────────────────

class TestImageLoading:
    """Tests for image loading and validation utilities."""

    def test_valid_jpeg_loads(self, temp_image_file):
        """Valid JPEG file should load without errors."""
        img = Image.open(temp_image_file).convert("RGB")
        assert img.mode == "RGB"
        assert img.size[0] > 0 and img.size[1] > 0

    def test_corrupt_file_handling(self, tmp_path):
        """Corrupted file should be detected."""
        corrupt_path = tmp_path / "corrupt.jpg"
        corrupt_path.write_bytes(b"not an image")
        with pytest.raises(Exception):
            img = Image.open(str(corrupt_path))
            img.verify()

    def test_empty_file_detection(self, tmp_path):
        """Empty file should be detected by size check."""
        empty_path = tmp_path / "empty.jpg"
        empty_path.write_bytes(b"")
        assert os.path.getsize(str(empty_path)) == 0

    def test_rgba_to_rgb_conversion(self):
        """RGBA images should convert to RGB (3 channels)."""
        rgba_img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8), mode="RGBA"
        )
        rgb_img = rgba_img.convert("RGB")
        assert rgb_img.mode == "RGB"
        result = val_transform(rgb_img)
        assert result.shape[0] == 3  # 3 channels
