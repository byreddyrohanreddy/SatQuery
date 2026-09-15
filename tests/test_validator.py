"""
Unit tests for the input validator.

Tests modality detection, format handling, and scene compatibility checks
using synthetic images.
"""

from __future__ import annotations

import numpy as np
import pytest

from satquery.validator import validate_inputs, _detect_modality


# ---------------------------------------------------------------------------
# Modality detection tests
# ---------------------------------------------------------------------------

class TestModalityDetection:
    """Tests for the optical vs SAR heuristic."""

    def test_rgb_image_detected_as_optical(self):
        """A 3-channel RGB image should be detected as optical."""
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        assert _detect_modality(arr) == "optical"

    def test_rgba_image_detected_as_optical(self):
        """A 4-channel RGBA image should be detected as optical."""
        arr = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        assert _detect_modality(arr) == "optical"

    def test_single_channel_uniform_detected_as_optical(self):
        """A single-channel image with low variance (uniform) → optical."""
        arr = np.full((100, 100), 128, dtype=np.uint8)
        # Uniform value → low CV → optical
        result = _detect_modality(arr)
        assert result == "optical"

    def test_single_channel_speckled_detected_as_sar(self):
        """A single-channel image with high speckle (high CV) → SAR."""
        # Create speckle-like distribution: exponential noise
        rng = np.random.RandomState(42)
        arr = rng.exponential(50, (100, 100)).clip(0, 255).astype(np.uint8)
        result = _detect_modality(arr)
        assert result == "SAR"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidateInputs:
    """Tests for the full validation pipeline."""

    def test_single_image_valid(self):
        """A single valid image should pass validation."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = validate_inputs([img])

        assert result.is_valid
        assert result.image_count == 1
        assert len(result.modalities) == 1
        assert result.modalities[0] in ("optical", "SAR")
        assert len(result.dimensions) == 1
        assert result.dimensions[0] == (100, 100)  # (width, height)

    def test_two_images_same_dims_no_warnings(self):
        """Two images with identical dimensions should produce no warnings."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        result = validate_inputs([img1, img2])

        assert result.is_valid
        assert result.image_count == 2
        # Same dimensions → no warnings about mismatch
        dim_warnings = [w for w in result.warnings if "dimension" in w.lower()]
        assert not dim_warnings

    def test_two_images_different_aspect_ratio_warns(self):
        """Two images with very different aspect ratios should generate a warning."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # Square
        img2 = np.random.randint(0, 256, (100, 400, 3), dtype=np.uint8)  # Very wide
        result = validate_inputs([img1, img2])

        assert result.is_valid  # Still valid, just warned
        assert any("dimension" in w.lower() or "aspect" in w.lower()
                    for w in result.warnings)

    def test_no_images_invalid(self):
        """Empty image list should be invalid."""
        result = validate_inputs([])

        assert not result.is_valid
        assert result.image_count == 0

    def test_three_images_truncated_with_warning(self):
        """More than 2 images should be truncated with a warning."""
        imgs = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
        result = validate_inputs(imgs)

        assert result.is_valid
        assert result.image_count == 2  # Truncated
        assert any("first two" in w.lower() for w in result.warnings)

    def test_formats_are_detected(self):
        """Format should be detected for each image."""
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = validate_inputs([img])

        assert len(result.formats) == 1
        assert result.formats[0]  # Not empty


class TestSceneCompatibility:
    """Tests for the scene compatibility checker."""

    def test_same_dimensions_compatible(self):
        """Images with same dimensions are compatible."""
        from satquery.validator import _check_scene_compatibility
        warnings = _check_scene_compatibility([(100, 100), (100, 100)])
        assert not warnings

    def test_similar_aspect_ratio_mild_warning(self):
        """Slightly different dimensions with same AR → mild warning."""
        from satquery.validator import _check_scene_compatibility
        warnings = _check_scene_compatibility([(100, 100), (200, 200)])
        # Same AR, different size → compatible with note
        assert len(warnings) <= 1

    def test_very_different_ar_strong_warning(self):
        """Very different aspect ratios → strong warning."""
        from satquery.validator import _check_scene_compatibility
        warnings = _check_scene_compatibility([(100, 100), (400, 100)])
        assert len(warnings) == 1
        assert "significantly" in warnings[0].lower() or "may not" in warnings[0].lower()
