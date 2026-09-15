"""
Unit tests for all specialist handlers.

Uses synthetic test images (solid colors, simple shapes) generated in-test,
so no real satellite imagery is required.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

def make_solid_color(width: int, height: int, rgb: tuple[int, int, int]) -> np.ndarray:
    """Create a solid-color RGB image as a numpy array."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = rgb
    return arr


def make_grayscale(width: int, height: int, value: int = 128) -> np.ndarray:
    """Create a solid grayscale image as a numpy array."""
    return np.full((height, width), value, dtype=np.uint8)


def make_two_color(width: int, height: int,
                   color1: tuple[int, int, int],
                   color2: tuple[int, int, int]) -> np.ndarray:
    """Create an image split vertically: left half color1, right half color2."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    mid = width // 2
    arr[:, :mid] = color1
    arr[:, mid:] = color2
    return arr


# ---------------------------------------------------------------------------
# vqa_caption handler tests
# ---------------------------------------------------------------------------

class TestVQACaptionHandler:
    """Tests for the VQA/caption handler using BLIP models."""

    def test_captioning_returns_nonempty_answer(self):
        """Captioning should produce a non-empty text answer."""
        from satquery.handlers.vqa_caption import vqa_caption_handler

        # Solid blue image (simple enough for BLIP to process)
        img = make_solid_color(224, 224, (0, 0, 255))
        result = vqa_caption_handler(images=[img])

        assert result.answer, "Caption should not be empty"
        assert isinstance(result.answer, str)
        assert result.confidence > 0
        assert result.model_used == "Salesforce/blip-image-captioning-base"

    def test_vqa_returns_nonempty_answer(self):
        """VQA should produce a non-empty answer to a question."""
        from satquery.handlers.vqa_caption import vqa_caption_handler

        img = make_solid_color(224, 224, (0, 255, 0))
        result = vqa_caption_handler(images=[img], question="What color is this?")

        assert result.answer, "VQA answer should not be empty"
        assert isinstance(result.answer, str)
        assert result.confidence > 0
        assert result.model_used == "Salesforce/blip-vqa-base"

    def test_captioning_confidence_basis(self):
        """Confidence basis should be a meaningful explanation."""
        from satquery.handlers.vqa_caption import vqa_caption_handler

        img = make_solid_color(224, 224, (128, 128, 128))
        result = vqa_caption_handler(images=[img])

        assert "BLIP" in result.confidence_basis
        assert result.confidence_basis  # Not empty

    def test_handles_grayscale_input(self):
        """Handler should work with single-channel grayscale images."""
        from satquery.handlers.vqa_caption import vqa_caption_handler

        gray = make_grayscale(224, 224, 128)
        result = vqa_caption_handler(images=[gray])

        assert result.answer, "Should handle grayscale input"


# ---------------------------------------------------------------------------
# grounding handler tests
# ---------------------------------------------------------------------------

class TestGroundingHandler:
    """Tests for the grounding handler (HSV thresholding + contours)."""

    def test_detects_water_in_blue_image(self):
        """A mostly-blue image should produce water detections."""
        from satquery.handlers.grounding import grounding_handler

        # Blue image (simulating water)
        img = make_solid_color(200, 200, (30, 80, 200))
        result = grounding_handler(images=[img], target_object="water")

        assert result.bounding_boxes, "Should detect water regions in blue image"
        assert result.confidence > 0
        assert "water" in result.answer.lower() or "region" in result.answer.lower()

    def test_detects_vegetation_in_green_image(self):
        """A mostly-green image should produce vegetation detections."""
        from satquery.handlers.grounding import grounding_handler

        img = make_solid_color(200, 200, (30, 180, 30))
        result = grounding_handler(images=[img], target_object="vegetation")

        assert result.bounding_boxes, "Should detect vegetation in green image"
        for bb in result.bounding_boxes:
            assert bb.label == "vegetation"

    def test_no_detection_for_unknown_target(self):
        """Unknown targets should return a clear explanation."""
        from satquery.handlers.grounding import grounding_handler

        img = make_solid_color(200, 200, (128, 128, 128))
        result = grounding_handler(images=[img], target_object="spacecraft")

        assert not result.bounding_boxes
        assert "placeholder" in result.answer.lower() or "could not" in result.answer.lower()

    def test_no_water_in_green_image(self):
        """A green image should not detect water."""
        from satquery.handlers.grounding import grounding_handler

        img = make_solid_color(200, 200, (30, 200, 30))
        result = grounding_handler(images=[img], target_object="water")

        # Might still detect some noise, but bboxes should be empty or very small
        assert result.confidence < 0.5

    def test_bounding_box_has_valid_coordinates(self):
        """All returned bounding boxes should have valid dimensions."""
        from satquery.handlers.grounding import grounding_handler

        img = make_solid_color(200, 200, (30, 80, 200))
        result = grounding_handler(images=[img], target_object="water")

        for bb in result.bounding_boxes:
            assert bb.x >= 0
            assert bb.y >= 0
            assert bb.width > 0
            assert bb.height > 0


# ---------------------------------------------------------------------------
# change_analysis handler tests
# ---------------------------------------------------------------------------

class TestChangeAnalysisHandler:
    """Tests for the change analysis handler (image differencing)."""

    def test_detects_significant_change(self):
        """Two very different images should show high change percentage."""
        from satquery.handlers.change_analysis import change_analysis_handler

        img1 = make_solid_color(200, 200, (0, 0, 0))      # Black
        img2 = make_solid_color(200, 200, (255, 255, 255)) # White

        result = change_analysis_handler(images=[img1, img2])

        assert "%" in result.answer
        assert result.bounding_boxes  # Should have changed regions
        assert result.confidence > 0

    def test_no_change_in_identical_images(self):
        """Two identical images should show minimal change."""
        from satquery.handlers.change_analysis import change_analysis_handler

        img = make_solid_color(200, 200, (100, 100, 100))
        result = change_analysis_handler(images=[img, img.copy()])

        assert "minimal" in result.answer.lower() or "0.0%" in result.answer
        assert not result.bounding_boxes  # No changed regions

    def test_partial_change(self):
        """Images that differ in only part should detect localized change."""
        from satquery.handlers.change_analysis import change_analysis_handler

        img1 = make_solid_color(200, 200, (100, 100, 100))
        img2 = img1.copy()
        # Change the right half to bright white (large luminance difference)
        img2[:, 100:] = [255, 255, 255]

        result = change_analysis_handler(images=[img1, img2])

        assert result.bounding_boxes
        assert result.confidence > 0

    def test_handles_different_dimensions(self):
        """Should resize images to match before differencing."""
        from satquery.handlers.change_analysis import change_analysis_handler

        img1 = make_solid_color(200, 200, (0, 0, 0))
        img2 = make_solid_color(300, 300, (255, 255, 255))

        result = change_analysis_handler(images=[img1, img2])

        # Should complete without error
        assert result.answer
        assert result.model_used == "image_differencing"

    def test_requires_two_images(self):
        """Should return error message with single image."""
        from satquery.handlers.change_analysis import change_analysis_handler

        img = make_solid_color(200, 200, (100, 100, 100))
        result = change_analysis_handler(images=[img])

        assert "requires" in result.answer.lower() or "2 images" in result.answer.lower()
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# fusion handler tests
# ---------------------------------------------------------------------------

class TestFusionHandler:
    """Tests for the optical+SAR fusion handler (dual captioning)."""

    def test_fusion_produces_combined_answer(self):
        """Fusion should mention both images in the answer."""
        from satquery.handlers.fusion import fusion_handler

        img1 = make_solid_color(224, 224, (100, 150, 200))  # "Optical"
        img2 = make_grayscale(224, 224, 128)                 # "SAR"

        result = fusion_handler(
            images=[img1, img2],
            modalities=["optical", "SAR"],
        )

        answer_lower = result.answer.lower()
        assert "optical" in answer_lower or "image 1" in answer_lower
        assert "sar" in answer_lower or "image 2" in answer_lower
        assert result.model_used == "fusion_dual_caption"

    def test_fusion_confidence_is_reduced(self):
        """Fusion confidence should be lower than individual captions."""
        from satquery.handlers.fusion import fusion_handler
        from satquery.handlers.vqa_caption import vqa_caption_handler

        img = make_solid_color(224, 224, (100, 150, 200))

        single_result = vqa_caption_handler(images=[img])
        fusion_result = fusion_handler(images=[img, img], modalities=["optical", "SAR"])

        # Fusion applies a 0.9 reduction factor
        assert fusion_result.confidence <= single_result.confidence

    def test_requires_two_images(self):
        """Should return error with single image."""
        from satquery.handlers.fusion import fusion_handler

        img = make_solid_color(224, 224, (100, 100, 100))
        result = fusion_handler(images=[img])

        assert "requires" in result.answer.lower() or "2 images" in result.answer.lower()
        assert result.confidence == 0.0

    def test_fusion_with_question(self):
        """Fusion with a question should pass it to both sub-analyses."""
        from satquery.handlers.fusion import fusion_handler

        img1 = make_solid_color(224, 224, (100, 150, 200))
        img2 = make_solid_color(224, 224, (50, 50, 50))

        result = fusion_handler(
            images=[img1, img2],
            question="What do you see?",
            modalities=["optical", "SAR"],
        )

        assert result.answer
        assert result.confidence > 0
