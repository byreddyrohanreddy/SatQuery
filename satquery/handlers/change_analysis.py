"""
Change Analysis handler for SatQuery AI.

Implements bi-temporal change detection using classical image differencing:
  1. Align/resize both images to the same dimensions
  2. Compute absolute pixel difference
  3. Threshold the difference to find changed regions
  4. Find contours and return bounding boxes + text summary

This is a working implementation using basic CV — not a learned change
detection model.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from satquery.confidence import change_confidence
from satquery.models import BoundingBox, HandlerResult

logger = logging.getLogger(__name__)

# Threshold for the difference image (0–255 scale after normalization)
DIFF_THRESHOLD = 30

# Minimum contour area as fraction of image area
MIN_CONTOUR_FRACTION = 0.002


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_grayscale(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert an image to 8-bit grayscale numpy array."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("L"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            arr = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif image.ndim == 2:
            arr = image
        else:
            raise ValueError(f"Unexpected array shape: {image.shape}")
    else:
        raise ValueError(f"Unexpected image type: {type(image)}")

    # Normalize to 0–255 uint8
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return arr


def _describe_location(bbox: BoundingBox, img_w: int, img_h: int) -> str:
    """Describe a bounding box's location in human terms (quadrant-based)."""
    cx = bbox.x + bbox.width / 2
    cy = bbox.y + bbox.height / 2

    v = "upper" if cy < img_h / 3 else ("lower" if cy > 2 * img_h / 3 else "middle")
    h = "left" if cx < img_w / 3 else ("right" if cx > 2 * img_w / 3 else "center")

    if v == "middle" and h == "center":
        return "center"
    return f"{v}-{h}"


# ---------------------------------------------------------------------------
# Public handler
# ---------------------------------------------------------------------------

def change_analysis_handler(
    images: list[Image.Image | np.ndarray],
    description: Optional[str] = None,
    **kwargs,
) -> HandlerResult:
    """
    Detect changes between two co-registered images.

    Parameters
    ----------
    images : list
        Exactly 2 images (bi-temporal pair).
    description : str, optional
        Optional description of what kind of change to look for (not used
        by the current heuristic, but passed through for future models).

    Returns
    -------
    HandlerResult
    """
    if len(images) < 2:
        return HandlerResult(
            answer="Change analysis requires exactly 2 images (bi-temporal pair).",
            bounding_boxes=[],
            confidence=0.0,
            confidence_basis="Insufficient inputs",
            model_used="image_differencing",
        )

    # Convert both to grayscale
    gray1 = _to_grayscale(images[0])
    gray2 = _to_grayscale(images[1])

    # Resize to common dimensions (use the first image's size)
    target_h, target_w = gray1.shape[:2]
    if gray2.shape != gray1.shape:
        gray2 = cv2.resize(gray2, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Apply Gaussian blur to reduce noise before differencing
    gray1_blur = cv2.GaussianBlur(gray1, (5, 5), 0)
    gray2_blur = cv2.GaussianBlur(gray2, (5, 5), 0)

    # Absolute difference
    diff = cv2.absdiff(gray1_blur, gray2_blur)

    # Threshold
    _, thresh = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Calculate change percentage
    total_pixels = target_w * target_h
    changed_pixels = cv2.countNonZero(thresh)
    change_pct = (changed_pixels / total_pixels) * 100

    # Find contours of changed regions
    min_area = int(total_pixels * MIN_CONTOUR_FRACTION)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant = [c for c in contours if cv2.contourArea(c) >= min_area]

    # Build bounding boxes
    bboxes = []
    locations = []
    for contour in significant:
        x, y, bw, bh = cv2.boundingRect(contour)
        bb = BoundingBox(x=x, y=y, width=bw, height=bh, label="changed_region")
        bboxes.append(bb)
        locations.append(_describe_location(bb, target_w, target_h))

    # Confidence
    conf, basis = change_confidence(change_pct)

    # Build human-readable summary
    if bboxes:
        location_summary = ", ".join(sorted(set(locations)))
        answer = (
            f"{change_pct:.1f}% of the scene shows detectable changes, "
            f"concentrated in the {location_summary} "
            f"({len(bboxes)} distinct region(s) identified)."
        )
    else:
        answer = (
            f"Minimal change detected ({change_pct:.1f}% of pixels). "
            f"The two images appear very similar."
        )

    return HandlerResult(
        answer=answer,
        bounding_boxes=bboxes,
        confidence=conf,
        confidence_basis=basis,
        model_used="image_differencing",
    )
