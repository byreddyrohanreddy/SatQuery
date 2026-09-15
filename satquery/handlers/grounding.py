"""
Grounding handler for SatQuery AI.

Implements object localization using classical computer vision:
  - HSV color thresholding for common target categories
  - Contour detection to produce bounding boxes

╔═══════════════════════════════════════════════════════════════════════╗
║  PLACEHOLDER IMPLEMENTATION                                          ║
║  This is a classical CV heuristic standing in for GeoChat's native   ║
║  grounding capability. It will be replaced post-fine-tuning with     ║
║  a learned grounding model that handles arbitrary target objects.     ║
║  Current support is limited to color-based targets:                  ║
║    water, vegetation, buildings/urban, bare soil/sand, roads          ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from satquery.confidence import grounding_confidence
from satquery.models import BoundingBox, HandlerResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HSV color ranges for common satellite imagery targets
# ---------------------------------------------------------------------------

# Each entry: (lower_hsv, upper_hsv) — multiple ranges per target for robustness
TARGET_COLOR_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    "water": [
        (np.array([90, 30, 30]), np.array([130, 255, 200])),   # Blue range
        (np.array([85, 20, 20]), np.array([135, 255, 180])),    # Broader blue
    ],
    "vegetation": [
        (np.array([30, 30, 30]), np.array([85, 255, 255])),     # Green range
        (np.array([25, 20, 20]), np.array([90, 255, 255])),     # Broader green
    ],
    "building": [
        (np.array([0, 0, 80]), np.array([30, 60, 200])),        # Gray/beige
        (np.array([0, 0, 100]), np.array([180, 40, 220])),      # Low-saturation gray
    ],
    "urban": [
        (np.array([0, 0, 80]), np.array([180, 40, 220])),       # Low-saturation gray
    ],
    "soil": [
        (np.array([5, 30, 50]), np.array([25, 200, 200])),      # Brown/tan
        (np.array([15, 20, 100]), np.array([35, 180, 240])),    # Sandy
    ],
    "sand": [
        (np.array([15, 20, 100]), np.array([35, 180, 240])),    # Sandy
        (np.array([20, 10, 150]), np.array([40, 150, 255])),    # Light sand
    ],
    "road": [
        (np.array([0, 0, 60]), np.array([180, 30, 180])),       # Dark gray
    ],
}

# Aliases — map common query terms to our target categories
TARGET_ALIASES: dict[str, str] = {
    "water": "water",
    "river": "water",
    "lake": "water",
    "ocean": "water",
    "sea": "water",
    "pond": "water",
    "flood": "water",
    "vegetation": "vegetation",
    "forest": "vegetation",
    "tree": "vegetation",
    "trees": "vegetation",
    "green": "vegetation",
    "crop": "vegetation",
    "crops": "vegetation",
    "farm": "vegetation",
    "farmland": "vegetation",
    "building": "building",
    "buildings": "building",
    "house": "building",
    "houses": "building",
    "structure": "building",
    "structures": "building",
    "urban": "urban",
    "city": "urban",
    "town": "urban",
    "soil": "soil",
    "dirt": "soil",
    "earth": "soil",
    "sand": "sand",
    "desert": "sand",
    "beach": "sand",
    "road": "road",
    "roads": "road",
    "street": "road",
    "path": "road",
}

# Minimum contour area as fraction of image area (filters noise)
MIN_CONTOUR_FRACTION = 0.001


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _resolve_target(target_object: str) -> str | None:
    """Map a user's target description to our known target category."""
    target_lower = target_object.lower().strip()

    # Direct match
    if target_lower in TARGET_ALIASES:
        return TARGET_ALIASES[target_lower]

    # Substring match — check if any known keyword appears in the query
    for keyword, category in TARGET_ALIASES.items():
        if keyword in target_lower:
            return category

    return None


def _threshold_and_contours(
    hsv_image: np.ndarray,
    color_ranges: list[tuple[np.ndarray, np.ndarray]],
    min_area: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Apply HSV thresholding and find contours."""
    combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv_image, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter by minimum area
    significant_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    return combined_mask, significant_contours


def _ensure_bgr(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert to BGR numpy array for OpenCV."""
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            arr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return arr
        arr = image
    else:
        raise ValueError(f"Unexpected image type: {type(image)}")

    # RGB → BGR for OpenCV
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    return arr


# ---------------------------------------------------------------------------
# Public handler
# ---------------------------------------------------------------------------

def grounding_handler(
    images: list[Image.Image | np.ndarray],
    target_object: Optional[str] = None,
    **kwargs,
) -> HandlerResult:
    """
    Localize a target object in the first image using color thresholding.

    PLACEHOLDER: This classical CV approach only works for color-based targets.
    It will be replaced with GeoChat's learned grounding post-fine-tuning.

    Parameters
    ----------
    images : list
        One or more images (only the first is used).
    target_object : str, optional
        What to look for (e.g., "water", "vegetation").

    Returns
    -------
    HandlerResult
    """
    bgr = _ensure_bgr(images[0])
    h, w = bgr.shape[:2]
    min_area = int(h * w * MIN_CONTOUR_FRACTION)

    # Resolve target to known category
    target = target_object or "vegetation"  # Default target
    category = _resolve_target(target)

    if category is None:
        # Unknown target — we can't do color-based detection
        conf, basis = grounding_confidence([], w, h)
        return HandlerResult(
            answer=(
                f'Could not localize "{target}" — this placeholder grounding '
                f"uses color thresholding and only supports: "
                f"{', '.join(sorted(TARGET_COLOR_RANGES.keys()))}. "
                f"A learned grounding model (GeoChat) will replace this."
            ),
            bounding_boxes=[],
            confidence=conf,
            confidence_basis=basis,
            model_used="classical_cv_hsv_thresholding",
        )

    color_ranges = TARGET_COLOR_RANGES[category]

    # Convert to HSV and detect
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask, contours = _threshold_and_contours(hsv, color_ranges, min_area)

    # Build bounding boxes from contours
    bboxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        bboxes.append(BoundingBox(x=x, y=y, width=bw, height=bh, label=category))

    # Confidence scoring
    conf, basis = grounding_confidence(bboxes, w, h)

    if bboxes:
        answer = (
            f'Detected {len(bboxes)} region(s) matching "{target}" '
            f"({category} color range) via HSV thresholding."
        )
    else:
        answer = (
            f'No regions matching "{target}" ({category} color range) '
            f"were detected via HSV thresholding. The target may not have "
            f"a strong color signature in this image."
        )

    return HandlerResult(
        answer=answer,
        bounding_boxes=bboxes,
        confidence=conf,
        confidence_basis=basis,
        model_used="classical_cv_hsv_thresholding",
    )
