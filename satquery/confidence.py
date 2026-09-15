"""
Confidence scoring heuristics for SatQuery AI handlers.

Each function returns a (score, basis) tuple where:
  - score is a float in [0, 1]
  - basis is a one-line human-readable explanation

These are clearly-labeled heuristics — they do not fabricate precision.
"""

from __future__ import annotations

from satquery.models import BoundingBox


def caption_confidence(is_vqa: bool = False) -> tuple[float, str]:
    """
    Confidence for the VQA/caption handler.

    Fixed values — BLIP base models are not fine-tuned for satellite imagery,
    so we give an honest moderate-confidence score.
    """
    if is_vqa:
        return (
            0.55,
            "BLIP-VQA base model; not fine-tuned for satellite imagery, "
            "answers may be generic",
        )
    return (
        0.65,
        "BLIP captioning base model; produces reasonable generic captions "
        "but is not trained on satellite data",
    )


def grounding_confidence(
    bounding_boxes: list[BoundingBox],
    image_width: int,
    image_height: int,
) -> tuple[float, str]:
    """
    Confidence for the grounding handler.

    Based on the proportion of the image covered by detected regions.
    Very small or very large detections are less trustworthy.
    """
    if not bounding_boxes:
        return (
            0.10,
            "No regions detected by color thresholding; target may not match "
            "predefined color ranges",
        )

    total_area = image_width * image_height
    detected_area = sum(bb.width * bb.height for bb in bounding_boxes)
    coverage = detected_area / max(total_area, 1)

    # Sweet spot: 1%–50% coverage is most plausible
    if 0.01 <= coverage <= 0.50:
        score = 0.40 + coverage * 0.8  # 0.41 – 0.80
    elif coverage > 0.50:
        score = max(0.30, 0.80 - (coverage - 0.50) * 0.6)  # Drops for huge areas
    else:
        score = 0.20  # Very tiny detections

    score = max(0.10, min(0.85, score))

    return (
        round(score, 2),
        f"Classical CV color thresholding; {coverage:.1%} of image area matched "
        f"({len(bounding_boxes)} region(s) found)",
    )


def change_confidence(change_percentage: float) -> tuple[float, str]:
    """
    Confidence for the change analysis handler.

    Changes in the 2%–60% range are most plausible for real-world change
    detection. Outside that range, results are less trustworthy (noise or
    completely different images).
    """
    if 2.0 <= change_percentage <= 60.0:
        score = 0.70
        note = "in expected range for meaningful scene changes"
    elif change_percentage < 2.0:
        score = 0.45
        note = "very low — may be noise or identical images"
    else:
        score = 0.35
        note = "very high — images may be completely different scenes"

    return (
        score,
        f"Image differencing heuristic; {change_percentage:.1f}% of pixels changed, "
        f"{note}",
    )


def fusion_confidence(
    optical_confidence: float,
    sar_confidence: float,
) -> tuple[float, str]:
    """
    Confidence for the optical+SAR fusion handler.

    Average of the two sub-caption confidences, reduced by 10% to account
    for cross-modal uncertainty in the composition step.
    """
    avg = (optical_confidence + sar_confidence) / 2.0
    score = round(avg * 0.90, 2)

    return (
        score,
        f"Average of optical ({optical_confidence:.2f}) and SAR ({sar_confidence:.2f}) "
        f"caption confidences, reduced 10% for cross-modal composition uncertainty",
    )
