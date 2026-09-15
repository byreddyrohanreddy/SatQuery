"""
Optical + SAR Fusion handler for SatQuery AI.

╔═══════════════════════════════════════════════════════════════════════╗
║  DOCUMENTED FALLBACK FUSION STRATEGY                                 ║
║                                                                      ║
║  Runs the vqa_caption handler independently on each of the two       ║
║  images (one optical, one SAR), then composes a combined answer.     ║
║                                                                      ║
║  This is the documented baseline approach — a proper fusion model    ║
║  (e.g., feature-level fusion or cross-attention) would replace this  ║
║  in a production system.                                             ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

from satquery.confidence import fusion_confidence
from satquery.handlers.vqa_caption import vqa_caption_handler
from satquery.models import HandlerResult

logger = logging.getLogger(__name__)


def fusion_handler(
    images: list[Image.Image | np.ndarray],
    question: Optional[str] = None,
    modalities: list[str] | None = None,
    **kwargs,
) -> HandlerResult:
    """
    Fuse analysis of optical and SAR imagery.

    Runs captioning/VQA independently on each image, then composes
    a combined answer.

    Parameters
    ----------
    images : list
        Exactly 2 images (optical + SAR pair).
    question : str, optional
        A question to answer about the scene.
    modalities : list[str], optional
        Detected modalities per image (e.g., ["optical", "SAR"]).
        Used for labeling in the composed answer.

    Returns
    -------
    HandlerResult
    """
    if len(images) < 2:
        return HandlerResult(
            answer="Optical+SAR fusion requires exactly 2 images.",
            bounding_boxes=[],
            confidence=0.0,
            confidence_basis="Insufficient inputs",
            model_used="fusion_dual_caption",
        )

    # Determine labels for each image
    if modalities and len(modalities) >= 2:
        label_1 = modalities[0].upper()
        label_2 = modalities[1].upper()
    else:
        label_1 = "Image 1"
        label_2 = "Image 2"

    # Run VQA/captioning on each image independently
    logger.info("Running captioning on %s image...", label_1)
    result_1 = vqa_caption_handler([images[0]], question=question)

    logger.info("Running captioning on %s image...", label_2)
    result_2 = vqa_caption_handler([images[1]], question=question)

    # Compose combined answer
    if question:
        combined_answer = (
            f"{label_1} analysis: {result_1.answer}. "
            f"{label_2} analysis: {result_2.answer}."
        )
    else:
        combined_answer = (
            f"Optical imagery shows: {result_1.answer}. "
            f"SAR imagery indicates: {result_2.answer}."
        )

    # Combined confidence
    conf, basis = fusion_confidence(result_1.confidence, result_2.confidence)

    # Merge any bounding boxes from both results
    all_bboxes = result_1.bounding_boxes + result_2.bounding_boxes

    return HandlerResult(
        answer=combined_answer,
        bounding_boxes=all_bboxes,
        confidence=conf,
        confidence_basis=basis,
        model_used="fusion_dual_caption",
    )
