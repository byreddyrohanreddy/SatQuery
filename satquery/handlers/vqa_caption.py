"""
VQA / Captioning handler for SatQuery AI.

Uses Hugging Face `transformers` with pre-trained BLIP models:
  - Salesforce/blip-image-captioning-base  — for image captioning
  - Salesforce/blip-vqa-base               — for visual question answering

Models are loaded once (lazy singleton) and reused across requests.
Runs on CPU — no GPU required.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image

from satquery.confidence import caption_confidence
from satquery.models import HandlerResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy model singletons — loaded on first use, then cached
# ---------------------------------------------------------------------------

_caption_model = None
_caption_processor = None
_vqa_model = None
_vqa_processor = None


def _get_caption_model():
    """Load the BLIP captioning model (lazy, once)."""
    global _caption_model, _caption_processor
    if _caption_model is None:
        logger.info("Loading BLIP captioning model (first call — this takes a moment)...")
        from transformers import BlipForConditionalGeneration, BlipProcessor

        model_name = "Salesforce/blip-image-captioning-base"
        _caption_processor = BlipProcessor.from_pretrained(model_name)
        _caption_model = BlipForConditionalGeneration.from_pretrained(
            model_name, use_safetensors=True
        )
        _caption_model.eval()
        logger.info("BLIP captioning model loaded.")
    return _caption_model, _caption_processor


def _get_vqa_model():
    """Load the BLIP VQA model (lazy, once)."""
    global _vqa_model, _vqa_processor
    if _vqa_model is None:
        logger.info("Loading BLIP VQA model (first call — this takes a moment)...")
        from transformers import BlipForQuestionAnswering, BlipProcessor

        model_name = "Salesforce/blip-vqa-base"
        _vqa_processor = BlipProcessor.from_pretrained(model_name)
        _vqa_model = BlipForQuestionAnswering.from_pretrained(
            model_name, use_safetensors=True
        )
        _vqa_model.eval()
        logger.info("BLIP VQA model loaded.")
    return _vqa_model, _vqa_processor


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ensure_pil(image: Image.Image | np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image if needed."""
    if isinstance(image, np.ndarray):
        # Handle single-channel images
        if image.ndim == 2:
            return Image.fromarray(image, mode="L").convert("RGB")
        if image.ndim == 3 and image.shape[2] == 1:
            return Image.fromarray(image[:, :, 0], mode="L").convert("RGB")
        if image.ndim == 3 and image.shape[2] == 4:
            return Image.fromarray(image, mode="RGBA").convert("RGB")
        return Image.fromarray(image, mode="RGB")
    return image.convert("RGB")


# ---------------------------------------------------------------------------
# Public handler
# ---------------------------------------------------------------------------

def vqa_caption_handler(
    images: list[Image.Image | np.ndarray],
    question: Optional[str] = None,
    **kwargs,
) -> HandlerResult:
    """
    Run BLIP captioning or VQA on the first image.

    Parameters
    ----------
    images : list
        One or more images (only the first is used).
    question : str, optional
        If provided, runs VQA. Otherwise runs captioning.

    Returns
    -------
    HandlerResult
    """
    import torch

    pil_image = _ensure_pil(images[0])

    if question and question.strip():
        # ── Visual Question Answering ────────────────────────────────────
        model, processor = _get_vqa_model()
        inputs = processor(pil_image, question, return_tensors="pt")

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=50)

        answer = processor.decode(output_ids[0], skip_special_tokens=True).strip()
        conf, basis = caption_confidence(is_vqa=True)

        return HandlerResult(
            answer=answer,
            bounding_boxes=[],
            confidence=conf,
            confidence_basis=basis,
            model_used="Salesforce/blip-vqa-base",
        )
    else:
        # ── Image Captioning ────────────────────────────────────────────
        model, processor = _get_caption_model()
        inputs = processor(pil_image, return_tensors="pt")

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=100)

        caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
        conf, basis = caption_confidence(is_vqa=False)

        return HandlerResult(
            answer=caption,
            bounding_boxes=[],
            confidence=conf,
            confidence_basis=basis,
            model_used="Salesforce/blip-image-captioning-base",
        )
