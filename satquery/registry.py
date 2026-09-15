"""
Model Registry for SatQuery AI.

A clean, commented mapping from task names to handler functions and metadata.
This file is intentionally simple and readable — it's part of the "auditable
architecture" story.

Adding a new handler:
  1. Implement it in satquery/handlers/
  2. Add an entry to REGISTRY below
  3. Add a matching tool definition in router.py (for LLM routing)
  4. Add a keyword rule in router.py (for fallback routing)
"""

from __future__ import annotations

from typing import Any, Callable

from satquery.handlers.change_analysis import change_analysis_handler
from satquery.handlers.fusion import fusion_handler
from satquery.handlers.grounding import grounding_handler
from satquery.handlers.vqa_caption import vqa_caption_handler

# ---------------------------------------------------------------------------
# Registry type
# ---------------------------------------------------------------------------

RegistryEntry = dict[str, Any]

# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, RegistryEntry] = {
    # ── Visual Question Answering / Image Captioning ─────────────────────
    "vqa_caption": {
        "handler": vqa_caption_handler,
        "requires_images": 1,       # Minimum number of images needed
        "max_images": 1,            # Maximum useful images
        "description": (
            "Generate a caption for a satellite image, or answer a specific "
            "question about it. Uses BLIP (Salesforce) models running locally."
        ),
        "models": [
            "Salesforce/blip-image-captioning-base",
            "Salesforce/blip-vqa-base",
        ],
        "argument_keys": ["question"],  # Arguments this handler accepts
    },

    # ── Object Grounding / Localization ──────────────────────────────────
    "grounding": {
        "handler": grounding_handler,
        "requires_images": 1,
        "max_images": 1,
        "description": (
            "Localize a target object in a satellite image using color-based "
            "thresholding. Returns bounding box(es). PLACEHOLDER for GeoChat's "
            "native grounding — currently limited to color-prominent targets "
            "(water, vegetation, buildings, soil, sand, roads)."
        ),
        "models": ["classical_cv_hsv_thresholding"],
        "argument_keys": ["target_object"],
    },

    # ── Bi-temporal Change Detection ─────────────────────────────────────
    "change_analysis": {
        "handler": change_analysis_handler,
        "requires_images": 2,
        "max_images": 2,
        "description": (
            "Detect and quantify changes between two images of the same area "
            "taken at different times. Uses image differencing with contour "
            "detection."
        ),
        "models": ["image_differencing"],
        "argument_keys": ["description"],
    },

    # ── Optical + SAR Fusion ─────────────────────────────────────────────
    "optical_sar_fusion": {
        "handler": fusion_handler,
        "requires_images": 2,
        "max_images": 2,
        "description": (
            "Fuse analysis of co-registered optical and SAR imagery. "
            "Runs captioning independently on each modality, then composes "
            "a combined answer. DOCUMENTED FALLBACK — a learned fusion model "
            "would replace this in production."
        ),
        "models": [
            "Salesforce/blip-image-captioning-base",
            "fusion_dual_caption",
        ],
        "argument_keys": ["question"],
    },
}


def get_handler(task_name: str) -> Callable:
    """Get the handler function for a task. Raises KeyError if not found."""
    if task_name not in REGISTRY:
        raise KeyError(
            f"Unknown task '{task_name}'. "
            f"Available tasks: {', '.join(REGISTRY.keys())}"
        )
    return REGISTRY[task_name]["handler"]


def get_entry(task_name: str) -> RegistryEntry:
    """Get the full registry entry for a task."""
    if task_name not in REGISTRY:
        raise KeyError(f"Unknown task '{task_name}'")
    return REGISTRY[task_name]


def list_tasks() -> list[str]:
    """List all registered task names."""
    return list(REGISTRY.keys())
