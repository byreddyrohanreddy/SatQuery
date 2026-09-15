"""
Pydantic models for the SatQuery AI pipeline.

All shared data structures used across the system: validation results,
handler outputs, execution traces, and API responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Output of the input validator."""

    image_count: int = Field(..., description="Number of images provided (1 or 2)")
    modalities: list[str] = Field(
        ...,
        description='Detected modality per image: "optical" or "SAR"',
    )
    formats: list[str] = Field(
        ...,
        description="Detected file format per image (e.g. JPEG, PNG, TIFF)",
    )
    dimensions: list[tuple[int, int]] = Field(
        ...,
        description="(width, height) per image",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings (e.g. dimension mismatch between image pair)",
    )
    is_valid: bool = Field(..., description="Whether the input set is usable")


# ---------------------------------------------------------------------------
# Handler output
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """A bounding box in pixel coordinates."""

    x: int = Field(..., description="Left edge (pixels)")
    y: int = Field(..., description="Top edge (pixels)")
    width: int = Field(..., description="Box width (pixels)")
    height: int = Field(..., description="Box height (pixels)")
    label: Optional[str] = Field(None, description="What this box represents")


class HandlerResult(BaseModel):
    """Structured output from any specialist handler."""

    answer: str = Field(..., description="Natural-language answer or description")
    bounding_boxes: list[BoundingBox] = Field(
        default_factory=list,
        description="Detected regions, if applicable",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score (0–1)",
    )
    confidence_basis: str = Field(
        ...,
        description="One-line explanation of how confidence was derived",
    )
    model_used: str = Field(
        ...,
        description="Name/identifier of the model or algorithm used",
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class RoutingDecision(BaseModel):
    """The task router's output: which task to run and how it was decided."""

    task_name: str = Field(..., description="Selected handler task name")
    task_arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments extracted for the handler (e.g. target_object)",
    )
    routing_path: str = Field(
        ...,
        description='"llm_routed" or "fallback_routed"',
    )
    fallback_reason: Optional[str] = Field(
        None,
        description="Why fallback was used (e.g. 'no_api_key', 'timeout', 'malformed_response')",
    )


# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------

class ExecutionTrace(BaseModel):
    """
    Full audit trail for a single query execution.

    This is the single most important output of the system — it must be
    returned in full with every response, never truncated or hidden.
    """

    task_selected: str = Field(..., description="Handler task that was executed")
    routing_path: str = Field(
        ...,
        description='"llm_routed" or "fallback_routed"',
    )
    fallback_reason: Optional[str] = Field(
        None,
        description="Why fallback was used, if applicable",
    )
    validator_output: ValidationResult = Field(
        ...,
        description="Full validation result for the input",
    )
    models_called: list[str] = Field(
        ...,
        description="Model(s) or algorithm(s) actually invoked",
    )
    model_called: Optional[str] = Field(
        None,
        description="Single or comma-separated model string for UI display",
    )
    parameters_used: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters passed to the handler",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the query was processed (UTC)",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_basis: str = Field(
        ...,
        description="How the confidence score was derived",
    )


# ---------------------------------------------------------------------------
# Visual Evidence (UI friendly normalized overlay)
# ---------------------------------------------------------------------------

class NormalizedBox(BaseModel):
    """Normalized bounding box (0.0 - 1.0 coordinate range for responsive overlay)."""

    x: float = Field(..., description="Normalized top-left x (0 to 1)")
    y: float = Field(..., description="Normalized top-left y (0 to 1)")
    width: float = Field(..., description="Normalized width (0 to 1)")
    height: float = Field(..., description="Normalized height (0 to 1)")
    label: Optional[str] = None


class VisualEvidence(BaseModel):
    """Visual evidence container matching frontend overlay schema."""

    type: Optional[str] = Field(
        None,
        description="'bounding_box' or 'change_regions'",
    )
    boxes: list[NormalizedBox] = Field(
        default_factory=list,
        description="List of normalized bounding boxes",
    )


# ---------------------------------------------------------------------------
# API response
# ---------------------------------------------------------------------------

class QueryResponse(BaseModel):
    """Top-level response returned by POST /query."""

    answer: str = Field(..., description="Natural-language answer")
    visual_evidence: VisualEvidence | list[BoundingBox] = Field(
        default_factory=VisualEvidence,
        description="Bounding boxes highlighting relevant regions",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_basis: str
    execution_trace: ExecutionTrace
    image_previews: list[str] = Field(
        default_factory=list,
        description="Base64 JPEG data URLs for browser rendering (converts TIFF/GeoTIFF)",
    )

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "answer": "The image shows an urban area with dense buildings.",
                "visual_evidence": [],
                "confidence": 0.65,
                "confidence_basis": "BLIP base model confidence on generic imagery",
                "execution_trace": {
                    "task_selected": "vqa_caption",
                    "routing_path": "fallback_routed",
                    "fallback_reason": "no_api_key",
                    "validator_output": {
                        "image_count": 1,
                        "modalities": ["optical"],
                        "formats": ["JPEG"],
                        "dimensions": [(512, 512)],
                        "warnings": [],
                        "is_valid": True,
                    },
                    "models_called": [
                        "Salesforce/blip-image-captioning-base"
                    ],
                    "parameters_used": {},
                    "timestamp": "2026-01-01T00:00:00",
                    "confidence": 0.65,
                    "confidence_basis": "BLIP base model confidence on generic imagery",
                },
            }
        }
    )


class ClarificationResponse(BaseModel):
    """Returned when the selected task can't run with the provided inputs."""

    message: str = Field(..., description="What's wrong and what the user should do")
    task_attempted: str
    routing_path: str
    fallback_reason: Optional[str] = None


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = "ok"
    anthropic_api_key_set: bool
    routing_mode: str = Field(
        ...,
        description='"llm" if API key is set, "fallback" otherwise',
    )
