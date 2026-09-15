"""
FastAPI layer for SatQuery AI.

Orchestration pipeline API and integrated web frontend:
  - POST /query  — accepts images + text query, returns full analysis + trace
  - GET  /health — system status including routing mode
  - GET  /       — serves index.html (frontend UI)
  - GET  /samples — serves sample satellite imagery
"""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from satquery.models import (
    BoundingBox,
    ClarificationResponse,
    HealthResponse,
    NormalizedBox,
    QueryResponse,
    ValidationResult,
    VisualEvidence,
)
from satquery.registry import get_entry, get_handler
from satquery.router import route_query
from satquery.trace import TraceBuilder
from satquery.validator import load_image_as_pil, validate_inputs

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env if present
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    try:
        with open(_env_path, encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))
    except Exception:
        pass

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SatQuery AI",
    description=(
        "Agentic satellite imagery query system. "
        "Accepts images + natural-language queries, routes to specialist "
        "handlers, returns analysis with full execution trace."
    ),
    version="0.1.0",
)

# Enable CORS for flexible local development and preview servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount samples folder for Quick Demo buttons
samples_dir = PROJECT_ROOT / "samples"
if samples_dir.exists():
    app.mount("/samples", StaticFiles(directory=str(samples_dir)), name="samples")


# ---------------------------------------------------------------------------
# Frontend Static Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def serve_index():
    """Serve the upload and query frontend."""
    index_path = PROJECT_ROOT / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "SatQuery AI API is running. index.html not found."}


@app.get("/results.html", include_in_schema=False)
async def serve_results():
    """Serve the results page."""
    results_path = PROJECT_ROOT / "results.html"
    if results_path.exists():
        return FileResponse(str(results_path))
    raise HTTPException(status_code=404, detail="results.html not found")


@app.get("/style.css", include_in_schema=False)
async def serve_style():
    """Serve the shared stylesheet."""
    style_path = PROJECT_ROOT / "style.css"
    if style_path.exists():
        return FileResponse(str(style_path), media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")


@app.get("/script.js", include_in_schema=False)
async def serve_script():
    """Serve the shared frontend JavaScript."""
    script_path = PROJECT_ROOT / "script.js"
    if script_path.exists():
        return FileResponse(str(script_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="script.js not found")


# ---------------------------------------------------------------------------
# Core pipeline (usable as a Python API without FastAPI)
# ---------------------------------------------------------------------------

def run_pipeline(
    images: list[Image.Image | np.ndarray],
    query: str,
    filenames: list[str] | None = None,
    modalities_override: list[str] | None = None,
) -> QueryResponse | ClarificationResponse:
    """
    Run the full SatQuery AI pipeline.

    This is the main Python API — the FastAPI endpoints are thin wrappers
    around this function.

    Parameters
    ----------
    images : list
        1 or 2 images as PIL Images or numpy arrays.
    query : str
        Natural-language query.
    filenames : list[str], optional
        Original filenames for format detection.
    modalities_override : list[str], optional
        Override auto-detected modalities (useful for testing).

    Returns
    -------
    QueryResponse or ClarificationResponse
    """
    trace = TraceBuilder()

    # ── Step 1: Validate inputs ──────────────────────────────────────────
    validation = validate_inputs(images, filenames=filenames)
    trace.set_validation(validation)

    if not validation.is_valid:
        raise ValueError(
            f"Invalid input: {'; '.join(validation.warnings)}"
        )

    # Apply modality override if provided
    if modalities_override:
        validation = validation.model_copy(
            update={"modalities": modalities_override}
        )

    # ── Step 2: Route query ──────────────────────────────────────────────
    routing = route_query(query, validation)
    trace.set_routing(
        task_name=routing.task_name,
        routing_path=routing.routing_path,
        fallback_reason=routing.fallback_reason,
    )

    # ── Step 3: Check input compatibility ────────────────────────────────
    entry = get_entry(routing.task_name)
    required_images = entry["requires_images"]

    if validation.image_count < required_images:
        return ClarificationResponse(
            message=(
                f"The selected task '{routing.task_name}' requires at least "
                f"{required_images} image(s), but only {validation.image_count} "
                f"was provided. Please provide {required_images} image(s)."
            ),
            task_attempted=routing.task_name,
            routing_path=routing.routing_path,
            fallback_reason=routing.fallback_reason,
        )

    # ── Step 4: Run handler ──────────────────────────────────────────────
    handler = get_handler(routing.task_name)

    # Build handler kwargs from routing arguments + metadata
    handler_kwargs = dict(routing.task_arguments)
    handler_kwargs["modalities"] = validation.modalities

    result = handler(images=images, **handler_kwargs)

    # ── Step 5: Build trace ──────────────────────────────────────────────
    trace.set_handler_info(
        models=[result.model_used],
        parameters=routing.task_arguments,
    )
    trace.set_confidence(result.confidence, result.confidence_basis)
    execution_trace = trace.build()

    # ── Step 6: Build visual evidence ────────────────────────────────────
    ev_type = None
    ev_boxes: list[NormalizedBox] = []
    if result.bounding_boxes:
        if routing.task_name == "change_analysis":
            ev_type = "change_regions"
        else:
            ev_type = "bounding_box"

        img_w, img_h = validation.dimensions[0] if validation.dimensions else (1, 1)
        img_w = max(img_w, 1)
        img_h = max(img_h, 1)

        for b in result.bounding_boxes:
            ev_boxes.append(
                NormalizedBox(
                    x=round(float(b.x) / img_w, 4),
                    y=round(float(b.y) / img_h, 4),
                    width=round(float(b.width) / img_w, 4),
                    height=round(float(b.height) / img_h, 4),
                    label=b.label,
                )
            )

    visual_evidence = VisualEvidence(type=ev_type, boxes=ev_boxes)

    # ── Step 7: Generate base64 JPEG previews for frontend display ───────
    image_previews: list[str] = []
    for img in images:
        try:
            pil_thumb = img if isinstance(img, Image.Image) else Image.fromarray(img)
            if pil_thumb.mode not in ("RGB", "L"):
                pil_thumb = pil_thumb.convert("RGB")
            # Downsample if very large (>1024) to keep response snappy
            w, h = pil_thumb.size
            if max(w, h) > 1024:
                scale = 1024.0 / max(w, h)
                pil_thumb = pil_thumb.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            if pil_thumb.mode == "L":
                pil_thumb = pil_thumb.convert("RGB")
            pil_thumb.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            image_previews.append(f"data:image/jpeg;base64,{b64}")
        except Exception as e:
            logger.warning("Failed to generate preview for image: %s", e)

    # ── Step 8: Build response ───────────────────────────────────────────
    return QueryResponse(
        answer=result.answer,
        visual_evidence=visual_evidence,
        confidence=result.confidence,
        confidence_basis=result.confidence_basis,
        execution_trace=execution_trace,
        image_previews=image_previews,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """System health check — also reports routing mode and active LLM provider."""
    gemini_set = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    anthropic_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_llm = gemini_set or anthropic_set
    provider = "gemini" if gemini_set else ("anthropic" if anthropic_set else None)
    return HealthResponse(
        status="ok",
        anthropic_api_key_set=has_llm,
        routing_mode=f"llm ({provider})" if has_llm else "fallback",
    )


@app.post("/query")
async def query_endpoint(
    query: str = Form(..., description="Natural-language query about the image(s)"),
    image1: Optional[UploadFile] = File(None, description="Image 1 file from frontend"),
    image2: Optional[UploadFile] = File(None, description="Image 2 file from frontend"),
    files: Optional[list[UploadFile]] = File(None, description="List of files (alternative)"),
):
    """
    Analyze satellite imagery with a natural-language query.

    Accepts 1–2 images (via image1/image2 form fields or files) + text query.
    Returns analysis result with full execution trace.
    """
    upload_list: list[UploadFile] = []
    if image1 and image1.filename:
        upload_list.append(image1)
    if image2 and image2.filename:
        upload_list.append(image2)
    if files:
        for f in files:
            if f and f.filename:
                upload_list.append(f)

    if not upload_list:
        raise HTTPException(status_code=400, detail="At least one image file is required")
    if len(upload_list) > 2:
        raise HTTPException(status_code=400, detail="At most 2 image files are supported")

    # Load images
    images = []
    filenames = []
    for f in upload_list:
        try:
            raw = await f.read()
            img, fmt = load_image_as_pil(raw, filename=f.filename or "")
            images.append(img)
            filenames.append(f.filename or "")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read image '{f.filename}': {e}",
            )

    # Run pipeline
    try:
        result = run_pipeline(images, query, filenames=filenames)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # Return appropriate response type
    if isinstance(result, ClarificationResponse):
        raise HTTPException(status_code=400, detail=result.message)
    return result.model_dump()
