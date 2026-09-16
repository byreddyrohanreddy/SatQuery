"""
Input Validator for SatQuery AI.

Accepts 1–2 images, detects format and modality (optical vs SAR) using simple
heuristics (band count + value distribution), and checks scene compatibility
for image pairs.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from satquery.models import ValidationResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format detection helpers
# ---------------------------------------------------------------------------

# Try to import rasterio for GeoTIFF support; fall back gracefully
try:
    import rasterio  # type: ignore

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import cv2  # type: ignore

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _array_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Normalize numeric arrays (e.g. uint16/float GeoTIFF) using a 2%-98% percentile
    cumulative contrast stretch per channel, which is the remote-sensing standard
    to preserve vivid color saturation without atmospheric glare/cloud washout.
    """
    if arr.dtype != np.uint8:
        if arr.ndim == 2:
            p2, p98 = np.percentile(arr.astype(float), (2, 98))
            if p98 > p2:
                clipped = np.clip(arr.astype(float), p2, p98)
                norm = ((clipped - p2) / (p98 - p2) * 255.0).astype(np.uint8)
            else:
                norm = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.ndim == 3:
            norm = np.zeros_like(arr, dtype=np.uint8)
            for c in range(arr.shape[2]):
                ch = arr[:, :, c].astype(float)
                p2, p98 = np.percentile(ch, (2, 98))
                if p98 > p2:
                    clipped = np.clip(ch, p2, p98)
                    norm[:, :, c] = ((clipped - p2) / (p98 - p2) * 255.0).astype(np.uint8)
                else:
                    norm[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
        else:
            norm = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        norm = arr

    if norm.ndim == 2:
        return Image.fromarray(norm, mode="L")
    elif norm.shape[2] == 3:
        if HAS_CV2:
            return Image.fromarray(cv2.cvtColor(norm, cv2.COLOR_BGR2RGB), mode="RGB")
        return Image.fromarray(norm, mode="RGB")
    elif norm.shape[2] == 4:
        if HAS_CV2:
            return Image.fromarray(cv2.cvtColor(norm, cv2.COLOR_BGRA2RGBA), mode="RGBA")
        return Image.fromarray(norm, mode="RGBA")
    else:
        return Image.fromarray(norm[:, :, :3], mode="RGB")


def load_image_as_pil(source: Union[str, Path, bytes, io.BytesIO], filename: str = "") -> tuple[Image.Image, str]:
    """
    Robust image loader that returns a PIL Image in (RGB or L) mode,
    supporting standard formats, 16-bit TIFFs, and GeoTIFFs.
    """
    raw_bytes: bytes | None = None
    if isinstance(source, (str, Path)):
        p = Path(source)
        filename = filename or p.name
        try:
            with open(p, "rb") as f:
                raw_bytes = f.read()
        except Exception:
            raw_bytes = None
    elif isinstance(source, io.BytesIO):
        raw_bytes = source.getvalue()
    elif isinstance(source, bytes):
        raw_bytes = source

    suffix = Path(filename).suffix.lower() if filename else ""

    # 1. Try rasterio if available (GeoTIFF)
    if HAS_RASTERIO and suffix in (".tif", ".tiff") and raw_bytes is not None:
        try:
            with rasterio.open(io.BytesIO(raw_bytes)) as src:
                data = src.read()
                if data.shape[0] == 1:
                    arr = data[0]
                else:
                    arr = np.moveaxis(data, 0, -1)
                return _array_to_pil(arr), "GeoTIFF"
        except Exception:
            pass

    # 2. Try Pillow
    if raw_bytes is not None:
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            img.load()
            fmt = img.format or suffix.lstrip(".").upper() or "IMAGE"
            if img.mode not in ("RGB", "L", "RGBA"):
                img = img.convert("RGB")
            return img, fmt
        except Exception:
            pass
    elif isinstance(source, (str, Path)):
        try:
            img = Image.open(str(source))
            img.load()
            fmt = img.format or suffix.lstrip(".").upper() or "IMAGE"
            if img.mode not in ("RGB", "L", "RGBA"):
                img = img.convert("RGB")
            return img, fmt
        except Exception:
            pass

    # 3. Try OpenCV for TIFF / GeoTIFF or formats Pillow cannot parse
    if HAS_CV2:
        try:
            if raw_bytes is not None:
                np_buf = np.frombuffer(raw_bytes, np.uint8)
                arr = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)
                # If OpenCV decoded as 2D grayscale, verify if it actually has distinct color channels
                if arr is not None and arr.ndim == 2:
                    arr_color = cv2.imdecode(np_buf, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                    if arr_color is not None and arr_color.ndim == 3 and arr_color.shape[2] == 3:
                        b, g, r = arr_color[:, :, 0], arr_color[:, :, 1], arr_color[:, :, 2]
                        if not (np.array_equal(b, g) and np.array_equal(g, r)):
                            arr = arr_color
            elif isinstance(source, (str, Path)):
                arr = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
                if arr is not None and arr.ndim == 2:
                    arr_color = cv2.imread(str(source), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
                    if arr_color is not None and arr_color.ndim == 3 and arr_color.shape[2] == 3:
                        b, g, r = arr_color[:, :, 0], arr_color[:, :, 1], arr_color[:, :, 2]
                        if not (np.array_equal(b, g) and np.array_equal(g, r)):
                            arr = arr_color
            else:
                arr = None

            if arr is not None:
                return _array_to_pil(arr), "GeoTIFF"
        except Exception:
            pass

    raise ValueError(f"Unable to read image format for: {filename or 'input'}")


def _load_image_from_path(path: Union[str, Path]) -> tuple[np.ndarray, str]:
    """
    Load an image from a file path.

    Returns (pixel_array, format_string).
    pixel_array shape: (H, W, C) for multi-band, (H, W) for single-band.
    """
    path = Path(path)
    try:
        img, fmt = load_image_as_pil(path)
        return np.array(img), fmt
    except Exception:
        img = Image.open(str(path))
        fmt = img.format or path.suffix.lstrip(".").upper() or "UNKNOWN"
        return np.array(img), fmt


def _load_image_from_bytes(data: bytes, filename: str = "") -> tuple[np.ndarray, str]:
    """Load an image from raw bytes (e.g. from an upload)."""
    try:
        img, fmt = load_image_as_pil(data, filename=filename)
        return np.array(img), fmt
    except Exception:
        img = Image.open(io.BytesIO(data))
        fmt = img.format or Path(filename).suffix.lstrip(".").upper() or "UNKNOWN"
        return np.array(img), fmt


def _load_image(source: Union[str, Path, bytes, Image.Image, np.ndarray],
                filename: str = "") -> tuple[np.ndarray, str]:
    """
    Unified image loader — accepts file path, raw bytes, PIL Image, or numpy array.
    """
    if isinstance(source, np.ndarray):
        return source, "NUMPY"
    if isinstance(source, Image.Image):
        return np.array(source), source.format or "PIL"
    if isinstance(source, bytes):
        return _load_image_from_bytes(source, filename)
    # str or Path
    return _load_image_from_path(source)


# ---------------------------------------------------------------------------
# Modality heuristic
# ---------------------------------------------------------------------------

def _detect_modality(arr: np.ndarray) -> str:
    """
    Classify an image as 'optical' or 'SAR' using a simple heuristic.

    Optical:
      - 3+ channels (RGB or more)
      - Pixel values in typical photo range with moderate spread

    SAR:
      - 1–2 channels (grayscale intensity)
      - High coefficient of variation (std/mean) indicating speckle noise
      - OR high kurtosis indicating heavy-tailed distribution

    This is a rough heuristic, not a trained classifier.
    """
    # Determine number of bands
    if arr.ndim == 2:
        n_bands = 1
        flat = arr.astype(np.float64).ravel()
    elif arr.ndim == 3:
        n_bands = arr.shape[2]
        # For modality detection, analyze the first channel
        flat = arr[:, :, 0].astype(np.float64).ravel()
    else:
        return "optical"  # Unusual shape — default to optical

    # Band count rule: 3+ bands → almost certainly optical
    if n_bands >= 3:
        return "optical"

    # For 1–2 band images, check value distribution for speckle characteristics
    mean_val = np.mean(flat)
    std_val = np.std(flat)

    if mean_val == 0:
        return "SAR"  # All-zero image, ambiguous — label as SAR

    # Coefficient of variation — SAR images tend to have high CV due to speckle
    cv = std_val / mean_val if mean_val > 0 else 0

    # Kurtosis (excess) — SAR speckle has heavy tails
    if std_val > 0:
        kurtosis = np.mean(((flat - mean_val) / std_val) ** 4) - 3.0
    else:
        kurtosis = 0.0

    # SAR heuristic: high coefficient of variation OR high kurtosis
    if cv > 0.5 or kurtosis > 3.0:
        return "SAR"

    return "optical"


# ---------------------------------------------------------------------------
# Scene compatibility check
# ---------------------------------------------------------------------------

def _check_scene_compatibility(dims: list[tuple[int, int]]) -> list[str]:
    """
    Check whether two images are plausibly of the same scene.
    Returns a list of warnings (empty if compatible).
    """
    warnings = []
    if len(dims) != 2:
        return warnings

    (w1, h1), (w2, h2) = dims

    # Same dimensions — definitely compatible
    if w1 == w2 and h1 == h2:
        return warnings

    # Check aspect ratio similarity
    ar1 = w1 / max(h1, 1)
    ar2 = w2 / max(h2, 1)
    ratio_diff = abs(ar1 - ar2) / max(ar1, ar2, 1e-6)

    if ratio_diff > 0.15:
        warnings.append(
            f"Image dimensions differ significantly: {w1}x{h1} vs {w2}x{h2} "
            f"(aspect ratio difference: {ratio_diff:.1%}). "
            f"These may not be the same scene."
        )
    elif (w1, h1) != (w2, h2):
        warnings.append(
            f"Image dimensions differ: {w1}x{h1} vs {w2}x{h2}. "
            f"Aspect ratios are compatible — images will be resized as needed."
        )

    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_inputs(
    images: list[Union[str, Path, bytes, Image.Image, np.ndarray]],
    filenames: list[str] | None = None,
) -> ValidationResult:
    """
    Validate 1–2 input images.

    Parameters
    ----------
    images : list
        1 or 2 images as file paths, raw bytes, PIL Images, or numpy arrays.
    filenames : list[str], optional
        Original filenames (useful when images are passed as bytes).

    Returns
    -------
    ValidationResult
        Structured validation output including modality detection and warnings.
    """
    if filenames is None:
        filenames = ["" for _ in images]

    warnings: list[str] = []

    # ── Count check ──────────────────────────────────────────────────────
    if not images:
        return ValidationResult(
            image_count=0,
            modalities=[],
            formats=[],
            dimensions=[],
            warnings=["No images provided"],
            is_valid=False,
        )

    if len(images) > 2:
        warnings.append(
            f"Received {len(images)} images but only 1 or 2 are supported. "
            f"Using the first two."
        )
        images = images[:2]
        filenames = filenames[:2]

    # ── Load and analyze each image ──────────────────────────────────────
    modalities: list[str] = []
    formats: list[str] = []
    dimensions: list[tuple[int, int]] = []

    for i, (img_source, fname) in enumerate(zip(images, filenames)):
        try:
            arr, fmt = _load_image(img_source, fname)
        except Exception as e:
            return ValidationResult(
                image_count=len(images),
                modalities=modalities,
                formats=formats,
                dimensions=dimensions,
                warnings=[f"Failed to load image {i + 1}: {e}"],
                is_valid=False,
            )

        h, w = arr.shape[:2]
        dimensions.append((w, h))
        formats.append(fmt)
        modalities.append(_detect_modality(arr))

    # ── Scene compatibility (pairs only) ─────────────────────────────────
    if len(dimensions) == 2:
        warnings.extend(_check_scene_compatibility(dimensions))

    return ValidationResult(
        image_count=len(images),
        modalities=modalities,
        formats=formats,
        dimensions=dimensions,
        warnings=warnings,
        is_valid=True,
    )
