"""
Task Router for SatQuery AI.

The core "agentic" piece — routes a user's query + validated input metadata
to the correct specialist handler.

TWO ROUTING PATHS:

  PRIMARY  — Anthropic tool-calling via the anthropic SDK.
             Sends the query + input description to Claude with 4 tool
             definitions, lets the model choose which handler to invoke.
             Wrapped in a timeout + try/except.

  FALLBACK — Deterministic keyword rules.
             Fires automatically if the primary path times out, errors,
             returns no tool_use block, or if ANTHROPIC_API_KEY is not set.
             Requires NO network access and NO API key.

Both paths produce a RoutingDecision with full transparency about which
path was used and why.
"""

from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any, Optional

from satquery.models import RoutingDecision, ValidationResult

logger = logging.getLogger(__name__)

# Load .env if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
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
# Tool definitions for Anthropic tool-calling
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "vqa_caption",
        "description": (
            "Generate a caption describing a satellite image, or answer a "
            "specific question about what is visible in the image. Use this "
            "when the user wants a description or asks a question about a "
            "single image."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The specific question to answer about the image. "
                        "Leave empty/null for general captioning."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "grounding",
        "description": (
            "Localize and highlight a specific object or feature in a "
            "satellite image. Use when the user asks to find, locate, "
            "detect, or highlight something in the image. Returns bounding "
            "box coordinates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_object": {
                    "type": "string",
                    "description": (
                        "The object or feature to locate in the image, "
                        "e.g. 'water', 'vegetation', 'buildings'."
                    ),
                },
            },
            "required": ["target_object"],
        },
    },
    {
        "name": "change_analysis",
        "description": (
            "Analyze changes between two images of the same area taken at "
            "different times. Use when the user provides two bi-temporal "
            "images and asks about changes, differences, or what happened "
            "between the two time periods."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "Optional description of what kind of change to "
                        "look for, e.g. 'urban expansion', 'deforestation'."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "optical_sar_fusion",
        "description": (
            "Combine analysis of co-registered optical and SAR satellite "
            "imagery of the same area. Use when the user provides both an "
            "optical image and a SAR image and wants a combined analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Optional question to answer about the scene using "
                        "both modalities."
                    ),
                },
            },
            "required": [],
        },
    },
]

# Timeout for the LLM API call (seconds)
LLM_TIMEOUT_SECONDS = 8


# ---------------------------------------------------------------------------
# Primary path: Anthropic tool-calling
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    """System prompt for the routing LLM."""
    return (
        "You are a satellite imagery analysis router. Given a user's query "
        "and information about the images they've provided, select the most "
        "appropriate analysis tool and extract the relevant arguments.\n\n"
        "Available image modalities: optical (standard photo-like satellite "
        "imagery) and SAR (synthetic aperture radar, grayscale speckled images).\n\n"
        "IMPORTANT: Always select exactly one tool. Extract arguments from "
        "the user's query (e.g., the target object they want to find, or the "
        "specific question they're asking)."
    )


def _build_user_message(query: str, validation: ValidationResult) -> str:
    """Compose the user message with query + input metadata."""
    lines = [
        f"User query: {query}",
        "",
        "Input images:",
        f"  - Count: {validation.image_count}",
        f"  - Modalities: {', '.join(validation.modalities)}",
        f"  - Formats: {', '.join(validation.formats)}",
        f"  - Dimensions: {validation.dimensions}",
    ]
    if validation.warnings:
        lines.append(f"  - Warnings: {'; '.join(validation.warnings)}")
    return "\n".join(lines)


def _call_anthropic(query: str, validation: ValidationResult) -> RoutingDecision:
    """
    Call the Anthropic API with tool definitions and return a routing decision.
    Raises on any failure (timeout, API error, malformed response).
    """
    import anthropic

    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY from env

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        system=_build_system_prompt(),
        tools=TOOL_DEFINITIONS,
        messages=[
            {
                "role": "user",
                "content": _build_user_message(query, validation),
            }
        ],
    )

    # Find the tool_use block in the response
    for block in message.content:
        if block.type == "tool_use":
            return RoutingDecision(
                task_name=block.name,
                task_arguments=block.input or {},
                routing_path="llm_routed",
                fallback_reason=None,
            )

    # No tool_use block found
    raise ValueError("LLM response contained no tool_use block")


def _call_gemini(query: str, validation: ValidationResult, api_key: str) -> RoutingDecision:
    """
    Call Google Gemini API with function declarations and return a routing decision.
    Uses standard library urllib.request (zero extra dependencies).
    """
    import json
    import urllib.request

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"

    gemini_tools = [{
        "function_declarations": [
            {
                "name": "vqa_caption",
                "description": (
                    "Generate a caption describing a satellite image, or answer a "
                    "specific question about what is visible in the image. Use this "
                    "when the user wants a description or asks a question about a "
                    "single image."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "question": {
                            "type": "STRING",
                            "description": "Specific question about the image, or empty for general captioning.",
                        }
                    },
                },
            },
            {
                "name": "grounding",
                "description": (
                    "Localize and highlight a specific object, target, or feature in a "
                    "satellite image. Use when the user asks to find, locate, detect, "
                    "or highlight something. Returns bounding box coordinates."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "target_object": {
                            "type": "STRING",
                            "description": "Object or feature to detect (e.g. water, vegetation, building).",
                        }
                    },
                    "required": ["target_object"],
                },
            },
            {
                "name": "change_analysis",
                "description": (
                    "Detect, locate, and quantify changes between two satellite "
                    "images of the same area taken at different times (bi-temporal "
                    "change analysis)."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "description": {
                            "type": "STRING",
                            "description": "Description of what changed or what to look for.",
                        }
                    },
                },
            },
            {
                "name": "optical_sar_fusion",
                "description": (
                    "Analyze and fuse complementary information from co-registered optical "
                    "and SAR satellite imagery of the same area."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "question": {
                            "type": "STRING",
                            "description": "The question to answer using both optical and SAR imagery.",
                        }
                    },
                },
            },
        ]
    }]

    user_text = _build_user_message(query, validation)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "tools": gemini_tools,
        "tool_config": {"function_calling_config": {"mode": "ANY"}},
    }
    encoded_data = json.dumps(payload).encode("utf-8")

    models_to_try = []
    custom_model = os.environ.get("GEMINI_MODEL")
    if custom_model:
        models_to_try.append(custom_model.strip())
    for fallback_m in ["gemini-flash-lite-latest", "gemini-3.6-flash", "gemini-flash-latest"]:
        if fallback_m not in models_to_try:
            models_to_try.append(fallback_m)

    result = None
    last_err = None

    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=encoded_data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_SECONDS) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                break
        except urllib.error.HTTPError as e:
            last_err = e
            # If high demand (503), model retired (404), or rate-limited (429), try next model
            if e.code in (503, 404, 429):
                logger.warning("Gemini model %s returned HTTP %s, trying next candidate...", model, e.code)
                continue
            raise
        except Exception as e:
            last_err = e
            logger.warning("Gemini model %s call failed (%s), trying next candidate...", model, e)
            continue

    if result is None:
        raise last_err or ValueError("Failed to obtain response from Gemini")

    candidates = result.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini returned no candidates")

    parts = candidates[0].get("content", {}).get("parts", [])
    for part in parts:
        if "functionCall" in part:
            fc = part["functionCall"]
            task_name = fc.get("name")
            args = fc.get("args") or {}
            return RoutingDecision(
                task_name=task_name,
                task_arguments=args,
                routing_path="llm_routed",
                fallback_reason=None,
            )

    raise ValueError("Gemini response contained no function call")


def _try_llm_routing(query: str, validation: ValidationResult) -> tuple[Optional[RoutingDecision], Optional[str]]:
    """
    Attempt LLM-based routing with timeout protection.
    Supports Gemini (GEMINI_API_KEY / GOOGLE_API_KEY) and Anthropic (ANTHROPIC_API_KEY).
    Returns (RoutingDecision, None) on success, or (None, reason) on failure.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not gemini_key and not anthropic_key:
        return None, "no_api_key"

    # 1. Try Gemini if configured
    if gemini_key:
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_gemini, query, validation, gemini_key)
                decision = future.result(timeout=LLM_TIMEOUT_SECONDS)
                return decision, None
        except FuturesTimeoutError:
            return None, "gemini_timeout"
        except Exception as e:
            logger.warning("Gemini routing failed: %s", e)
            if not anthropic_key:
                return None, f"gemini_error: {type(e).__name__}"

    # 2. Try Anthropic if configured
    if anthropic_key:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return None, "anthropic_package_not_installed"

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_anthropic, query, validation)
                decision = future.result(timeout=LLM_TIMEOUT_SECONDS)
                return decision, None
        except FuturesTimeoutError:
            return None, "anthropic_timeout"
        except Exception as e:
            logger.warning("Anthropic routing failed: %s", e)
            return None, f"anthropic_error: {type(e).__name__}"

    return None, "no_api_key"


# ---------------------------------------------------------------------------
# Fallback path: deterministic keyword rules
# ---------------------------------------------------------------------------

# Keyword patterns for each task (checked in priority order)
_CHANGE_KEYWORDS = re.compile(
    r"\b(change[ds]?|changed|differ(?:s|ent|ence|ences)?|between|before\s+and\s+after|"
    r"bi[- ]?temporal|over\s+time|evolution|transform|progress|compare)\b",
    re.IGNORECASE,
)

_GROUNDING_KEYWORDS = re.compile(
    r"\b(where|highlight|locate|find|detect|show\s+me|point\s+out|"
    r"identify|mark|outline|bounding\s+box|localize)\b",
    re.IGNORECASE,
)


def _extract_target_object(query: str) -> str:
    """
    Extract a target object from a grounding query.
    Simple heuristic: look for common patterns like "find [the] X",
    "locate X", "where is [the] X".
    """
    patterns = [
        r"(?:find|locate|detect|highlight|show)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
        r"where\s+(?:is|are)\s+(?:the\s+)?(\w+(?:\s+\w+)?)",
        r"(?:any\s+)?(\w+)\s+(?:in|on|within)\s+(?:the\s+)?(?:image|scene|area)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            target = match.group(1).strip()
            # Filter out generic words
            if target.lower() not in ("the", "a", "an", "this", "that", "any", "some"):
                return target

    # Fallback: take the last noun-like word from the query
    words = query.split()
    for word in reversed(words):
        clean = re.sub(r"[^a-zA-Z]", "", word).lower()
        if clean and clean not in (
            "the", "a", "an", "is", "are", "in", "on", "this", "that",
            "image", "satellite", "picture", "photo", "where", "find",
            "can", "you", "please", "show", "me", "what", "how",
        ):
            return clean

    return "object"  # Ultimate fallback


def _fallback_route(query: str, validation: ValidationResult) -> RoutingDecision:
    """
    Deterministic keyword-based routing.
    No network access, no API key required.
    """
    query_lower = query.lower()
    modalities = validation.modalities

    # Rule 1: Two images with different modalities → fusion
    if (
        validation.image_count == 2
        and len(set(modalities)) > 1
    ):
        return RoutingDecision(
            task_name="optical_sar_fusion",
            task_arguments={"question": query if query.strip() else None},
            routing_path="fallback_routed",
        )

    # Rule 2: Change-related keywords → change analysis
    if _CHANGE_KEYWORDS.search(query_lower):
        return RoutingDecision(
            task_name="change_analysis",
            task_arguments={"description": query},
            routing_path="fallback_routed",
        )

    # Rule 3: Grounding keywords → grounding
    if _GROUNDING_KEYWORDS.search(query_lower):
        target = _extract_target_object(query)
        return RoutingDecision(
            task_name="grounding",
            task_arguments={"target_object": target},
            routing_path="fallback_routed",
        )

    # Rule 4: Default → captioning / VQA
    return RoutingDecision(
        task_name="vqa_caption",
        task_arguments={"question": query if query.strip() else None},
        routing_path="fallback_routed",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_query(
    query: str,
    validation: ValidationResult,
) -> RoutingDecision:
    """
    Route a user query to the appropriate specialist handler.

    Tries LLM-based routing first; falls back to deterministic keywords
    if the LLM path is unavailable or fails.

    Parameters
    ----------
    query : str
        The user's natural-language query.
    validation : ValidationResult
        Output from the input validator.

    Returns
    -------
    RoutingDecision
        Which handler to call, with what arguments, and how the decision
        was made.
    """
    # Try LLM routing first
    result, failure_reason = _try_llm_routing(query, validation)

    if result is not None:
        logger.info("LLM routing selected: %s", result.task_name)
        return result

    # Fall back to deterministic routing
    logger.info("Using fallback routing (reason: %s)", failure_reason)
    decision = _fallback_route(query, validation)
    decision.fallback_reason = failure_reason
    return decision
