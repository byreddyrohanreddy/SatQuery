"""
Unit tests for the task router.

Critically: tests that explicitly unset ANTHROPIC_API_KEY and verify
the fallback path routes correctly. This protects us on demo day if
the venue wifi or API is unreliable.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from satquery.models import ValidationResult
from satquery.router import route_query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_validation(
    image_count: int = 1,
    modalities: list[str] | None = None,
) -> ValidationResult:
    """Create a minimal ValidationResult for routing tests."""
    if modalities is None:
        modalities = ["optical"] * image_count
    return ValidationResult(
        image_count=image_count,
        modalities=modalities,
        formats=["JPEG"] * image_count,
        dimensions=[(256, 256)] * image_count,
        warnings=[],
        is_valid=True,
    )


# ---------------------------------------------------------------------------
# Fallback routing tests (no API key)
# ---------------------------------------------------------------------------

class TestFallbackRouting:
    """
    All tests in this class explicitly unset ANTHROPIC_API_KEY to ensure
    the fallback (deterministic) routing path works correctly offline.
    """

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_fallback_fires_when_no_api_key(self):
        """Without ANTHROPIC_API_KEY, routing should use fallback path."""
        # Remove the key entirely
        os.environ.pop("ANTHROPIC_API_KEY", None)

        validation = make_validation(image_count=1)
        decision = route_query("Describe this image", validation)

        assert decision.routing_path == "fallback_routed"
        assert decision.fallback_reason == "no_api_key"

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_change_keywords_route_to_change_analysis(self):
        """Queries with change-related keywords → change_analysis."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        test_queries = [
            "What changed between these two images?",
            "Show me the differences",
            "Compare before and after",
            "What is different in the second image?",
            "Has the area changed over time?",
        ]
        validation = make_validation(image_count=2)

        for query in test_queries:
            decision = route_query(query, validation)
            assert decision.task_name == "change_analysis", (
                f"Query '{query}' should route to change_analysis, "
                f"got {decision.task_name}"
            )
            assert decision.routing_path == "fallback_routed"

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_grounding_keywords_route_to_grounding(self):
        """Queries asking to find/locate → grounding."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        test_queries = [
            "Where is the water in this image?",
            "Highlight the vegetation",
            "Find the buildings",
            "Locate the river",
            "Detect any urban areas",
        ]
        validation = make_validation(image_count=1)

        for query in test_queries:
            decision = route_query(query, validation)
            assert decision.task_name == "grounding", (
                f"Query '{query}' should route to grounding, "
                f"got {decision.task_name}"
            )

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_different_modalities_route_to_fusion(self):
        """Two images with different modalities → optical_sar_fusion."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        validation = make_validation(
            image_count=2,
            modalities=["optical", "SAR"],
        )
        decision = route_query("Analyze these images", validation)

        assert decision.task_name == "optical_sar_fusion"
        assert decision.routing_path == "fallback_routed"

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_default_routes_to_vqa_caption(self):
        """Generic queries with no special keywords → vqa_caption."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        validation = make_validation(image_count=1)
        decision = route_query("What is in this satellite image?", validation)

        assert decision.task_name == "vqa_caption"

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_empty_query_routes_to_vqa_caption(self):
        """An empty query should default to captioning."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        validation = make_validation(image_count=1)
        decision = route_query("", validation)

        assert decision.task_name == "vqa_caption"

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_fusion_priority_over_change_keywords(self):
        """Different modalities should trigger fusion even with change keywords."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        validation = make_validation(
            image_count=2,
            modalities=["optical", "SAR"],
        )
        decision = route_query("What changed?", validation)

        # Fusion takes priority when modalities differ
        assert decision.task_name == "optical_sar_fusion"

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_grounding_extracts_target_object(self):
        """Grounding routing should extract a target object from the query."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        validation = make_validation(image_count=1)
        decision = route_query("Find the water bodies", validation)

        assert decision.task_name == "grounding"
        assert "target_object" in decision.task_arguments
        assert decision.task_arguments["target_object"]  # Not empty

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_routing_decision_has_complete_fields(self):
        """Every routing decision should have all required fields set."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        validation = make_validation(image_count=1)
        decision = route_query("Describe this", validation)

        assert decision.task_name
        assert decision.routing_path in ("llm_routed", "fallback_routed")
        assert isinstance(decision.task_arguments, dict)
        # Fallback should always have a reason
        if decision.routing_path == "fallback_routed":
            assert decision.fallback_reason is not None


# ---------------------------------------------------------------------------
# Routing path recording tests
# ---------------------------------------------------------------------------

class TestRoutingTransparency:
    """Tests for routing path transparency and trace completeness."""

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_fallback_reason_is_recorded(self):
        """When fallback fires, the reason should be recorded."""
        os.environ.pop("ANTHROPIC_API_KEY", None)

        validation = make_validation()
        decision = route_query("test", validation)

        assert decision.routing_path == "fallback_routed"
        assert decision.fallback_reason is not None
        assert len(decision.fallback_reason) > 0
