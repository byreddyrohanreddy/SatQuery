"""
Execution Trace builder for SatQuery AI.

Uses a builder pattern to accumulate trace information across the pipeline,
then produces a complete ExecutionTrace pydantic model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from satquery.models import ExecutionTrace, ValidationResult


class TraceBuilder:
    """
    Accumulates execution metadata across the pipeline and produces
    a complete ExecutionTrace.

    Usage:
        trace = (
            TraceBuilder()
            .set_validation(validation_result)
            .set_routing("vqa_caption", "fallback_routed", fallback_reason="no_api_key")
            .set_handler_info(models=["blip-base"], parameters={"question": "..."})
            .set_confidence(0.65, "BLIP base model confidence")
            .build()
        )
    """

    def __init__(self) -> None:
        self._task_selected: str = ""
        self._routing_path: str = ""
        self._fallback_reason: Optional[str] = None
        self._validator_output: Optional[ValidationResult] = None
        self._models_called: list[str] = []
        self._parameters_used: dict[str, Any] = {}
        self._confidence: float = 0.0
        self._confidence_basis: str = ""
        self._timestamp: datetime = datetime.now(timezone.utc)

    def set_validation(self, result: ValidationResult) -> TraceBuilder:
        """Record the validator output."""
        self._validator_output = result
        return self

    def set_routing(
        self,
        task_name: str,
        routing_path: str,
        fallback_reason: Optional[str] = None,
    ) -> TraceBuilder:
        """Record the routing decision."""
        self._task_selected = task_name
        self._routing_path = routing_path
        self._fallback_reason = fallback_reason
        return self

    def set_handler_info(
        self,
        models: list[str],
        parameters: dict[str, Any] | None = None,
    ) -> TraceBuilder:
        """Record which models/algorithms were called and with what parameters."""
        self._models_called = models
        self._parameters_used = parameters or {}
        return self

    def set_confidence(self, score: float, basis: str) -> TraceBuilder:
        """Record the confidence score and its derivation."""
        self._confidence = score
        self._confidence_basis = basis
        return self

    def build(self) -> ExecutionTrace:
        """Produce the final ExecutionTrace. Raises if required fields are missing."""
        if self._validator_output is None:
            raise ValueError("TraceBuilder: validator output was never set")
        if not self._task_selected:
            raise ValueError("TraceBuilder: routing was never set")

        return ExecutionTrace(
            task_selected=self._task_selected,
            routing_path=self._routing_path,
            fallback_reason=self._fallback_reason,
            validator_output=self._validator_output,
            models_called=self._models_called,
            model_called=", ".join(self._models_called) if self._models_called else "Not reported",
            parameters_used=self._parameters_used,
            timestamp=self._timestamp,
            confidence=self._confidence,
            confidence_basis=self._confidence_basis,
        )
