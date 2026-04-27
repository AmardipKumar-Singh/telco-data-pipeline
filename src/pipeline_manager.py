"""PipelineManager: orchestrates stages, handles retries, and logs metrics.

Design decision: PipelineManager uses a chain-of-responsibility pattern.
Each stage's output becomes the next stage's input.  Stages are registered
as a list (not a graph) for simplicity; dependency resolution can be added
later by replacing the list with a DAG (e.g., networkx).

Retry logic uses exponential backoff per stage to handle transient failures
(e.g., momentary Kafka unavailability) without surfacing them to operators.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.exceptions import OrchestrationError, StageError
from src.stages.base import PipelineStage
from src.validator import DataValidator

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Holds the outcome of a single stage execution."""

    stage_name: str
    success: bool
    duration_s: float
    output_rows: Optional[int] = None
    error: Optional[str] = None


class PipelineManager:
    """Orchestrates the execution of a sequence of PipelineStages.

    Args:
        stages: Ordered list of PipelineStage instances.
        validator: Optional DataValidator injected between stages.
        max_retries: Number of retry attempts per stage on failure.
        retry_backoff_s: Base backoff delay in seconds (doubles each retry).
        name: Human-readable pipeline name.
    """

    def __init__(
        self,
        stages: Optional[list[PipelineStage]] = None,
        validator: Optional[DataValidator] = None,
        max_retries: int = 3,
        retry_backoff_s: float = 5.0,
        name: str = "PipelineManager",
    ) -> None:
        self.stages: list[PipelineStage] = stages or []
        self.validator = validator
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.name = name
        self._run_log: list[StageResult] = []

    # ------------------------------------------------------------------
    # Stage registration
    # ------------------------------------------------------------------

    def add_stage(self, stage: PipelineStage) -> "PipelineManager":
        """Append a stage to the pipeline chain.  Returns self for chaining."""
        self.stages.append(stage)
        logger.debug("%s: stage %s added (total=%d)", self.name, stage.name, len(self.stages))
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, initial_data: Any) -> Any:
        """Execute all stages sequentially with retry and validation.

        Args:
            initial_data: Data payload to pass into the first stage.

        Returns:
            Output of the final stage.

        Raises:
            OrchestrationError: If any stage fails after all retries.
        """
        self._run_log.clear()
        data = initial_data
        pipeline_start = time.perf_counter()
        logger.info("%s run starting (%d stages)", self.name, len(self.stages))

        for stage in self.stages:
            data = self._run_stage_with_retry(stage, data)
            if self.validator is not None:
                try:
                    self.validator.validate(data)
                except Exception as exc:
                    raise OrchestrationError(
                        f"Validation failed after stage '{stage.name}': {exc}"
                    ) from exc

        total = time.perf_counter() - pipeline_start
        logger.info("%s completed in %.2fs", self.name, total)
        self._log_summary(total)
        return data

    def _run_stage_with_retry(self, stage: PipelineStage, data: Any) -> Any:
        """Run a single stage with exponential backoff retries.

        Args:
            stage: Stage to execute.
            data: Input data for the stage.

        Returns:
            Stage output.

        Raises:
            OrchestrationError: After all retries are exhausted.
        """
        backoff = self.retry_backoff_s
        for attempt in range(1, self.max_retries + 2):  # +1 for initial attempt
            try:
                start = time.perf_counter()
                result = stage.run(data)
                duration = time.perf_counter() - start
                self._run_log.append(
                    StageResult(
                        stage_name=stage.name,
                        success=True,
                        duration_s=round(duration, 4),
                        output_rows=stage.metrics.get("output_rows"),
                    )
                )
                return result
            except (StageError, Exception) as exc:  # noqa: BLE001
                logger.warning(
                    "%s stage '%s' attempt %d/%d failed: %s",
                    self.name, stage.name, attempt, self.max_retries + 1, exc,
                )
                if attempt > self.max_retries:
                    self._run_log.append(
                        StageResult(
                            stage_name=stage.name,
                            success=False,
                            duration_s=0.0,
                            error=str(exc),
                        )
                    )
                    raise OrchestrationError(
                        f"Stage '{stage.name}' failed after {self.max_retries} retries: {exc}"
                    ) from exc
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
        return data  # Unreachable, satisfies type checker

    # ------------------------------------------------------------------
    # Metrics & reporting
    # ------------------------------------------------------------------

    @property
    def run_log(self) -> list[StageResult]:
        """Return a copy of the execution log."""
        return list(self._run_log)

    def _log_summary(self, total_s: float) -> None:
        """Emit a structured summary log of the completed run."""
        successful = sum(1 for r in self._run_log if r.success)
        failed = len(self._run_log) - successful
        total_rows = sum(r.output_rows or 0 for r in self._run_log)
        logger.info(
            "%s summary | stages=%d ok=%d failed=%d | total_rows=%d | duration=%.2fs",
            self.name, len(self._run_log), successful, failed, total_rows, total_s,
        )

    def __repr__(self) -> str:
        return f"PipelineManager(name={self.name!r}, stages={len(self.stages)})"

    def __str__(self) -> str:
        stage_names = " → ".join(s.name for s in self.stages)
        return f"[{self.name}] {stage_names}"
