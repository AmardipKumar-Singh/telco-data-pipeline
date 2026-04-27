"""Abstract base class for all pipeline transformation stages.

Design decision: PipelineStage is designed around the Template Method
pattern.  Subclasses implement process() (the variant algorithm) while the
base class provides validate() as a hook that subclasses can override to
enforce schema contracts before processing begins.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Abstract base class for composable pipeline transformation stages.

    Args:
        name: Human-readable stage name used in logging and metrics.
        config: Stage-specific configuration dictionary.
    """

    def __init__(self, name: str, config: Optional[dict[str, Any]] = None) -> None:
        self.name = name
        self.config = config or {}
        self._metrics: dict[str, Any] = {}
        logger.debug("Stage %s initialised", self.name)

    # ------------------------------------------------------------------
    # Template method: run() calls validate() then process()
    # ------------------------------------------------------------------

    def run(self, data: Any) -> Any:
        """Execute the stage: validate input, then process.

        Args:
            data: Input data payload.

        Returns:
            Transformed data payload.
        """
        logger.info("Stage [%s] starting", self.name)
        self.validate(data)
        start = time.perf_counter()
        result = self.process(data)
        elapsed = time.perf_counter() - start
        self._metrics["last_duration_s"] = round(elapsed, 4)
        logger.info("Stage [%s] completed in %.4fs", self.name, elapsed)
        return result

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Apply the stage transformation to ``data``.

        Args:
            data: Input payload (format defined by each subclass).

        Returns:
            Transformed payload.
        """

    def validate(self, data: Any) -> None:
        """Validate input before processing.  Override in subclasses.

        Raises:
            ValidationError: If the data violates schema expectations.
        """
        # Default: no-op.  Subclasses add schema checks here.

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> dict[str, Any]:
        """Return execution metrics for this stage."""
        return dict(self._metrics)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    def __str__(self) -> str:
        return f"[Stage:{self.name}]"
