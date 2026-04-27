"""DataValidator: schema validation and data quality checks.

Design decision: Validator is a standalone class (not a PipelineStage)
because validation logic is orthogonal to transformation logic — it answers
"is this data safe to process?" rather than "transform this data".
This lets PipelineManager inject validation between any two stages without
modifying stage implementations (Open/Closed principle).

Updated default required_columns to match the AI Telco Troubleshooting
Challenge dataset schema (question / answer / choices).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from src.exceptions import ValidationError

logger = logging.getLogger(__name__)

# Default required columns for the Telco Troubleshooting dataset
TELCO_REQUIRED_COLUMNS: list[str] = ["question", "answer"]


class DataValidator:
    """Schema and quality validator for pipeline DataFrames.

    Args:
        required_columns: Column names that must be present.
                          Defaults to TELCO_REQUIRED_COLUMNS.
        null_threshold:   Maximum fraction of nulls allowed per column (0-1).
        name:             Human-readable name for logging.
    """

    def __init__(
        self,
        required_columns: Optional[list[str]] = None,
        null_threshold: float = 0.05,
        name: str = "DataValidator",
    ) -> None:
        self.required_columns = (
            required_columns if required_columns is not None
            else TELCO_REQUIRED_COLUMNS
        )
        self.null_threshold = null_threshold
        self.name = name
        self._report: dict[str, Any] = {}

    def validate(self, data: pd.DataFrame) -> None:
        """Run all validation checks on ``data``.

        Args:
            data: DataFrame to validate.

        Raises:
            ValidationError: If any check fails.
        """
        self._check_not_empty(data)
        self._check_required_columns(data)
        self._check_null_threshold(data)
        logger.info(
            "%s passed all checks (%d rows, %d cols)", self.name, *data.shape
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_not_empty(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValidationError(f"{self.name}: DataFrame is empty")

    def _check_required_columns(self, df: pd.DataFrame) -> None:
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            raise ValidationError(
                f"{self.name}: Missing required columns: {sorted(missing)}"
            )

    def _check_null_threshold(self, df: pd.DataFrame) -> None:
        # Only check required columns for null threshold to avoid false
        # positives on optional columns like 'context' or 'difficulty'
        cols_to_check = [c for c in self.required_columns if c in df.columns]
        null_rates = df[cols_to_check].isnull().mean()
        violations = null_rates[null_rates > self.null_threshold]
        if not violations.empty:
            details = violations.to_dict()
            raise ValidationError(
                f"{self.name}: Null rate threshold ({self.null_threshold}) "
                f"exceeded in required columns: {details}"
            )

    @property
    def report(self) -> dict[str, Any]:
        """Return the last validation report."""
        return dict(self._report)

    def __repr__(self) -> str:
        return (
            f"DataValidator(required={self.required_columns!r}, "
            f"null_threshold={self.null_threshold})"
        )

    def __str__(self) -> str:
        return f"[Validator:required={self.required_columns}]"
