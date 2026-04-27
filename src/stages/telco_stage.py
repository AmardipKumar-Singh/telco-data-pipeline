"""Telco-specific feature engineering stage.

Domain knowledge encoded here:
- Alarm correlation within a time window (for root-cause analysis)
- KPI normalization specific to RAN metrics (RSRP, SINR, PRB utilization)
- Severity scoring aligned with 3GPP alarm severity levels
- Cell-level feature aggregation for federated learning dataset preparation
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.exceptions import StageError, ValidationError
from .base import PipelineStage

logger = logging.getLogger(__name__)

# 3GPP TS 32.111-2 severity mapping
SEVERITY_SCORE_MAP: dict[str, float] = {
    "CRITICAL": 4.0,
    "MAJOR": 3.0,
    "MINOR": 2.0,
    "WARNING": 1.0,
    "INDETERMINATE": 0.5,
    "CLEARED": 0.0,
}


class TelcoFeatureEngineeringStage(PipelineStage):
    """Feature engineering tailored to telecom alarm and RAN KPI data.

    Transformations applied:
    1. Numeric severity encoding (3GPP TS 32.111-2 mapping).
    2. KPI column normalization (z-score per cell_id group).
    3. Alarm recurrence rate within configurable time window.
    4. Cell-level feature vector for FL dataset partitioning.

    Args:
        name: Stage name.
        config: Should contain ``kpi_columns`` and ``window_duration_minutes``.
    """

    def __init__(
        self,
        name: str = "TelcoFeatureEngineeringStage",
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._kpi_cols: list[str] = config.get("kpi_columns", []) if config else []
        self._window_min: int = config.get("window_duration_minutes", 15) if config else 15

    # ------------------------------------------------------------------
    # PipelineStage interface
    # ------------------------------------------------------------------

    def validate(self, data: Any) -> None:
        """Check that required alarm columns are present."""
        if not isinstance(data, pd.DataFrame):
            raise StageError(
                f"{self.name} expects a pandas DataFrame, got {type(data).__name__}"
            )
        required = {"alarm_id", "cell_id", "alarm_type", "severity", "timestamp"}
        missing = required - set(data.columns)
        if missing:
            raise ValidationError(
                f"{self.name}: Missing required columns: {missing}"
            )

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all telco feature engineering transforms.

        Args:
            data: Raw alarm event DataFrame from the HF dataset.

        Returns:
            Enriched DataFrame with new feature columns.
        """
        df = data.copy()
        df = self._encode_severity(df)
        df = self._normalize_kpis(df)
        df = self._compute_alarm_recurrence(df)
        df = self._add_cell_features(df)
        self._metrics["output_cols"] = list(df.columns)
        self._metrics["output_rows"] = len(df)
        logger.info("%s produced %d features on %d rows", self.name, len(df.columns), len(df))
        return df

    # ------------------------------------------------------------------
    # Feature transformations
    # ------------------------------------------------------------------

    def _encode_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map 3GPP severity strings to numeric scores."""
        df["severity_score"] = (
            df["severity"].str.upper().map(SEVERITY_SCORE_MAP).fillna(0.0)
        )
        return df

    def _normalize_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalize KPI columns within each cell_id group."""
        for col in self._kpi_cols:
            if col not in df.columns:
                logger.debug("%s: KPI column '%s' not found, skipping", self.name, col)
                continue
            grouped = df.groupby("cell_id")[col]
            df[f"{col}_norm"] = (df[col] - grouped.transform("mean")) / (
                grouped.transform("std").replace(0, 1)  # Avoid div-by-zero
            )
        return df

    def _compute_alarm_recurrence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-(cell, alarm_type) event rate within rolling window."""
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values(["cell_id", "alarm_type", "timestamp"])
        window = f"{self._window_min}min"
        df["alarm_recurrence_rate"] = (
            df.groupby(["cell_id", "alarm_type"])["alarm_id"]
            .transform(lambda s: s.expanding().count())
            .astype(float)
        )
        return df

    def _add_cell_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate cell-level summary features for FL partitioning."""
        cell_stats = (
            df.groupby("cell_id")
            .agg(
                cell_alarm_count=("alarm_id", "count"),
                cell_avg_severity=("severity_score", "mean"),
                cell_unique_alarm_types=("alarm_type", "nunique"),
            )
            .reset_index()
        )
        df = df.merge(cell_stats, on="cell_id", how="left")
        return df
