"""Telco-specific feature engineering stage.

Adapted for the real schema of the AI Telco Troubleshooting Challenge:

    train.csv columns:
        ID        — unique row identifier  (e.g. "ID_1P7PJMPV0R")
        question  — full troubleshooting scenario text (multi-line)
        answer    — correct option label   (e.g. "C2", "A1", "B3")

Derived features added by this stage:
    question_length   int   — total character count of the question
    num_options       int   — number of numbered options found in the question
    answer_letter     str   — letter part of the answer label  (e.g. "C" from "C2")
    answer_number     int   — numeric part of the answer label (e.g. 2  from "C2")
    question_lines    int   — line count (proxy for scenario complexity)
    has_table         bool  — True if the question contains a markdown/ASCII table
    has_figure        bool  — True if the question references a figure/chart
    scenario_type     str   — coarse category inferred from question keywords
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.exceptions import StageError, ValidationError
from .base import PipelineStage

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: set[str] = {"question", "answer"}

# Keywords used to infer a coarse scenario category
_SCENARIO_KEYWORDS: dict[str, list[str]] = {
    "throughput":    ["throughput", "mbps", "gbps", "bandwidth", "speed"],
    "coverage":      ["coverage", "rsrp", "rsrq", "rssi", "signal", "rssnr"],
    "interference":  ["interference", "sinr", "noise", "snr", "iq"],
    "handover":      ["handover", "handoff", "ho failure", "ho success"],
    "latency":       ["latency", "delay", "rtt", "ping", "jitter"],
    "connectivity":  ["connection", "attach", "detach", "pdn", "bearer"],
    "capacity":      ["prb", "utilisation", "utilization", "congestion", "load"],
}


class TelcoFeatureEngineeringStage(PipelineStage):
    """Feature engineering for the AI Telco Troubleshooting Challenge dataset.

    Derives structured features from the free-text question field and the
    compact answer label for downstream SQL aggregation and ML evaluation.

    Args:
        name:   Stage name.
        config: Optional config dict (no required keys for current version).
    """

    def __init__(
        self,
        name: str = "TelcoFeatureEngineeringStage",
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)

    # ------------------------------------------------------------------
    # PipelineStage interface
    # ------------------------------------------------------------------

    def validate(self, data: Any) -> None:
        """Verify the DataFrame contains the minimum required columns."""
        if not isinstance(data, pd.DataFrame):
            raise StageError(
                f"{self.name} expects a pandas DataFrame, "
                f"got {type(data).__name__}"
            )
        missing = REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValidationError(
                f"{self.name}: Missing required columns: {missing}. "
                f"Found: {list(data.columns)}"
            )

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transforms.

        Args:
            data: Raw DataFrame from HuggingFaceConnector.read().

        Returns:
            Enriched DataFrame with derived feature columns.
        """
        df = data.copy()
        df = self._parse_answer_label(df)
        df = self._add_text_features(df)
        df = self._detect_content_type(df)
        df = self._infer_scenario_type(df)

        self._metrics["output_cols"] = list(df.columns)
        self._metrics["output_rows"] = len(df)
        logger.info(
            "%s: %d rows → %d columns",
            self.name, len(df), len(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    # Feature transformations
    # ------------------------------------------------------------------

    def _parse_answer_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Split the answer label (e.g. 'C2') into letter + number parts."""
        def _letter(val: str) -> str:
            m = re.match(r"^([A-Za-z]+)", str(val).strip())
            return m.group(1).upper() if m else ""

        def _number(val: str) -> int:
            m = re.search(r"(\d+)$", str(val).strip())
            return int(m.group(1)) if m else -1

        df["answer_letter"] = df["answer"].apply(_letter)
        df["answer_number"] = df["answer"].apply(_number)
        return df

    def _add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add length and structure features from the question text."""
        q = df["question"].astype(str)
        df["question_length"] = q.str.len()
        df["question_lines"]  = q.str.count("\n") + 1

        # Count numbered options: lines starting with "1.", "2.", ... or "C1.", "A2."
        def _count_options(text: str) -> int:
            return len(re.findall(
                r"(?m)^\s*(?:[A-Za-z]?\d+[.)\s]|[A-Za-z][.)\s])",
                text
            ))

        df["num_options"] = q.apply(_count_options)
        return df

    def _detect_content_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect presence of tables and figure references in the question."""
        q = df["question"].astype(str)
        df["has_table"]  = q.str.contains(
            r"\|.*\||	.*	|[+]{2,}[-]{2,}", regex=True
        )
        df["has_figure"] = q.str.contains(
            r"(?i)(figure|fig\.|chart|graph|diagram|image|table)", regex=True
        )
        return df

    def _infer_scenario_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign a coarse scenario category based on keyword matching."""
        q_lower = df["question"].astype(str).str.lower()

        def _categorise(text: str) -> str:
            for category, keywords in _SCENARIO_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    return category
            return "other"

        df["scenario_type"] = q_lower.apply(_categorise)
        return df
