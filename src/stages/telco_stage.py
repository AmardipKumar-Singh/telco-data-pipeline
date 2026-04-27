"""Telco-specific feature engineering stage.

Updated for the AI Telco Troubleshooting Challenge dataset
(cabbage-dog/The-AI-Telco-Troubleshooting-Challenge).

Dataset schema (actual columns detected at runtime):
  - question       : str   — troubleshooting question text
  - choices        : list  — candidate answer options (A/B/C/D or list)
  - answer         : str   — correct answer label
  - context        : str   — optional network log / alarm context
  - difficulty     : str   — easy / medium / hard (if present)
  - category       : str   — fault domain (if present)

Derived features added by this stage:
  - question_length        : int   — character count of question
  - num_choices            : int   — number of candidate answers
  - answer_index           : int   — 0-based index of correct answer in choices
  - has_context            : bool  — whether a context log is provided
  - context_length         : int   — char count of context (0 if absent)
  - difficulty_score       : float — ordinal encoding of difficulty
  - choices_flat           : str   — choices joined as a single string
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.exceptions import StageError, ValidationError
from .base import PipelineStage

logger = logging.getLogger(__name__)

DIFFICULTY_SCORE_MAP: dict[str, float] = {
    "easy":   1.0,
    "medium": 2.0,
    "hard":   3.0,
}

# Minimum columns the dataset must expose
REQUIRED_COLUMNS: set[str] = {"question", "answer"}


class TelcoFeatureEngineeringStage(PipelineStage):
    """Feature engineering for the AI Telco Troubleshooting Challenge dataset.

    Transforms raw QA rows from the HuggingFace dataset into an enriched
    DataFrame suitable for Spark-based aggregations and SQL persistence.

    Args:
        name: Stage name.
        config: Optional config dict; supports ``difficulty_map`` override.
    """

    def __init__(
        self,
        name: str = "TelcoFeatureEngineeringStage",
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self._difficulty_map: dict[str, float] = (
            config.get("difficulty_map", DIFFICULTY_SCORE_MAP) if config else DIFFICULTY_SCORE_MAP
        )

    # ------------------------------------------------------------------
    # PipelineStage interface
    # ------------------------------------------------------------------

    def validate(self, data: Any) -> None:
        """Verify the DataFrame contains the minimum required columns."""
        if not isinstance(data, pd.DataFrame):
            raise StageError(
                f"{self.name} expects a pandas DataFrame, got {type(data).__name__}"
            )
        missing = REQUIRED_COLUMNS - set(data.columns)
        if missing:
            raise ValidationError(
                f"{self.name}: Missing required columns: {missing}. "
                f"Found: {list(data.columns)}"
            )

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transforms to the QA dataset.

        Args:
            data: Raw DataFrame from HuggingFaceConnector.read().

        Returns:
            Enriched DataFrame with derived NLP / metadata features.
        """
        df = data.copy()
        df = self._normalize_choices(df)
        df = self._add_text_features(df)
        df = self._add_answer_index(df)
        df = self._add_context_features(df)
        df = self._encode_difficulty(df)

        self._metrics["output_cols"] = list(df.columns)
        self._metrics["output_rows"] = len(df)
        logger.info(
            "%s produced %d features on %d rows",
            self.name, len(df.columns), len(df),
        )
        return df

    # ------------------------------------------------------------------
    # Feature transformations
    # ------------------------------------------------------------------

    def _normalize_choices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure 'choices' is a Python list on every row.

        The HF dataset may store choices as a list, a dict
        {'A': '...', 'B': '...'}, or a JSON string.
        """
        if "choices" not in df.columns:
            df["choices"] = [[] for _ in range(len(df))]
            return df

        def _to_list(val: Any) -> list:
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return list(val.values())
            if isinstance(val, str):
                import json
                try:
                    parsed = json.loads(val)
                    return list(parsed.values()) if isinstance(parsed, dict) else parsed
                except (json.JSONDecodeError, TypeError):
                    return [val]
            return []

        df["choices"] = df["choices"].apply(_to_list)
        df["choices_flat"] = df["choices"].apply(lambda c: " | ".join(str(x) for x in c))
        return df

    def _add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add length-based text features for the question field."""
        df["question_length"] = df["question"].astype(str).str.len()
        df["num_choices"] = df["choices"].apply(len)
        return df

    def _add_answer_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the 0-based position of the answer within choices.

        Handles both label-style answers ('A', 'B'…) and full-text answers.
        """
        label_map = {chr(65 + i): i for i in range(26)}   # A→0, B→1, …

        def _index(row: pd.Series) -> int:
            ans = str(row["answer"]).strip()
            choices = row["choices"]
            if not choices:
                return -1
            # Direct label match
            if ans.upper() in label_map:
                idx = label_map[ans.upper()]
                return idx if idx < len(choices) else -1
            # Full-text match
            try:
                return choices.index(ans)
            except ValueError:
                return -1

        df["answer_index"] = df.apply(_index, axis=1)
        return df

    def _add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive features from the optional network-log context field."""
        if "context" not in df.columns:
            df["context"] = None

        df["has_context"] = df["context"].notna() & (df["context"].astype(str).str.strip() != "")
        df["context_length"] = df["context"].fillna("").astype(str).str.len()
        return df

    def _encode_difficulty(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ordinal-encode the difficulty column (easy/medium/hard → 1/2/3)."""
        if "difficulty" not in df.columns:
            df["difficulty_score"] = np.nan
            return df
        df["difficulty_score"] = (
            df["difficulty"].str.lower().map(self._difficulty_map)
        )
        return df
