"""Tests for pipeline stages.

Updated fixtures use the AI Telco Troubleshooting Challenge schema
(question / choices / answer / context / difficulty) instead of the
former alarm-event schema.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.exceptions import StageError, ValidationError
from src.stages.telco_stage import TelcoFeatureEngineeringStage
from src.stages.sql_stage import SQLAggregationStage


# ─────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def telco_qa_df() -> pd.DataFrame:
    """Minimal DataFrame matching the HF dataset schema."""
    return pd.DataFrame({
        "question": [
            "Which protocol is used for network troubleshooting?",
            "What does a high RSRP value indicate?",
            "Which layer handles packet routing?",
        ],
        "choices": [
            ["ICMP", "HTTP", "FTP", "SMTP"],
            ["Poor signal", "Strong signal", "No signal", "Interference"],
            ["Physical", "Data Link", "Network", "Transport"],
        ],
        "answer": ["A", "B", "C"],
        "context": [
            "Router logs show ICMP unreachable messages.",
            None,
            "Packet loss observed at layer 3.",
        ],
        "difficulty": ["easy", "medium", "hard"],
        "category": ["Networking", "RAN", "Routing"],
    })


@pytest.fixture
def stage() -> TelcoFeatureEngineeringStage:
    return TelcoFeatureEngineeringStage(
        config={"difficulty_map": {"easy": 1.0, "medium": 2.0, "hard": 3.0}}
    )


# ─────────────────────────────────────────────────────────────
# TelcoFeatureEngineeringStage
# ─────────────────────────────────────────────────────────────

class TestTelcoFeatureEngineeringStage:
    def test_validate_passes_valid_df(self, stage, telco_qa_df):
        stage.validate(telco_qa_df)  # should not raise

    def test_validate_raises_on_missing_columns(self, stage):
        bad_df = pd.DataFrame({"foo": [1, 2]})
        with pytest.raises(ValidationError, match="Missing required columns"):
            stage.validate(bad_df)

    def test_validate_raises_on_non_dataframe(self, stage):
        with pytest.raises(StageError):
            stage.validate({"not": "a dataframe"})

    def test_process_adds_question_length(self, stage, telco_qa_df):
        result = stage.process(telco_qa_df)
        assert "question_length" in result.columns
        assert (result["question_length"] > 0).all()

    def test_process_adds_num_choices(self, stage, telco_qa_df):
        result = stage.process(telco_qa_df)
        assert "num_choices" in result.columns
        assert (result["num_choices"] == 4).all()

    def test_process_adds_answer_index(self, stage, telco_qa_df):
        result = stage.process(telco_qa_df)
        assert "answer_index" in result.columns
        # A→0, B→1, C→2
        assert list(result["answer_index"]) == [0, 1, 2]

    def test_process_adds_has_context(self, stage, telco_qa_df):
        result = stage.process(telco_qa_df)
        assert "has_context" in result.columns
        assert result["has_context"].sum() == 2   # row 1 has no context

    def test_process_encodes_difficulty(self, stage, telco_qa_df):
        result = stage.process(telco_qa_df)
        assert "difficulty_score" in result.columns
        assert list(result["difficulty_score"]) == [1.0, 2.0, 3.0]

    def test_run_updates_metrics(self, stage, telco_qa_df):
        stage.run(telco_qa_df)
        assert stage.metrics["output_rows"] == 3
        assert "question_length" in stage.metrics["output_cols"]

    def test_choices_flat_created(self, stage, telco_qa_df):
        result = stage.process(telco_qa_df)
        assert "choices_flat" in result.columns
        assert "ICMP" in result["choices_flat"].iloc[0]


# ─────────────────────────────────────────────────────────────
# SQLAggregationStage
# ─────────────────────────────────────────────────────────────

class TestSQLAggregationStage:
    @pytest.fixture
    def enriched_df(self, stage, telco_qa_df) -> pd.DataFrame:
        return stage.process(telco_qa_df)

    def test_process_returns_dataframe(self, enriched_df):
        sql_stage = SQLAggregationStage(config={})
        result = sql_stage.process(enriched_df)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_validate_raises_on_empty(self):
        sql_stage = SQLAggregationStage(config={})
        with pytest.raises(StageError, match="empty"):
            sql_stage.validate(pd.DataFrame())

    def test_validate_raises_on_non_dataframe(self):
        sql_stage = SQLAggregationStage(config={})
        with pytest.raises(StageError):
            sql_stage.validate([1, 2, 3])

    def test_aggregation_columns_present(self, enriched_df):
        sql_stage = SQLAggregationStage(config={})
        result = sql_stage.process(enriched_df)
        assert "question_count" in result.columns
        assert "avg_question_length" in result.columns
