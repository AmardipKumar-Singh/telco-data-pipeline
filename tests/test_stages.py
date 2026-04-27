"""Tests for pipeline stages — real train.csv schema: ID, question, answer."""

from __future__ import annotations

import pandas as pd
import pytest

from src.exceptions import StageError, ValidationError
from src.stages.telco_stage import TelcoFeatureEngineeringStage
from src.stages.sql_stage import SQLAggregationStage


@pytest.fixture
def telco_qa_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ID": ["ID_001", "ID_002", "ID_003"],
        "question": [
            "Analyze the 5G drive-test data.\nThe throughput drops below 600Mbps.\n1. High interference\n2. Low RSRP\n3. PRB congestion",
            "A handover failure is observed at cell boundary.\nC1. Incorrect HO threshold\nC2. Missing neighbour\nC3. Coverage hole",
            "The SINR is degraded.\n| Cell | RSRP | SINR |\nFigure 1 shows the KPI trend.\nA1. UE issue\nA2. Site fault",
        ],
        "answer": ["C2", "A1", "B3"],
    })


@pytest.fixture
def stage() -> TelcoFeatureEngineeringStage:
    return TelcoFeatureEngineeringStage(config={})


class TestTelcoFeatureEngineeringStage:
    def test_validate_passes_valid_df(self, stage, telco_qa_df):
        stage.validate(telco_qa_df)

    def test_validate_raises_on_missing_columns(self, stage):
        with pytest.raises(ValidationError, match="Missing required columns"):
            stage.validate(pd.DataFrame({"foo": [1]}))

    def test_validate_raises_on_non_dataframe(self, stage):
        with pytest.raises(StageError):
            stage.validate({"not": "a dataframe"})

    def test_answer_letter_parsed(self, stage, telco_qa_df):
        result = stage.process(telco_qa_df)
        assert list(result["answer_letter"]) == ["C", "A", "B"]

    def test_answer_number_parsed(self, stage, telco_qa_df):
        result = stage.process(telco_qa_df)
        assert list(result["answer_number"]) == [2, 1, 3]

    def test_question_length_positive(self, stage, telco_qa_df):
        result = stage.process(telco_qa_df)
        assert (result["question_length"] > 0).all()

    def test_question_lines_counted(self, stage, telco_qa_df):
        result = stage.process(telco_qa_df)
        assert (result["question_lines"] >= 1).all()

    def test_has_table_detected(self, stage, telco_qa_df):
        result = stage.process(telco_qa_df)
        # Third row has a table with "|"
        assert result["has_table"].iloc[2] == True

    def test_has_figure_detected(self, stage, telco_qa_df):
        result = stage.process(telco_qa_df)
        assert result["has_figure"].iloc[2] == True

    def test_scenario_type_assigned(self, stage, telco_qa_df):
        result = stage.process(telco_qa_df)
        assert result["scenario_type"].iloc[0] == "throughput"
        assert result["scenario_type"].iloc[1] == "handover"

    def test_metrics_updated(self, stage, telco_qa_df):
        stage.run(telco_qa_df)
        assert stage.metrics["output_rows"] == 3
        assert "answer_letter" in stage.metrics["output_cols"]


class TestSQLAggregationStage:
    @pytest.fixture
    def enriched_df(self, stage, telco_qa_df):
        return stage.process(telco_qa_df)

    def test_process_returns_dataframe(self, enriched_df):
        sql_stage = SQLAggregationStage(config={})
        result = sql_stage.process(enriched_df)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_aggregation_columns_present(self, enriched_df):
        sql_stage = SQLAggregationStage(config={})
        result = sql_stage.process(enriched_df)
        assert "question_count" in result.columns
        assert "scenario_type" in result.columns

    def test_validate_raises_on_empty(self):
        with pytest.raises(StageError, match="empty"):
            SQLAggregationStage(config={}).validate(pd.DataFrame())
