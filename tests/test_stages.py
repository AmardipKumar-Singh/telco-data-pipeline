"""Unit tests for PipelineStage implementations."""

import pytest
import pandas as pd
import numpy as np

from src.stages.telco_stage import TelcoFeatureEngineeringStage
from src.stages.sql_stage import SQLAggregationStage
from src.exceptions import StageError, ValidationError


@pytest.fixture
def alarm_df():
    """Minimal alarm DataFrame matching required schema."""
    return pd.DataFrame({
        "alarm_id": ["A001", "A002", "A003"],
        "cell_id": ["CELL-001", "CELL-001", "CELL-002"],
        "alarm_type": ["LINK_DOWN", "HIGH_PRB", "LINK_DOWN"],
        "severity": ["CRITICAL", "MAJOR", "MINOR"],
        "timestamp": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        "rsrp": [-85.0, -92.0, -78.0],
        "sinr": [12.0, 8.0, 20.0],
    })


# ---------------------------------------------------------------------------
# TelcoFeatureEngineeringStage
# ---------------------------------------------------------------------------

def test_telco_stage_encodes_severity(alarm_df):
    stage = TelcoFeatureEngineeringStage(
        config={"kpi_columns": ["rsrp", "sinr"], "window_duration_minutes": 15}
    )
    result = stage.run(alarm_df)
    assert "severity_score" in result.columns
    assert result.loc[result["severity"] == "CRITICAL", "severity_score"].iloc[0] == 4.0


def test_telco_stage_normalizes_kpis(alarm_df):
    stage = TelcoFeatureEngineeringStage(
        config={"kpi_columns": ["rsrp", "sinr"]}
    )
    result = stage.run(alarm_df)
    assert "rsrp_norm" in result.columns


def test_telco_stage_missing_column_raises():
    df = pd.DataFrame({"alarm_id": [1], "cell_id": ["C1"]})  # Missing required cols
    stage = TelcoFeatureEngineeringStage()
    with pytest.raises((ValidationError, StageError)):
        stage.run(df)


def test_telco_stage_adds_cell_features(alarm_df):
    stage = TelcoFeatureEngineeringStage(config={"kpi_columns": []})
    result = stage.run(alarm_df)
    assert "cell_alarm_count" in result.columns
    assert "cell_avg_severity" in result.columns


def test_telco_stage_metrics_populated(alarm_df):
    stage = TelcoFeatureEngineeringStage(config={"kpi_columns": []})
    stage.run(alarm_df)
    assert stage.metrics.get("output_rows") == len(alarm_df)


# ---------------------------------------------------------------------------
# SQLAggregationStage
# ---------------------------------------------------------------------------

def test_sql_stage_aggregates(alarm_df):
    # We need severity_score for the query — run telco stage first
    telco = TelcoFeatureEngineeringStage(config={"kpi_columns": []})
    enriched = telco.run(alarm_df)

    stage = SQLAggregationStage(config={
        "query": "SELECT cell_id, COUNT(*) as alarm_count FROM staging_events GROUP BY cell_id",
        "output_table": "results",
    })
    result = stage.run(enriched)
    assert "cell_id" in result.columns
    assert "alarm_count" in result.columns
    assert len(result) == 2  # CELL-001, CELL-002


def test_sql_stage_rejects_empty_df():
    stage = SQLAggregationStage()
    with pytest.raises(StageError):
        stage.run(pd.DataFrame())


def test_sql_stage_rejects_non_dataframe():
    stage = SQLAggregationStage()
    with pytest.raises(StageError):
        stage.run([{"alarm_id": "x"}])
