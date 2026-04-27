"""Unit tests for PipelineManager."""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from src.pipeline_manager import PipelineManager, StageResult
from src.stages.base import PipelineStage
from src.exceptions import OrchestrationError


class PassThroughStage(PipelineStage):
    """Stage that returns data unchanged — used for testing orchestration."""
    def process(self, data):
        return data


class FailingStage(PipelineStage):
    """Stage that always raises — tests retry/failure logic."""
    def __init__(self, name="FailingStage", fail_times: int = 99):
        super().__init__(name)
        self._fail_count = 0
        self._fail_times = fail_times

    def process(self, data):
        if self._fail_count < self._fail_times:
            self._fail_count += 1
            raise RuntimeError("Intentional failure")
        return data


@pytest.fixture
def sample_df():
    return pd.DataFrame({"alarm_id": ["A1", "A2"], "cell_id": ["C1", "C2"],
                         "alarm_type": ["T1", "T2"], "severity": ["CRITICAL", "MINOR"],
                         "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"])})


def test_pipeline_runs_pass_through(sample_df):
    mgr = PipelineManager([PassThroughStage("stage1"), PassThroughStage("stage2")])
    result = mgr.run(sample_df)
    pd.testing.assert_frame_equal(result, sample_df)


def test_pipeline_run_log_populated(sample_df):
    mgr = PipelineManager([PassThroughStage("s1"), PassThroughStage("s2")])
    mgr.run(sample_df)
    assert len(mgr.run_log) == 2
    assert all(r.success for r in mgr.run_log)


def test_pipeline_retries_and_fails(sample_df):
    failing = FailingStage(fail_times=99)
    mgr = PipelineManager([failing], max_retries=2, retry_backoff_s=0)
    with pytest.raises(OrchestrationError):
        mgr.run(sample_df)
    assert mgr.run_log[-1].success is False


def test_pipeline_retries_succeed_eventually(sample_df):
    # Fails twice, then succeeds on 3rd attempt
    recovering = FailingStage(fail_times=2)
    mgr = PipelineManager([recovering], max_retries=3, retry_backoff_s=0)
    result = mgr.run(sample_df)
    pd.testing.assert_frame_equal(result, sample_df)


def test_pipeline_add_stage_chaining(sample_df):
    mgr = (
        PipelineManager()
        .add_stage(PassThroughStage("s1"))
        .add_stage(PassThroughStage("s2"))
        .add_stage(PassThroughStage("s3"))
    )
    assert len(mgr.stages) == 3


def test_pipeline_str_repr():
    mgr = PipelineManager([PassThroughStage("s1")], name="TestPipe")
    assert "TestPipe" in repr(mgr)
    assert "s1" in str(mgr)
