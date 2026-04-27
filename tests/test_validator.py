"""Tests for DataValidator using the Telco QA dataset schema."""

from __future__ import annotations

import pandas as pd
import pytest

from src.exceptions import ValidationError
from src.validator import DataValidator


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return pd.DataFrame({
        "question": ["Q1", "Q2", "Q3"],
        "answer": ["A", "B", "C"],
        "choices": [["A", "B"], ["C", "D"], ["E", "F"]],
    })


@pytest.fixture
def validator() -> DataValidator:
    return DataValidator()


class TestDataValidator:
    def test_valid_df_passes(self, validator, valid_df):
        validator.validate(valid_df)   # no exception

    def test_empty_df_raises(self, validator):
        with pytest.raises(ValidationError, match="empty"):
            validator.validate(pd.DataFrame())

    def test_missing_required_column_raises(self, validator):
        df = pd.DataFrame({"question": ["Q1"]})   # missing 'answer'
        with pytest.raises(ValidationError, match="Missing required columns"):
            validator.validate(df)

    def test_null_threshold_enforced_on_required_columns(self):
        v = DataValidator(
            required_columns=["question", "answer"], null_threshold=0.1
        )
        df = pd.DataFrame({
            "question": ["Q1", None, None],   # 66 % nulls — over threshold
            "answer": ["A", "B", "C"],
        })
        with pytest.raises(ValidationError, match="Null rate threshold"):
            v.validate(df)

    def test_optional_columns_not_checked_for_nulls(self):
        """context is not required — should not trigger null threshold."""
        v = DataValidator(required_columns=["question", "answer"], null_threshold=0.05)
        df = pd.DataFrame({
            "question": ["Q1", "Q2"],
            "answer": ["A", "B"],
            "context": [None, None],   # 100 % null but optional
        })
        v.validate(df)   # should pass

    def test_repr(self, validator):
        assert "DataValidator" in repr(validator)
        assert "question" in repr(validator)
