"""Tests for DataValidator — real schema: ID, question, answer."""

from __future__ import annotations

import pandas as pd
import pytest

from src.exceptions import ValidationError
from src.validator import DataValidator


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ID":       ["ID_001", "ID_002"],
        "question": ["Q1 text", "Q2 text"],
        "answer":   ["C2", "A1"],
    })


@pytest.fixture
def validator() -> DataValidator:
    return DataValidator()


class TestDataValidator:
    def test_valid_df_passes(self, validator, valid_df):
        validator.validate(valid_df)

    def test_empty_df_raises(self, validator):
        with pytest.raises(ValidationError, match="empty"):
            validator.validate(pd.DataFrame())

    def test_missing_column_raises(self, validator):
        df = pd.DataFrame({"ID": ["x"], "question": ["q"]})   # missing answer
        with pytest.raises(ValidationError, match="Missing required columns"):
            validator.validate(df)

    def test_null_threshold_on_required_columns(self):
        v = DataValidator(required_columns=["ID", "question", "answer"], null_threshold=0.1)
        df = pd.DataFrame({
            "ID":       [None, None, "x"],
            "question": ["q1", "q2", "q3"],
            "answer":   ["A1", "B2", "C3"],
        })
        with pytest.raises(ValidationError, match="Null rate threshold"):
            v.validate(df)

    def test_repr(self, validator):
        assert "DataValidator" in repr(validator)
        assert "ID" in repr(validator)
