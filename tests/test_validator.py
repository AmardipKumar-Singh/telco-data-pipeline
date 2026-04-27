"""Unit tests for DataValidator."""

import pytest
import pandas as pd
import numpy as np

from src.validator import DataValidator
from src.exceptions import ValidationError


@pytest.fixture
def valid_df():
    return pd.DataFrame({
        "alarm_id": ["A1", "A2"],
        "cell_id": ["C1", "C2"],
        "severity": ["CRITICAL", "MINOR"],
    })


def test_validator_passes_valid_data(valid_df):
    v = DataValidator(required_columns=["alarm_id", "cell_id", "severity"])
    v.validate(valid_df)  # Should not raise


def test_validator_raises_on_empty_df():
    v = DataValidator()
    with pytest.raises(ValidationError, match="empty"):
        v.validate(pd.DataFrame())


def test_validator_raises_on_missing_columns(valid_df):
    v = DataValidator(required_columns=["alarm_id", "missing_col"])
    with pytest.raises(ValidationError, match="Missing"):
        v.validate(valid_df)


def test_validator_raises_on_null_threshold():
    df = pd.DataFrame({
        "alarm_id": ["A1", None, None],  # 66% nulls
        "cell_id": ["C1", "C2", "C3"],
    })
    v = DataValidator(null_threshold=0.05)
    with pytest.raises(ValidationError, match="Null rate"):
        v.validate(df)


def test_validator_repr():
    v = DataValidator(required_columns=["a", "b"], null_threshold=0.1)
    assert "DataValidator" in repr(v)
    assert "null_threshold=0.1" in repr(v)
