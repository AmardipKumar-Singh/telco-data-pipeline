"""Unit tests for DataConnector implementations."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.connectors.base import DataConnector
from src.connectors.hdfs_connector import HDFSConnector
from src.exceptions import HDFSConnectorError, ConnectorError


# ---------------------------------------------------------------------------
# DataConnector (abstract base) tests
# ---------------------------------------------------------------------------

class ConcreteConnector(DataConnector):
    """Minimal concrete connector for testing the ABC contract."""
    def read(self, **kwargs): return []
    def write(self, data, **kwargs): pass


def test_connector_repr_and_str():
    c = ConcreteConnector({"key": "val"}, name="test")
    assert "ConcreteConnector" in repr(c)
    assert "test" in str(c)


def test_connector_context_manager():
    c = ConcreteConnector({})
    with c as conn:
        assert conn._connected is True
    assert c._connected is False


def test_abstract_connector_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DataConnector({})  # type: ignore


# ---------------------------------------------------------------------------
# HDFSConnector tests
# ---------------------------------------------------------------------------

@pytest.fixture
def local_hdfs(tmp_path):
    cfg = {"mode": "local", "local_base_path": str(tmp_path), "parquet_compression": "snappy"}
    conn = HDFSConnector(cfg)
    conn.connect()
    return conn, tmp_path


def test_hdfs_write_and_read(local_hdfs):
    conn, tmp_path = local_hdfs
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    conn.write(df, "test.parquet")
    result = conn.read("test.parquet")
    pd.testing.assert_frame_equal(df, result)


def test_hdfs_read_missing_file_raises(local_hdfs):
    conn, _ = local_hdfs
    with pytest.raises(HDFSConnectorError):
        conn.read("nonexistent.parquet")


def test_hdfs_repr(local_hdfs):
    conn, _ = local_hdfs
    assert "HDFSConnector" in repr(conn)
