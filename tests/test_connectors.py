"""Tests for data connectors — HuggingFaceConnector with CSV-based snapshot."""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.connectors.huggingface_connector import HuggingFaceConnector
from src.exceptions import ConnectorError


@pytest.fixture
def hf_config() -> dict:
    return {
        "dataset_id": "cabbage-dog/The-AI-Telco-Troubleshooting-Challenge",
        "split": "train",
        "cache_dir": "/tmp/hf_cache",
        "local_snapshot": "/nonexistent/path",
    }


class TestHuggingFaceConnector:
    def test_repr(self, hf_config):
        c = HuggingFaceConnector(config=hf_config)
        assert "HuggingFaceConnector" in repr(c)
        assert "cabbage-dog" in repr(c)

    def test_str(self, hf_config):
        c = HuggingFaceConnector(config=hf_config)
        assert "train" in str(c)

    def test_write_raises_not_implemented(self, hf_config):
        c = HuggingFaceConnector(config=hf_config)
        with pytest.raises(NotImplementedError):
            c.write(pd.DataFrame())

    @patch("src.connectors.huggingface_connector.Path.exists", return_value=False)
    @patch("src.connectors.huggingface_connector.HuggingFaceConnector._load_from_hub")
    def test_falls_back_to_hub_when_no_snapshot(self, mock_hub, mock_exists, hf_config):
        mock_hub.return_value = pd.DataFrame({
            "ID": ["ID_001"], "question": ["Q?"], "answer": ["C2"]
        })
        c = HuggingFaceConnector(config=hf_config)
        df = c.read()
        assert isinstance(df, pd.DataFrame)
        mock_hub.assert_called_once()

    @patch("src.connectors.huggingface_connector.Path.exists", return_value=True)
    @patch("src.connectors.huggingface_connector.HuggingFaceConnector._load_from_snapshot")
    def test_reads_from_snapshot_when_exists(self, mock_snap, mock_exists, hf_config):
        mock_snap.return_value = pd.DataFrame({
            "ID": ["ID_001"], "question": ["Q?"], "answer": ["C2"]
        })
        c = HuggingFaceConnector(config=hf_config)
        df = c.read()
        mock_snap.assert_called_once()
        assert "ID" in df.columns

    @patch("src.connectors.huggingface_connector.Path.exists", return_value=False)
    @patch(
        "src.connectors.huggingface_connector.HuggingFaceConnector._load_from_hub",
        side_effect=ConnectorError("Hub unavailable"),
    )
    def test_raises_connector_error_on_failure(self, mock_hub, mock_exists, hf_config):
        c = HuggingFaceConnector(config=hf_config)
        with pytest.raises(ConnectorError):
            c.read()
