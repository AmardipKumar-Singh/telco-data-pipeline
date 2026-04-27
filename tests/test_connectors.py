"""Tests for data connectors, including the new HuggingFaceConnector."""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.connectors.huggingface_connector import HuggingFaceConnector
from src.exceptions import ConnectorError


@pytest.fixture
def hf_config() -> dict:
    return {
        "dataset_id": "cabbage-dog/The-AI-Telco-Troubleshooting-Challenge",
        "split": "train",
        "cache_dir": "/tmp/hf_cache",
        "local_snapshot": "/nonexistent/path",  # forces Hub path in tests
    }


class TestHuggingFaceConnector:
    def test_repr(self, hf_config):
        c = HuggingFaceConnector(config=hf_config)
        assert "HuggingFaceConnector" in repr(c)
        assert "cabbage-dog" in repr(c)

    def test_write_raises(self, hf_config):
        c = HuggingFaceConnector(config=hf_config)
        with pytest.raises(NotImplementedError):
            c.write(pd.DataFrame())

    @patch("src.connectors.huggingface_connector.Path.exists", return_value=False)
    @patch("src.connectors.huggingface_connector.HuggingFaceConnector._load_from_hub")
    def test_read_falls_back_to_hub_when_no_snapshot(
        self, mock_hub, mock_exists, hf_config
    ):
        mock_hub.return_value = pd.DataFrame({"question": ["Q1"], "answer": ["A"]})
        c = HuggingFaceConnector(config=hf_config)
        df = c.read()
        assert isinstance(df, pd.DataFrame)
        mock_hub.assert_called_once()

    @patch("src.connectors.huggingface_connector.Path.exists", return_value=False)
    @patch(
        "src.connectors.huggingface_connector.HuggingFaceConnector._load_from_hub",
        side_effect=ConnectorError("Hub unavailable"),
    )
    def test_read_raises_connector_error_on_failure(
        self, mock_hub, mock_exists, hf_config
    ):
        c = HuggingFaceConnector(config=hf_config)
        with pytest.raises(ConnectorError):
            c.read()
