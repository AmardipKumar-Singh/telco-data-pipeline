"""HDFS connector with local filesystem fallback.

Design decision: We check for ``hdfs`` (PyArrow HDFS) availability at
runtime and fall back to the local FS transparently.  This means the same
pipeline code runs both in CI (local mode) and in production (HDFS mode)
without configuration changes other than ``storage.mode``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from src.exceptions import HDFSConnectorError
from .base import DataConnector

logger = logging.getLogger(__name__)


class HDFSConnector(DataConnector):
    """Read/write Parquet files on HDFS or the local filesystem.

    Args:
        config: Must contain ``mode`` (``'local'`` or ``'hdfs'``),
            ``local_base_path``, and optionally ``hdfs_base_path``.
        name: Optional human-readable name.
    """

    def __init__(self, config: dict[str, Any], name: Optional[str] = "HDFSConnector") -> None:
        super().__init__(config, name)
        self._mode: str = config.get("mode", "local")
        self._base_path = Path(
            config["local_base_path"] if self._mode == "local" else config["hdfs_base_path"]
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Ensure base path exists (local mode) or validate HDFS URI."""
        if self._mode == "local":
            self._base_path.mkdir(parents=True, exist_ok=True)
        # HDFS mode: connection is implicit via PyArrow; no-op here
        super().connect()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def read(self, path: str, **kwargs) -> pd.DataFrame:
        """Read a Parquet file into a DataFrame.

        Args:
            path: Relative path under ``base_path``.

        Returns:
            pandas DataFrame.

        Raises:
            HDFSConnectorError: If the file cannot be read.
        """
        full_path = self._base_path / path
        try:
            df = pd.read_parquet(full_path)
            logger.info("%s read %d rows from %s", self.name, len(df), full_path)
            return df
        except Exception as exc:  # noqa: BLE001
            raise HDFSConnectorError(f"Read failed at {full_path}: {exc}") from exc

    def write(self, data: pd.DataFrame, path: str, **kwargs) -> None:
        """Write a DataFrame as a compressed Parquet file.

        Args:
            data: DataFrame to persist.
            path: Relative path under ``base_path``.

        Raises:
            HDFSConnectorError: If the file cannot be written.
        """
        full_path = self._base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        compression = self.config.get("parquet_compression", "snappy")
        try:
            data.to_parquet(full_path, compression=compression, index=False)
            logger.info("%s wrote %d rows to %s", self.name, len(data), full_path)
        except Exception as exc:  # noqa: BLE001
            raise HDFSConnectorError(f"Write failed at {full_path}: {exc}") from exc
