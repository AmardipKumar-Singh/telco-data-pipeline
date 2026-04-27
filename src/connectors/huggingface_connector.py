"""HuggingFace dataset connector for the Telco Troubleshooting pipeline.

Inherits from DataConnector so it slots seamlessly into PipelineManager
alongside the existing KafkaConnector / SQLConnector / HDFSConnector.

OOP decision: load-from-local-snapshot is the *primary* path (fast, offline);
streaming from the Hub is the fallback so the pipeline still works in CI/CD
environments that have internet access but no pre-cached copy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.connectors.base import DataConnector
from src.exceptions import ConnectorError

logger = logging.getLogger(__name__)

# Default local snapshot deposited by `hf download`
_DEFAULT_SNAPSHOT = (
    "~/.cache/huggingface/hub/"
    "datasets--cabbage-dog--The-AI-Telco-Troubleshooting-Challenge/"
    "snapshots/fa38dde4e55b5a821154c8311b48b4738b92eee7"
)
_HF_DATASET_ID = "cabbage-dog/The-AI-Telco-Troubleshooting-Challenge"


class HuggingFaceConnector(DataConnector):
    """Load the AI Telco Troubleshooting Challenge dataset into a DataFrame.

    Priority:
    1. ``local_snapshot`` path (fastest — no network required).
    2. ``datasets.load_dataset`` from the HuggingFace Hub.

    Args:
        config: Connector config dict.  Recognised keys:
            - ``dataset_id``     (str)  HF repo id.
            - ``split``          (str)  Dataset split, default ``"train"``.
            - ``cache_dir``      (str)  HF cache directory.
            - ``local_snapshot`` (str)  Absolute path to the snapshot dir.
        name: Human-readable connector name.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str = "HuggingFaceConnector",
    ) -> None:
        super().__init__(config, name)
        self._dataset_id: str = config.get("dataset_id", _HF_DATASET_ID)
        self._split: str = config.get("split", "train")
        self._cache_dir: Optional[str] = config.get("cache_dir")
        self._snapshot: str = config.get("local_snapshot", _DEFAULT_SNAPSHOT)

    # ------------------------------------------------------------------
    # DataConnector interface
    # ------------------------------------------------------------------

    def read(self, **kwargs) -> pd.DataFrame:
        """Return the dataset as a pandas DataFrame.

        Tries the local snapshot first; falls back to Hub download.

        Returns:
            DataFrame with at minimum columns: question, answer, choices,
            context, difficulty (actual columns depend on dataset version).

        Raises:
            ConnectorError: If neither load path succeeds.
        """
        snapshot_path = Path(self._snapshot).expanduser()
        if snapshot_path.exists():
            return self._load_from_snapshot(snapshot_path)
        logger.warning(
            "%s: snapshot not found at %s — falling back to Hub download",
            self.name, snapshot_path,
        )
        return self._load_from_hub()

    def write(self, data: Any, **kwargs) -> None:
        """Not supported — HuggingFace connector is read-only."""
        raise NotImplementedError(
            f"{self.name} does not support write operations."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_snapshot(self, snapshot_path: Path) -> pd.DataFrame:
        """Load from the locally downloaded HuggingFace snapshot."""
        logger.info("%s: loading from local snapshot %s", self.name, snapshot_path)
        parquet_files = sorted(snapshot_path.rglob("*.parquet"))
        json_files = sorted(snapshot_path.rglob("*.jsonl")) + sorted(snapshot_path.rglob("*.json"))

        try:
            if parquet_files:
                logger.info("%s: found %d parquet file(s)", self.name, len(parquet_files))
                frames = [pd.read_parquet(p) for p in parquet_files]
                df = pd.concat(frames, ignore_index=True)
            elif json_files:
                logger.info("%s: found %d JSON/JSONL file(s)", self.name, len(json_files))
                frames = [
                    pd.read_json(p, lines=p.suffix == ".jsonl") for p in json_files
                ]
                df = pd.concat(frames, ignore_index=True)
            else:
                # Try datasets.load_from_disk (Arrow format)
                from datasets import load_from_disk
                ds = load_from_disk(str(snapshot_path))
                split_ds = ds[self._split] if hasattr(ds, "__getitem__") else ds
                df = split_ds.to_pandas()

            logger.info(
                "%s: loaded %d rows, %d columns from snapshot",
                self.name, len(df), len(df.columns),
            )
            return df
        except Exception as exc:
            raise ConnectorError(
                f"{self.name}: failed to load snapshot at {snapshot_path}: {exc}"
            ) from exc

    def _load_from_hub(self) -> pd.DataFrame:
        """Stream dataset from the HuggingFace Hub."""
        try:
            from datasets import load_dataset
            logger.info(
                "%s: streaming %s [%s] from Hub",
                self.name, self._dataset_id, self._split,
            )
            ds = load_dataset(
                self._dataset_id,
                split=self._split,
                cache_dir=self._cache_dir,
            )
            df = ds.to_pandas()
            logger.info(
                "%s: hub download complete — %d rows", self.name, len(df)
            )
            return df
        except Exception as exc:
            raise ConnectorError(
                f"{self.name}: HuggingFace Hub load failed for "
                f"'{self._dataset_id}': {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"HuggingFaceConnector(dataset_id={self._dataset_id!r}, "
            f"split={self._split!r})"
        )

    def __str__(self) -> str:
        return f"[HFConnector:{self._dataset_id}:{self._split}]"
