"""HuggingFace dataset connector for the Telco Troubleshooting pipeline.

Real dataset schema (cabbage-dog/The-AI-Telco-Troubleshooting-Challenge):
    train.csv         — 2 400 rows: ID, question, answer
    phase_1_test.csv  — test split (no answer column)
    SampleSubmission.csv

Inherits from DataConnector so it slots into PipelineManager alongside
KafkaConnector / SQLConnector / HDFSConnector.

Load priority:
  1. Local HF cache snapshot  → scan for *.csv  (fastest, offline)
  2. datasets.load_from_disk  → Arrow format fallback
  3. Hub download             → requires internet
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.connectors.base import DataConnector
from src.exceptions import ConnectorError

logger = logging.getLogger(__name__)

_DEFAULT_SNAPSHOT = (
    "~/.cache/huggingface/hub/"
    "datasets--cabbage-dog--The-AI-Telco-Troubleshooting-Challenge/"
    "snapshots/fa38dde4e55b5a821154c8311b48b4738b92eee7"
)
_HF_DATASET_ID  = "cabbage-dog/The-AI-Telco-Troubleshooting-Challenge"
_DEFAULT_SPLIT  = "train"           # reads train.csv from snapshot
_SPLIT_FILE_MAP = {                 # snapshot CSV filename → split name
    "train":        "train.csv",
    "test":         "phase_1_test.csv",
    "submission":   "SampleSubmission.csv",
}


class HuggingFaceConnector(DataConnector):
    """Load the AI Telco Troubleshooting Challenge dataset into a DataFrame.

    Args:
        config: Connector config dict.  Recognised keys:
            - ``dataset_id``     (str)  HF repo id.
            - ``split``          (str)  'train' | 'test'.  Default: 'train'.
            - ``cache_dir``      (str)  HF cache directory.
            - ``local_snapshot`` (str)  Path to the downloaded snapshot dir.
        name: Human-readable connector name.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str = "HuggingFaceConnector",
    ) -> None:
        super().__init__(config, name)
        self._dataset_id: str  = config.get("dataset_id", _HF_DATASET_ID)
        self._split: str       = config.get("split", _DEFAULT_SPLIT)
        self._cache_dir: Optional[str] = config.get("cache_dir")
        self._snapshot: str    = config.get("local_snapshot", _DEFAULT_SNAPSHOT)

    # ------------------------------------------------------------------
    # DataConnector interface
    # ------------------------------------------------------------------

    def read(self, **kwargs) -> pd.DataFrame:
        """Return the dataset as a pandas DataFrame.

        Tries the local snapshot first (CSV files), then falls back to
        the HuggingFace Hub.

        Returns:
            DataFrame with columns: ID, question, answer
            (answer column absent in the 'test' split).

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
        """Load the correct CSV from the locally downloaded snapshot."""
        logger.info(
            "%s: loading split='%s' from local snapshot %s",
            self.name, self._split, snapshot_path,
        )
        try:
            # ── Priority 1: named CSV matching the split ─────────────
            target_csv = _SPLIT_FILE_MAP.get(self._split)
            if target_csv:
                csv_path = snapshot_path / target_csv
                if csv_path.exists():
                    logger.info("%s: reading %s", self.name, csv_path.name)
                    df = pd.read_csv(csv_path)
                    logger.info(
                        "%s: loaded %d rows, columns=%s",
                        self.name, len(df), list(df.columns),
                    )
                    return df

            # ── Priority 2: any CSV in the snapshot ──────────────────
            csv_files = sorted(snapshot_path.rglob("*.csv"))
            # Exclude SampleSubmission unless explicitly requested
            if self._split != "submission":
                csv_files = [f for f in csv_files
                             if "submission" not in f.name.lower()]
            if csv_files:
                logger.info(
                    "%s: found %d CSV(s), loading %s",
                    self.name, len(csv_files), csv_files[0].name,
                )
                frames = [pd.read_csv(f) for f in csv_files]
                df = pd.concat(frames, ignore_index=True)
                logger.info(
                    "%s: loaded %d rows, columns=%s",
                    self.name, len(df), list(df.columns),
                )
                return df

            # ── Priority 3: Arrow / load_from_disk fallback ──────────
            from datasets import load_from_disk
            logger.info("%s: no CSV found — trying load_from_disk", self.name)
            ds = load_from_disk(str(snapshot_path))
            split_ds = ds[self._split] if hasattr(ds, "__getitem__") else ds
            df = split_ds.to_pandas()
            logger.info(
                "%s: loaded %d rows via load_from_disk", self.name, len(df)
            )
            return df

        except Exception as exc:
            raise ConnectorError(
                f"{self.name}: failed to load snapshot at "
                f"{snapshot_path}: {exc}"
            ) from exc

    def _load_from_hub(self) -> pd.DataFrame:
        """Stream the dataset from the HuggingFace Hub."""
        try:
            from datasets import load_dataset
            logger.info(
                "%s: downloading %s [%s] from Hub",
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
