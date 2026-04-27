"""Entry point for the Telco Data Pipeline.

Usage:
    python main.py --config config/pipeline_config.yaml
    python main.py --config config/pipeline_config.yaml --local

The --local flag bypasses Kafka/Postgres and runs entirely on the local
filesystem using the HuggingFace dataset cloned to data/raw/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.connectors.hdfs_connector import HDFSConnector
from src.connectors.sql_connector import SQLConnector
from src.pipeline_manager import PipelineManager
from src.stages.sql_stage import SQLAggregationStage
from src.stages.spark_stage import SparkTransformationStage
from src.stages.telco_stage import TelcoFeatureEngineeringStage
from src.validator import DataValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


def load_config(path: str) -> dict:
    """Load and return the YAML pipeline configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_telco_dataset(data_path: str) -> "pd.DataFrame":  # noqa: F821
    """Load the AI Telco Troubleshooting dataset.

    Tries HuggingFace ``datasets`` library first (reads from the cloned
    repo), then falls back to scanning for Parquet files directly.
    """
    import pandas as pd

    hf_path = Path(data_path)
    parquet_files = list(hf_path.glob("**/*.parquet"))
    json_files = list(hf_path.glob("**/*.json")) + list(hf_path.glob("**/*.jsonl"))

    if parquet_files:
        logger.info("Loading %d Parquet file(s) from %s", len(parquet_files), hf_path)
        frames = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(frames, ignore_index=True)
    elif json_files:
        logger.info("Loading %d JSON/JSONL file(s) from %s", len(json_files), hf_path)
        frames = [pd.read_json(f, lines=f.suffix == ".jsonl") for f in json_files]
        return pd.concat(frames, ignore_index=True)
    else:
        try:
            from datasets import load_dataset
            ds = load_dataset("cabbage-dog/The-AI-Telco-Troubleshooting-Challenge")
            return ds["train"].to_pandas()
        except Exception as exc:  # noqa: BLE001
            logger.warning("HuggingFace load failed (%s); generating synthetic demo data", exc)
            return _synthetic_telco_data()


def _synthetic_telco_data() -> "pd.DataFrame":  # noqa: F821
    """Generate minimal synthetic telco alarm data for demo/testing."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n = 1000
    severities = ["CRITICAL", "MAJOR", "MINOR", "WARNING", "CLEARED"]
    alarm_types = ["LINK_DOWN", "HIGH_PRB", "LOW_RSRP", "HANDOVER_FAIL", "CELL_OUTAGE"]
    return pd.DataFrame({
        "alarm_id": [f"ALM-{i:05d}" for i in range(n)],
        "cell_id": [f"CELL-{rng.integers(1, 50):03d}" for _ in range(n)],
        "alarm_type": rng.choice(alarm_types, n),
        "severity": rng.choice(severities, n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1min"),
        "rsrp": rng.normal(-90, 15, n),
        "sinr": rng.normal(15, 8, n),
        "prb_utilization": rng.uniform(0.1, 0.95, n),
        "handover_success_rate": rng.uniform(0.7, 1.0, n),
        "resolution_category": rng.choice(["AUTO_CLEARED", "MANUAL", "PENDING", "ESCALATED"], n),
    })


def build_pipeline(config: dict, local: bool = False) -> PipelineManager:
    """Construct and wire up the full pipeline from config.

    Args:
        config: Parsed pipeline YAML config.
        local: If True, skip Kafka/Postgres and run locally.

    Returns:
        A ready-to-run PipelineManager instance.
    """
    stage_cfg = {s["name"]: s.get("config", {}) for s in config["stages"]}

    telco_stage = TelcoFeatureEngineeringStage(
        config=stage_cfg.get("feature_engineering", {})
    )
    spark_stage = SparkTransformationStage(
        spark=None,  # None = pandas fallback in local mode
        config=stage_cfg.get("spark_transform", {}),
    )
    sql_stage = SQLAggregationStage(
        sql_connector=None,  # None = SQLite fallback in local mode
        config=stage_cfg.get("sql_aggregation", {}),
    )

    validator = DataValidator(
        required_columns=stage_cfg.get("validate", {}).get(
            "required_columns",
            ["alarm_id", "cell_id", "alarm_type", "severity", "timestamp"],
        ),
        null_threshold=stage_cfg.get("validate", {}).get("null_threshold", 0.05),
    )

    manager = (
        PipelineManager(max_retries=config["orchestration"]["max_retries"], name="TelcoPipeline")
        .add_stage(telco_stage)
        .add_stage(sql_stage)
    )
    return manager


def main() -> None:
    parser = argparse.ArgumentParser(description="Telco Data Pipeline")
    parser.add_argument("--config", default="config/pipeline_config.yaml", help="Path to YAML config")
    parser.add_argument("--local", action="store_true", help="Run in local mode (no Kafka/Postgres)")
    args = parser.parse_args()

    logger.info("Loading config from %s", args.config)
    config = load_config(args.config)

    logger.info("Loading telco dataset from %s", config["storage"]["local_base_path"].replace("intermediate", "raw"))
    data = load_telco_dataset("./data/raw")
    logger.info("Dataset loaded: %d rows, %d columns", *data.shape)
    logger.info("Columns: %s", list(data.columns))

    pipeline = build_pipeline(config, local=args.local)
    logger.info("Pipeline: %s", pipeline)

    result = pipeline.run(data)

    logger.info("Pipeline complete. Output shape: %s", result.shape)
    logger.info("Run log:")
    for entry in pipeline.run_log:
        status = "✓" if entry.success else "✗"
        logger.info("  %s %s (%.4fs, rows=%s)", status, entry.stage_name, entry.duration_s, entry.output_rows)

    # Save results to intermediate storage
    hdfs = HDFSConnector(config["storage"])
    hdfs.connect()
    hdfs.write(result, "telco_aggregated_results.parquet")
    logger.info("Results written to intermediate storage")


if __name__ == "__main__":
    main()
