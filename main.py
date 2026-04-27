"""Entry point for the Telco Troubleshooting Data Pipeline.

Usage:
    python main.py
    python main.py --config config/pipeline_config.yaml
    python main.py --config config/pipeline_config.yaml --local

The --local flag bypasses Kafka/Postgres and runs entirely on the local
filesystem using the HuggingFace dataset cached at the snapshot path
defined in config/pipeline_config.yaml.

Dataset:
    cabbage-dog/The-AI-Telco-Troubleshooting-Challenge
    Local snapshot: ~/.cache/huggingface/hub/datasets--cabbage-dog--...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.connectors.hdfs_connector import HDFSConnector
from src.connectors.huggingface_connector import HuggingFaceConnector
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


def build_pipeline(config: dict, local: bool = False) -> PipelineManager:
    """Construct and wire up the full pipeline from config.

    Args:
        config: Parsed pipeline YAML config.
        local:  If True, skip Kafka/Postgres and run locally.

    Returns:
        A ready-to-run PipelineManager instance.
    """
    stage_cfg = {s["name"]: s.get("config", {}) for s in config["stages"]}

    telco_stage = TelcoFeatureEngineeringStage(
        config=stage_cfg.get("feature_engineering", {})
    )
    sql_stage = SQLAggregationStage(
        sql_connector=None,   # None → SQLite fallback in local mode
        config=stage_cfg.get("sql_aggregation", {}),
    )

    manager = (
        PipelineManager(
            max_retries=config["orchestration"]["max_retries"],
            name="TelcoTroubleshootingPipeline",
        )
        .add_stage(telco_stage)
        .add_stage(sql_stage)
    )
    return manager


def main() -> None:
    parser = argparse.ArgumentParser(description="Telco Troubleshooting Data Pipeline")
    parser.add_argument(
        "--config", default="config/pipeline_config.yaml",
        help="Path to YAML config"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run in local mode (no Kafka/Postgres)"
    )
    args = parser.parse_args()

    logger.info("Loading config from %s", args.config)
    config = load_config(args.config)

    # ── 1. Ingest ──────────────────────────────────────────────────────
    hf_cfg = config["huggingface"]
    connector = HuggingFaceConnector(config=hf_cfg)
    connector.connect()
    data = connector.read()
    logger.info(
        "Dataset loaded: %d rows, %d columns — %s",
        *data.shape, list(data.columns),
    )

    # ── 2. Validate ────────────────────────────────────────────────────
    val_cfg = next(
        (s.get("config", {}) for s in config["stages"] if s["name"] == "validate"),
        {}
    )
    validator = DataValidator(
        required_columns=val_cfg.get("required_columns", ["question", "answer"]),
        null_threshold=val_cfg.get("null_threshold", 0.05),
    )
    validator.validate(data)

    # ── 3. Feature engineering + SQL aggregation via PipelineManager ───
    pipeline = build_pipeline(config, local=args.local)
    logger.info("Pipeline: %s", pipeline)
    result = pipeline.run(data)

    logger.info("Pipeline complete. Output shape: %s", result.shape)
    logger.info("Run log:")
    for entry in pipeline.run_log:
        status = "✓" if entry.success else "✗"
        logger.info(
            "  %s %s (%.4fs, rows=%s)",
            status, entry.stage_name, entry.duration_s, entry.output_rows,
        )

    # ── 4. Persist results ─────────────────────────────────────────────
    hdfs = HDFSConnector(config["storage"])
    hdfs.connect()
    hdfs.write(result, "telco_qa_aggregated_results.parquet")
    logger.info("Results written to intermediate storage.")


if __name__ == "__main__":
    main()
