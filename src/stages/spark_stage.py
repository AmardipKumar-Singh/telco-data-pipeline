"""PySpark transformation stage.

Design decision: SparkTransformationStage accepts a *list* of operation
dictionaries (from pipeline_config.yaml) rather than hard-coded Spark code.
This makes the stage data-driven and testable: operations can be swapped
in config without touching Python source.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.exceptions import StageError
from .base import PipelineStage

try:
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
except ImportError:
    SparkSession = DataFrame = F = Window = None  # type: ignore

logger = logging.getLogger(__name__)


class SparkTransformationStage(PipelineStage):
    """Applies configurable Spark DataFrame transformations.

    Supported operations (configured in pipeline_config.yaml):
    - ``filter``: Apply a SQL-style WHERE condition.
    - ``window_agg``: Compute rolling aggregations over a time window.
    - ``drop_nulls``: Drop rows where any configured column is null.

    Args:
        spark: Active SparkSession.
        name: Stage name.
        config: Dict with ``operations`` list and optional Spark settings.
    """

    def __init__(
        self,
        spark: Optional[Any],
        name: str = "SparkTransformationStage",
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self.spark = spark

    # ------------------------------------------------------------------
    # PipelineStage interface
    # ------------------------------------------------------------------

    def validate(self, data: Any) -> None:
        """Ensure we have a non-empty Spark DataFrame."""
        if DataFrame is None:
            raise StageError("PySpark is not installed")
        if not isinstance(data, DataFrame):
            raise StageError(f"{self.name} expects a Spark DataFrame, got {type(data).__name__}")
        if data.rdd.isEmpty():
            logger.warning("%s received an empty DataFrame", self.name)

    def process(self, data: Any) -> Any:
        """Apply all configured operations sequentially.

        Args:
            data: Input Spark DataFrame.

        Returns:
            Transformed Spark DataFrame.
        """
        df: DataFrame = data
        for op in self.config.get("operations", []):
            op_type = op.get("type")
            logger.debug("%s applying op: %s", self.name, op_type)
            if op_type == "filter":
                df = df.filter(op["condition"])
            elif op_type == "window_agg":
                df = self._window_agg(df, op)
            elif op_type == "drop_nulls":
                df = df.dropna(subset=op.get("columns"))
            else:
                logger.warning("%s unknown operation type: %s", self.name, op_type)
        self._metrics["output_rows"] = df.count()
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _window_agg(
        self, df: Any, op: dict[str, Any]
    ) -> Any:
        """Apply a time-based rolling window aggregation."""
        window_minutes = op.get("window_minutes", 60)
        partition_cols = op.get("partition_by", [])
        order_col = op.get("order_by", "timestamp")
        window_spec = (
            Window.partitionBy(*partition_cols)
            .orderBy(F.col(order_col).cast("long"))
            .rangeBetween(-window_minutes * 60, 0)
        )
        df = df.withColumn("alarm_count_window", F.count("alarm_id").over(window_spec))
        df = df.withColumn("avg_severity_window", F.avg("severity_score").over(window_spec))
        return df
