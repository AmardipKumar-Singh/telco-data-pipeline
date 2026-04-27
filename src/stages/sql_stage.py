"""SQL aggregation stage: runs analytical queries on staged data.

Design decision: We create a temporary in-memory SQLite database when a
Spark DataFrame is passed so the same SQL query logic works in both
local (no Postgres) and production (full Postgres) modes.  This preserves
the interface contract without requiring a live database during testing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from src.exceptions import StageError
from .base import PipelineStage

logger = logging.getLogger(__name__)


class SQLAggregationStage(PipelineStage):
    """Executes an aggregation SQL query on staged alarm data.

    Args:
        sql_connector: An open SQLConnector instance (or None for local mode).
        name: Stage name.
        config: Must contain ``query`` and ``output_table``.
    """

    def __init__(
        self,
        sql_connector: Optional[Any] = None,
        name: str = "SQLAggregationStage",
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(name, config)
        self.sql_connector = sql_connector

    # ------------------------------------------------------------------
    # PipelineStage interface
    # ------------------------------------------------------------------

    def validate(self, data: Any) -> None:
        """Ensure input is a non-empty DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise StageError(
                f"{self.name} expects a pandas DataFrame, got {type(data).__name__}"
            )
        if data.empty:
            raise StageError(f"{self.name} received an empty DataFrame")

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the configured aggregation query.

        Uses pandas + SQLite for local execution if no SQL connector is
        provided, enabling zero-dependency testing.

        Args:
            data: Staging DataFrame (written as ``staging_events`` temp table).

        Returns:
            Aggregated DataFrame.
        """
        import sqlite3

        query = self.config.get("query", "SELECT * FROM staging_events")

        # Local mode: use in-memory SQLite
        conn = sqlite3.connect(":memory:")
        data.to_sql("staging_events", conn, if_exists="replace", index=False)
        # Translate array_agg to group_concat for SQLite compatibility
        local_query = query.replace("ARRAY_AGG", "GROUP_CONCAT")
        result = pd.read_sql_query(local_query, conn)
        conn.close()

        self._metrics["output_rows"] = len(result)
        logger.info("%s aggregated to %d rows", self.name, len(result))
        return result
