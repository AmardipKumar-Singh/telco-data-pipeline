"""SQL aggregation stage: runs analytical queries on staged data.

Design decision: We create a temporary in-memory SQLite database when no
SQL connector is provided so the same query logic works in both local
(no Postgres) and production (full Postgres) modes.

Updated default aggregation query for the AI Telco Troubleshooting
Challenge dataset — aggregates by category and difficulty instead of
the former alarm_type / cell_id columns.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from src.exceptions import StageError
from .base import PipelineStage

logger = logging.getLogger(__name__)

_DEFAULT_QUERY = """
SELECT
    COALESCE(category, 'uncategorised') AS category,
    COALESCE(difficulty, 'unknown')    AS difficulty,
    COUNT(*)                              AS question_count,
    ROUND(AVG(question_length), 1)        AS avg_question_length,
    ROUND(AVG(num_choices), 2)            AS avg_num_choices,
    ROUND(AVG(context_length), 1)         AS avg_context_length,
    SUM(has_context)                      AS questions_with_context,
    ROUND(
        100.0 * SUM(CASE WHEN answer_index >= 0 THEN 1 ELSE 0 END) / COUNT(*),
        2
    )                                     AS answer_match_pct
FROM staging_events
GROUP BY category, difficulty
ORDER BY category, difficulty
"""


class SQLAggregationStage(PipelineStage):
    """Executes an aggregation SQL query on staged Telco QA data.

    Args:
        sql_connector: An open SQLConnector instance (or None for local mode).
        name:          Stage name.
        config:        May contain ``query`` and ``output_table`` keys.
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
                f"{self.name} expects a pandas DataFrame, "
                f"got {type(data).__name__}"
            )
        if data.empty:
            raise StageError(f"{self.name} received an empty DataFrame")

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the configured aggregation query.

        Uses pandas + SQLite for local execution when no SQL connector is
        provided, enabling zero-dependency testing.

        Args:
            data: Enriched DataFrame from TelcoFeatureEngineeringStage
                  (written as ``staging_events`` temp table).

        Returns:
            Aggregated DataFrame.
        """
        import sqlite3

        query = self.config.get("query", _DEFAULT_QUERY)

        conn = sqlite3.connect(":memory:")
        try:
            data.to_sql("staging_events", conn, if_exists="replace", index=False)
            result = pd.read_sql_query(query, conn)
        finally:
            conn.close()

        self._metrics["output_rows"] = len(result)
        logger.info("%s aggregated to %d rows", self.name, len(result))
        return result
