"""SQL aggregation stage: runs analytical queries on staged data.

Updated default query for the real Telco Troubleshooting schema:
    Aggregates by scenario_type and answer_letter.
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
    COALESCE(scenario_type, 'other')    AS scenario_type,
    COALESCE(answer_letter, '?')        AS answer_letter,
    COUNT(*)                               AS question_count,
    ROUND(AVG(question_length), 1)         AS avg_question_length,
    ROUND(AVG(question_lines),  2)         AS avg_question_lines,
    ROUND(AVG(num_options),     2)         AS avg_num_options,
    SUM(has_table)                         AS questions_with_table,
    SUM(has_figure)                        AS questions_with_figure
FROM staging_events
GROUP BY scenario_type, answer_letter
ORDER BY question_count DESC
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

    def validate(self, data: Any) -> None:
        if not isinstance(data, pd.DataFrame):
            raise StageError(
                f"{self.name} expects a pandas DataFrame, "
                f"got {type(data).__name__}"
            )
        if data.empty:
            raise StageError(f"{self.name} received an empty DataFrame")

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the configured aggregation query via SQLite.

        Args:
            data: Enriched DataFrame from TelcoFeatureEngineeringStage.

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
