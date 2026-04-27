"""SQL connector with connection pooling and upsert support.

Design decision: SQLConnector wraps SQLAlchemy (not raw psycopg2) so the
same class can target PostgreSQL, MySQL, or SQLite by changing the DSN —
useful for local testing without a running Postgres instance.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

from src.exceptions import SQLConnectorError
from .base import DataConnector

logger = logging.getLogger(__name__)


class SQLConnector(DataConnector):
    """PostgreSQL connector with connection pooling.

    Supports full-replace writes (``mode='replace'``) and upserts
    (``mode='upsert'``) keyed on ``conflict_columns``.

    Args:
        config: Must contain ``host``, ``port``, ``database``, ``user``,
            ``password``, ``pool_size``, and ``max_overflow``.
        name: Optional human-readable name.
    """

    def __init__(self, config: dict[str, Any], name: Optional[str] = "SQLConnector") -> None:
        super().__init__(config, name)
        self._engine = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Create a pooled SQLAlchemy engine."""
        dsn = (
            f"postgresql+psycopg2://{self.config['user']}:{self.config['password']}"
            f"@{self.config['host']}:{self.config.get('port', 5432)}/{self.config['database']}"
        )
        try:
            self._engine = create_engine(
                dsn,
                poolclass=QueuePool,
                pool_size=self.config.get("pool_size", 10),
                max_overflow=self.config.get("max_overflow", 5),
                pool_pre_ping=True,  # Verify connection health on checkout
            )
            # Verify connectivity
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            super().connect()
        except SQLAlchemyError as exc:
            raise SQLConnectorError(f"DB connection failed: {exc}") from exc

    def close(self) -> None:
        """Dispose the connection pool."""
        if self._engine:
            self._engine.dispose()
        super().close()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def read(self, query: str, **kwargs) -> pd.DataFrame:
        """Execute ``query`` and return results as a DataFrame.

        Args:
            query: SQL SELECT statement.

        Returns:
            pandas DataFrame with query results.

        Raises:
            SQLConnectorError: On query execution failure.
        """
        try:
            df = pd.read_sql(query, self._engine)
            logger.info("%s read %d rows", self.name, len(df))
            return df
        except SQLAlchemyError as exc:
            raise SQLConnectorError(f"Read failed: {exc}") from exc

    def write(
        self,
        data: pd.DataFrame,
        table: str,
        mode: str = "append",
        conflict_columns: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """Write a DataFrame to ``table``.

        Args:
            data: DataFrame to persist.
            table: Target table name.
            mode: ``'replace'`` truncates then inserts; ``'append'`` inserts;
                ``'upsert'`` uses ON CONFLICT DO UPDATE.
            conflict_columns: Key columns for upsert conflict resolution.

        Raises:
            SQLConnectorError: On write failure.
        """
        try:
            if mode in ("replace", "append"):
                pd_mode = "replace" if mode == "replace" else "append"
                data.to_sql(table, self._engine, if_exists=pd_mode, index=False, method="multi")
            elif mode == "upsert":
                self._upsert(data, table, conflict_columns or [])
            logger.info("%s wrote %d rows to %s (mode=%s)", self.name, len(data), table, mode)
        except SQLAlchemyError as exc:
            raise SQLConnectorError(f"Write failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _upsert(self, df: pd.DataFrame, table: str, conflict_columns: list[str]) -> None:
        """PostgreSQL INSERT ... ON CONFLICT DO UPDATE upsert."""
        cols = list(df.columns)
        update_cols = [c for c in cols if c not in conflict_columns]
        placeholders = ", ".join([f":{c}" for c in cols])
        conflict_target = ", ".join(conflict_columns)
        updates = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])
        sql = (
            f"INSERT INTO {table} ({', '.join(cols)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_target}) DO UPDATE SET {updates}"
        )
        with self._engine.begin() as conn:
            conn.execute(text(sql), df.to_dict(orient="records"))
