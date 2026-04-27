"""Abstract base class for all data connectors.

Design decision: We use ABC + abstractmethod instead of duck typing so that
incomplete connector implementations fail loudly at class definition time,
not at runtime when process() is first called.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DataConnector(ABC):
    """Abstract base for all data source and sink connectors.

    All connectors must implement :meth:`read` and :meth:`write`.  The
    :meth:`connect` / :meth:`close` lifecycle hooks are provided with
    default no-ops so subclasses only override what they need.

    Args:
        config: Connector-specific configuration dictionary.
        name: Optional human-readable name used in logs and metrics.
    """

    def __init__(self, config: dict[str, Any], name: Optional[str] = None) -> None:
        self.config = config
        self.name = name or self.__class__.__name__
        self._connected: bool = False
        logger.debug("%s initialised with config keys: %s", self.name, list(config.keys()))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish the underlying connection.  Override in subclasses."""
        self._connected = True
        logger.info("%s connected", self.name)

    def close(self) -> None:
        """Release the underlying connection.  Override in subclasses."""
        self._connected = False
        logger.info("%s connection closed", self.name)

    def __enter__(self) -> "DataConnector":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def read(self, **kwargs) -> Any:
        """Read data from the source.

        Returns:
            Data in a connector-appropriate format (e.g., list[dict],
            pandas DataFrame, Spark DataFrame).
        """

    @abstractmethod
    def write(self, data: Any, **kwargs) -> None:
        """Write data to the sink.

        Args:
            data: Data payload matching what :meth:`read` would return.
        """

    # ------------------------------------------------------------------
    # Dunder helpers for debugging
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, connected={self._connected})"

    def __str__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"[{self.__class__.__name__}:{self.name}] {status}"
