"""Custom exceptions for the Telco Data Pipeline.

Defines a clear exception hierarchy so callers can catch at the right granularity.
All pipeline-specific errors inherit from PipelineError, making it easy to wrap
generic exceptions at integration boundaries without losing information.
"""


class PipelineError(Exception):
    """Base class for all pipeline exceptions."""


class ConnectorError(PipelineError):
    """Raised when a DataConnector fails to read or write data."""


class KafkaConnectorError(ConnectorError):
    """Kafka-specific connector failure."""


class SQLConnectorError(ConnectorError):
    """SQL database connector failure."""


class HDFSConnectorError(ConnectorError):
    """HDFS / local filesystem connector failure."""


class StageError(PipelineError):
    """Raised when a PipelineStage fails during processing."""


class ValidationError(PipelineError):
    """Raised when DataValidator finds schema or quality violations."""


class OrchestrationError(PipelineError):
    """Raised by PipelineManager on unrecoverable orchestration failure."""
