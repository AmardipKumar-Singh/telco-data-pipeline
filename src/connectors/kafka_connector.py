"""Kafka connector: produces alarm events and consumes processed records.

Design decision: KafkaConnector composes (not inherits) kafka-python's
KafkaProducer/KafkaConsumer so we keep a clean DataConnector interface
and can swap the Kafka client library without touching the pipeline code.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Generator, Optional

from src.exceptions import KafkaConnectorError
from .base import DataConnector

try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import KafkaError
except ImportError:  # Allow import without Kafka in unit-test environments
    KafkaConsumer = KafkaProducer = KafkaError = None  # type: ignore

logger = logging.getLogger(__name__)


class KafkaConnector(DataConnector):
    """Bidirectional Kafka connector for telco alarm event streaming.

    Args:
        config: Must contain ``bootstrap_servers``, ``topic_input``,
            ``topic_output``, ``consumer_group``, and ``batch_size``.
        name: Optional human-readable name.
    """

    def __init__(self, config: dict[str, Any], name: Optional[str] = "KafkaConnector") -> None:
        super().__init__(config, name)
        self._producer: Optional[Any] = None
        self._consumer: Optional[Any] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Initialise Kafka producer and consumer."""
        if KafkaProducer is None:
            raise KafkaConnectorError("kafka-python is not installed")
        try:
            self._producer = KafkaProducer(
                bootstrap_servers=self.config["bootstrap_servers"],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
            )
            self._consumer = KafkaConsumer(
                self.config["topic_input"],
                bootstrap_servers=self.config["bootstrap_servers"],
                group_id=self.config["consumer_group"],
                auto_offset_reset=self.config.get("auto_offset_reset", "earliest"),
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                enable_auto_commit=True,
            )
            super().connect()
        except Exception as exc:  # noqa: BLE001
            raise KafkaConnectorError(f"Failed to connect to Kafka: {exc}") from exc

    def close(self) -> None:
        """Flush producer and close consumer."""
        if self._producer:
            self._producer.flush()
            self._producer.close()
        if self._consumer:
            self._consumer.close()
        super().close()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def read(self, batch_size: Optional[int] = None, **kwargs) -> list[dict]:
        """Poll a batch of alarm events from the input topic.

        Args:
            batch_size: Number of messages to poll.  Defaults to
                ``config["batch_size"]``.

        Returns:
            List of deserialized event dictionaries.

        Raises:
            KafkaConnectorError: On poll failure.
        """
        if not self._connected:
            raise KafkaConnectorError("Call connect() before read()")
        n = batch_size or self.config.get("batch_size", 500)
        timeout = self.config.get("poll_timeout_ms", 5000)
        try:
            records = self._consumer.poll(timeout_ms=timeout, max_records=n)
            events = [msg.value for tp_msgs in records.values() for msg in tp_msgs]
            logger.info("%s polled %d events", self.name, len(events))
            return events
        except Exception as exc:  # noqa: BLE001
            raise KafkaConnectorError(f"Read failed: {exc}") from exc

    def write(self, data: list[dict], **kwargs) -> None:
        """Produce processed records to the output topic.

        Args:
            data: List of event dictionaries to publish.

        Raises:
            KafkaConnectorError: On produce failure.
        """
        if not self._connected:
            raise KafkaConnectorError("Call connect() before write()")
        topic = self.config.get("topic_output", "telco_alarms_processed")
        try:
            futures = [self._producer.send(topic, value=record) for record in data]
            for future in futures:
                future.get(timeout=10)  # Block and raise on error
            logger.info("%s produced %d records to %s", self.name, len(data), topic)
        except Exception as exc:  # noqa: BLE001
            raise KafkaConnectorError(f"Write failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Generator for streaming (backpressure-aware)
    # ------------------------------------------------------------------

    def stream(self, max_batches: Optional[int] = None) -> Generator[list[dict], None, None]:
        """Yield batches indefinitely (or up to ``max_batches``).

        Implements backpressure by yielding control after each batch,
        letting the caller throttle ingestion rate.

        Args:
            max_batches: Stop after this many batches.  ``None`` = infinite.

        Yields:
            Lists of event dictionaries, one batch per iteration.
        """
        count = 0
        while max_batches is None or count < max_batches:
            batch = self.read()
            if batch:
                yield batch
            count += 1
