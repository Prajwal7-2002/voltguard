"""Lightweight in-process event bus for VoltGuard.

Allows decoupled communication between components (e.g., the predictor
publishes a "fault_detected" event and the dashboard subscribes to it).

Usage:
    from voltguard.core.events import event_bus

    # Subscribe
    event_bus.subscribe("fault_detected", my_handler)

    # Publish
    event_bus.publish("fault_detected", {"vehicle_id": "v1", "fault_code": 1})
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Type alias for event handler functions
EventHandler = Callable[[dict[str, Any]], None]


class EventBus:
    """Simple pub/sub event bus for in-process communication."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._history: list[dict[str, Any]] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Register a handler for an event type."""
        self._subscribers[event_type].append(handler)
        logger.debug("Subscribed %s to '%s'", handler.__name__, event_type)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a handler for an event type."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    def publish(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Publish an event to all subscribers."""
        event = {
            "type": event_type,
            "data": data or {},
        }
        self._history.append(event)

        for handler in self._subscribers.get(event_type, []):
            try:
                handler(event["data"])
            except Exception:
                logger.exception("Handler %s failed for event '%s'", handler.__name__, event_type)

    def get_history(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Return event history, optionally filtered by type."""
        if event_type is None:
            return list(self._history)
        return [e for e in self._history if e["type"] == event_type]

    def clear(self) -> None:
        """Reset all subscribers and history."""
        self._subscribers.clear()
        self._history.clear()


# Module-level singleton
event_bus = EventBus()
