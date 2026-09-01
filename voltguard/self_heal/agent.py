"""Self-healing agent — dispatches corrective actions and logs them.

Maps each (fault_code, root_cause) pair to a specific remediation action,
writes a structured audit log, and publishes events for the feedback loop.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from voltguard.core.events import event_bus
from voltguard.core.logging import get_logger

logger = get_logger(__name__)

# ── Action mappings ───────────────────────────────────────────────────────

_BATTERY_ACTIONS: dict[str, str] = {
    "Low State of Charge": "Recharge battery",
    "Under-voltage": "Recharge battery",
    "Battery Over-temperature": "Trigger battery cooling system",
    "Battery Under-temperature": "Initiate battery pre-heating",
    "Battery Aging": "Schedule battery diagnostics",
    "Battery Fault — unspecified cause": "Schedule complete battery diagnostic",
}

_MOTOR_ACTIONS: dict[str, str] = {
    "Motor Overcurrent": "Throttle motor or inspect controller",
    "Motor Overcurrent (SHAP)": "Throttle motor or inspect controller",
    "Abnormal Motor RPM": "Check motor and sensor connections",
    "Abnormal Motor RPM (SHAP)": "Check motor RPM sensors",
    "Motor Fault — needs inspection": "Throttle motor and run motor diagnostics",
}


def take_action(
    fault_code: int,
    root_cause: str,
    input_data: dict,
    vehicle_id: str = "",
    tick: int = 0,
    source_file: str | None = None,
    log_index: int | None = None,
    log_dir: str | Path = "logs",
) -> str:
    """Determine and execute the corrective action for a fault.

    Args:
        fault_code: Predicted fault code (0, 1, or 2).
        root_cause: Root cause string from the analyzer.
        input_data: Raw sensor data that triggered the fault.
        vehicle_id: Identifier for the vehicle (used in fleet mode).
        tick: Current simulation tick.
        source_file: Optional source file identifier.
        log_index: Optional row index within the source file.
        log_dir: Directory for audit logs.

    Returns:
        The action string that was taken.
    """
    # Determine action
    if fault_code == 0:
        action = "No action needed"
    elif fault_code == 1:
        action = _BATTERY_ACTIONS.get(root_cause, "Schedule complete battery diagnostic")
    elif fault_code == 2:
        action = _MOTOR_ACTIONS.get(root_cause, "Throttle motor and run motor diagnostics")
    else:
        action = "Unknown fault — requires manual inspection"

    # Publish event for the feedback loop
    if fault_code != 0:
        event_bus.publish("action_taken", {
            "fault_code": fault_code,
            "root_cause": root_cause,
            "action": action,
            "vehicle_id": vehicle_id,
            "tick": tick,
            "input_data": input_data,
        })
        logger.info(
            "[%s] Fault=%d | Cause=%s | Action=%s",
            vehicle_id or "unknown", fault_code, root_cause, action,
        )

    # Write audit log
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vehicle_id": vehicle_id,
        "tick": tick,
        "fault_code": fault_code,
        "root_cause": root_cause,
        "action_taken": action,
        "data": input_data,
    }
    if source_file:
        log_entry["source_file"] = source_file
    if log_index is not None:
        log_entry["log_index"] = log_index

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "self_healing_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return action
