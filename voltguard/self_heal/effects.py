"""Self-healing effects — how corrective actions change vehicle state.

This is the feedback loop that makes VoltGuard a *closed-loop* system.
When the agent says "Trigger cooling", this module defines what that
actually DOES to the vehicle's sensor readings over subsequent ticks.
"""

from __future__ import annotations

from config.settings import cfg
from voltguard.core.logging import get_logger

logger = get_logger(__name__)


def apply_healing_effects(state: dict, action: str) -> dict:
    """Apply the effects of a healing action to a vehicle's state.

    This mutates the state dict in-place AND returns it. The state dict
    matches the fields in VehicleState (see ingestion/schema.py).

    Args:
        state: Mutable vehicle state dictionary.
        action: The action string from the self-heal agent.

    Returns:
        The mutated state dictionary.
    """
    h = cfg.healing  # shorthand

    # ── Cooling system ────────────────────────────────────────────────────
    if action == "Trigger battery cooling system":
        state["is_cooling"] = True
        state["cooling_ticks_remaining"] = h.cooling_duration
        logger.info("[%s] Cooling system activated for %d ticks", state.get("vehicle_id", "?"), h.cooling_duration)

    # ── Pre-heating ───────────────────────────────────────────────────────
    if action == "Initiate battery pre-heating":
        state["is_preheating"] = True
        state["preheat_ticks_remaining"] = h.cooling_duration  # same duration
        logger.info("[%s] Pre-heating activated for %d ticks", state.get("vehicle_id", "?"), h.cooling_duration)

    # ── Motor throttle ────────────────────────────────────────────────────
    if action in ("Throttle motor or inspect controller", "Throttle motor and run motor diagnostics"):
        state["is_throttled"] = True
        logger.info("[%s] Motor throttled", state.get("vehicle_id", "?"))

    return state


def tick_healing_effects(state: dict) -> dict:
    """Apply per-tick healing effects to a vehicle's state.

    Called every simulation tick to gradually apply the effects of
    active healing actions (e.g., cooling reduces temperature over time).

    Args:
        state: Mutable vehicle state dictionary.

    Returns:
        The mutated state dictionary.
    """
    h = cfg.healing

    # ── Active cooling: reduce temperature each tick ──────────────────────
    if state.get("is_cooling") and state.get("cooling_ticks_remaining", 0) > 0:
        state["battery_temp"] -= h.cooling_rate
        state["cooling_ticks_remaining"] -= 1
        if state["cooling_ticks_remaining"] <= 0:
            state["is_cooling"] = False
            logger.info("[%s] Cooling cycle complete", state.get("vehicle_id", "?"))

    # ── Active pre-heating: increase temperature each tick ────────────────
    if state.get("is_preheating") and state.get("preheat_ticks_remaining", 0) > 0:
        state["battery_temp"] += h.preheat_rate
        state["preheat_ticks_remaining"] -= 1
        if state["preheat_ticks_remaining"] <= 0:
            state["is_preheating"] = False
            logger.info("[%s] Pre-heating cycle complete", state.get("vehicle_id", "?"))

    # ── Motor throttle: reduce RPM and current ────────────────────────────
    if state.get("is_throttled"):
        state["motor_rpm"] *= (1 - h.throttle_rpm_reduction)
        state["current"] *= (1 - h.throttle_current_reduction)
        # Release throttle once RPM is safe
        if abs(state["motor_rpm"]) < cfg.motor.rpm_overspeed * 0.7 and abs(state["current"]) < cfg.motor.current_overcurrent * 0.7:
            state["is_throttled"] = False
            logger.info("[%s] Motor throttle released", state.get("vehicle_id", "?"))

    # ── SOC recharge (if charging action active) ──────────────────────────
    if state.get("is_charging"):
        state["soc"] = min(100.0, state["soc"] + h.recharge_soc_rate)
        state["voltage"] = 35 + (state["soc"] / 100.0) * (49 - 35)
        if state["soc"] >= 80.0:
            state["is_charging"] = False
            logger.info("[%s] Recharge complete", state.get("vehicle_id", "?"))

    return state
