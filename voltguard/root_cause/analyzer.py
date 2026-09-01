"""Root cause analysis engine with automated work order recommendations.

Combines domain rules with SHAP feature importance to determine the
specific root cause of a predicted fault, recommend an action, and
suggest the exact spare part to order.

Research-backed: "Don't just provide a dashboard; provide an alert that
tells them exactly which spare part to order and when." — IJFMR 2026
"""

from __future__ import annotations

from voltguard.core.logging import get_logger

logger = get_logger(__name__)


def resolve_fault(
    fault_code: int,
    shap_values: dict[str, float],
    input_data: dict[str, float],
) -> tuple[str, str]:
    """Determine root cause, recommended action, and spare part for a fault.

    Returns:
        Tuple of (root_cause_description, action_and_part_recommendation)
    """
    if not shap_values:
        return "Unknown Fault", "Reduce throttle and schedule manual inspection"

    dominant_feature = max(shap_values, key=lambda k: abs(float(shap_values[k])))

    # ── Code 0: Nominal ───────────────────────────────────────────────────
    if fault_code == 0:
        return "No Fault", "No action needed"

    # ── Code 1: Stator Winding Overheat ───────────────────────────────────
    if fault_code == 1:
        motor_speed = input_data.get("motor_speed", 0)
        coolant = input_data.get("coolant", 25)

        if motor_speed > 4000 and coolant > 50:
            return (
                "Sustained High-RPM Stator Overheat",
                "Derate motor torque by 40%. Part: Stator winding coil assembly"
            )
        if dominant_feature == "torque":
            return (
                "Excessive Torque Load on Stator",
                "Limit peak torque output. Part: Stator winding coil assembly"
            )
        return (
            "Stator Thermal Stress (General)",
            "Reduce motor load. Part: Copper winding assembly + thermal paste"
        )

    # ── Code 2: Battery Thermal Stress ────────────────────────────────────
    if fault_code == 2:
        power = input_data.get("power_draw", 0)
        ambient = input_data.get("ambient", 30)

        if ambient > 40:
            return (
                "Battery Overheating (Indian Summer Conditions)",
                "Throttle discharge rate by 60%. Part: Battery thermal pad + cooling fan"
            )
        if abs(power) > 500:
            return (
                "Battery Thermal Stress from High Power Draw",
                "Reduce acceleration intensity. Part: Battery cell thermal management kit"
            )
        return (
            "Battery Thermal Anomaly",
            "Schedule battery thermal paste inspection. Part: Thermal interface material"
        )

    # ── Code 3: Coolant System Failure ────────────────────────────────────
    if fault_code == 3:
        coolant = input_data.get("coolant", 25)
        motor_speed = input_data.get("motor_speed", 0)

        if coolant > 80:
            return (
                "Critical Coolant Pump Failure",
                "Activate backup pump immediately. Part: Coolant pump unit + radiator flush kit"
            )
        if motor_speed < 100:
            return (
                "Coolant Stagnation (Pump Dead at Idle)",
                "Alert maintenance crew. Part: Coolant pump motor + flow sensor"
            )
        return (
            "Cooling System Degradation",
            "Schedule coolant system check. Part: Radiator + coolant hose assembly"
        )

    # ── Code 4: Inverter Over-Current ─────────────────────────────────────
    if fault_code == 4:
        i_q = input_data.get("i_q", 0)
        i_d = input_data.get("i_d", 0)

        if abs(i_q) > abs(i_d):
            return (
                "Inverter Over-Current (Q-axis Spike)",
                "Cut inverter current limits. Part: IGBT power module"
            )
        return (
            "Inverter Over-Current (D-axis Spike)",
            "Reduce regenerative braking intensity. Part: IGBT module + DC-link capacitor"
        )

    return "Unknown Fault Code", "Needs manual review"
