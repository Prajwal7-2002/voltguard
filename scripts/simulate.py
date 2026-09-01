"""CLI fleet simulation with the closed-loop self-healing pipeline.

Usage:
    python scripts/simulate.py
    python scripts/simulate.py --vehicles 3 --ticks 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from voltguard.simulator.fleet import FleetSimulator

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VoltGuard fleet simulation")
    parser.add_argument("--vehicles", type=int, default=3, help="Number of vehicles")
    parser.add_argument("--ticks", type=int, default=20, help="Number of simulation ticks")
    parser.add_argument("--model", type=str, default="latest", help="Model version")
    args = parser.parse_args()

    console.print(f"\n[bold cyan]VoltGuard PMSM Fleet Simulation[/bold cyan]")
    console.print(f"  Vehicles: {args.vehicles}  |  Ticks: {args.ticks}\n")

    fleet = FleetSimulator(
        num_vehicles=args.vehicles,
        model_dir=args.model,
    )

    fault_counts = {0: 0, 1: 0}
    healed_count = 0

    for tick_results in fleet.run(num_ticks=args.ticks):
        tick = tick_results[0]["tick"]

        # Build table for this tick
        table = Table(title=f"Tick {tick}/{args.ticks}", show_lines=False, padding=(0, 1))
        table.add_column("Vehicle", style="cyan", width=8)
        table.add_column("Stator °C", justify="right", width=10)
        table.add_column("Coolant °C", justify="right", width=10)
        table.add_column("Motor RPM", justify="right", width=10)
        table.add_column("Torque", justify="right", width=8)
        table.add_column("Fault", justify="center", width=14)
        table.add_column("Conf", justify="right", width=6)
        table.add_column("Root Cause", width=28)
        table.add_column("Action", width=30)

        for r in tick_results:
            fault_code = r["predicted_fault"]
            fault_counts[fault_code] = fault_counts.get(fault_code, 0) + 1

            # Color-code fault
            if fault_code == 0:
                fault_str = "[green]OK Normal[/green]"
            else:
                fault_str = "[red]! Thermal[/red]"

            if fault_code != 0:
                healed_count += 1

            rd = r["reading"]
            table.add_row(
                r["vehicle_id"],
                f"{rd.get('stator_winding', 0):.1f}",
                f"{rd.get('coolant', 0):.1f}",
                f"{rd.get('motor_speed', 0):.0f}",
                f"{rd.get('torque', 0):.1f}",
                fault_str,
                f"{r['confidence']:.2f}",
                r["root_cause"],
                r["action"],
            )

        console.print(table)

    # Summary
    total = sum(fault_counts.values())
    console.print(f"\n[bold]{'='*60}[/bold]")
    console.print(f"[bold cyan]  Simulation Complete[/bold cyan]")
    console.print(f"  Total predictions: {total}")
    console.print(f"  Normal: [green]{fault_counts.get(0, 0)}[/green]  |  Thermal faults: [red]{fault_counts.get(1, 0)}[/red]")
    console.print(f"  Self-healing actions taken: [bold]{healed_count}[/bold]")
    console.print(f"[bold]{'='*60}[/bold]\n")


if __name__ == "__main__":
    main()
