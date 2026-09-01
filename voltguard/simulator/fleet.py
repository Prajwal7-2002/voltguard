"""Fleet simulator — orchestrates multiple VehicleSimulators.

Runs N vehicles in parallel, feeds their readings through the full
pipeline (predict → root-cause → self-heal → feedback), and collects
results for the dashboard.
"""

from __future__ import annotations

from voltguard.core.logging import get_logger
from voltguard.diagnostics.predictor import FaultPredictor
from voltguard.root_cause.analyzer import resolve_fault
from voltguard.simulator.vehicle import VehicleSimulator

logger = get_logger(__name__)


class FleetSimulator:
    """Manages a fleet of simulated EV vehicles using PMSM playback.

    Usage:
        fleet = FleetSimulator(num_vehicles=5, model_dir="models/latest")
        for tick_results in fleet.run(num_ticks=100):
            # tick_results is a list of dicts, one per vehicle
            ...
    """

    def __init__(
        self,
        num_vehicles: int = 5,
        model_dir: str = "latest",
    ) -> None:
        self.vehicles = [
            VehicleSimulator(vehicle_id=f"v-{i+1:03d}")
            for i in range(num_vehicles)
        ]
        self.predictor = FaultPredictor(model_dir)
        self.tick_count = 0
        self.history: list[list[dict]] = []

        logger.info("Fleet initialized: %d vehicles", num_vehicles)

    def run(self, num_ticks: int = 100):
        """Generator that yields results for each tick.

        Yields:
            List of result dicts, one per vehicle, for the current tick.
        """
        for t in range(num_ticks):
            self.tick_count = t + 1
            tick_results = []

            for vehicle in self.vehicles:
                result = self._process_vehicle_tick(vehicle, self.tick_count)
                tick_results.append(result)

            self.history.append(tick_results)
            yield tick_results

    def _process_vehicle_tick(self, vehicle: VehicleSimulator, tick: int) -> dict:
        """Run one tick of the full pipeline for a single vehicle."""
        # 1. Vehicle generates sensor reading from Kaggle CSV
        reading = vehicle.tick()

        # 2. Predict fault
        prediction = self.predictor.predict(vehicle.vehicle_id, reading)

        # 3. Root cause analysis
        cause, recommended_action = resolve_fault(
            prediction["predicted_fault"],
            prediction["explanations"],
            input_data=reading,
        )

        # 4. Feedback: log the action
        if prediction["predicted_fault"] != 0:
            vehicle.apply_effect(recommended_action)

        return {
            "vehicle_id": vehicle.vehicle_id,
            "tick": tick,
            "reading": reading,
            "predicted_fault": prediction["predicted_fault"],
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"],
            "explanations": prediction["explanations"],
            "root_cause": cause,
            "action": recommended_action,
        }

    def get_vehicle_histories(self) -> dict[str, list[dict]]:
        """Return per-vehicle history for dashboard plotting."""
        histories: dict[str, list[dict]] = {v.vehicle_id: [] for v in self.vehicles}
        for tick_results in self.history:
            for result in tick_results:
                histories[result["vehicle_id"]].append(result)
        return histories
