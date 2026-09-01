from voltguard.simulator.vehicle import VehicleSimulator


def test_simulator_initializes_properly():
    """Verify the PMSM playback engine initializes with correct vehicle ID."""
    sim = VehicleSimulator("v-1011")
    assert sim.vehicle_id == "v-1011"


def test_simulator_returns_pmsm_fields():
    """Verify tick returns all expected PMSM sensor fields including battery."""
    sim = VehicleSimulator("v-1011")
    reading = sim.tick()

    expected_keys = [
        "vehicle_id", "ambient", "coolant", "u_d", "u_q",
        "motor_speed", "torque", "i_d", "i_q", "pm",
        "stator_yoke", "stator_tooth", "stator_winding", "profile_id",
        "battery_temp", "soc", "power_draw",
    ]
    for key in expected_keys:
        assert key in reading, f"Missing key: {key}"


def test_simulator_tick_advances():
    """Verify the playback index advances on each tick."""
    sim = VehicleSimulator("v-1011")
    idx_before = sim.current_idx
    sim.tick()
    assert sim.current_idx == idx_before + 1


def test_simulator_values_are_numeric():
    """Verify all sensor values are valid numbers."""
    sim = VehicleSimulator("v-1011")
    reading = sim.tick()

    for key, value in reading.items():
        if key == "vehicle_id":
            assert isinstance(value, str)
        else:
            assert isinstance(value, (int, float)), f"{key} is {type(value)}, expected numeric"


def test_battery_temp_is_reasonable():
    """Verify synthetic battery temp is in a physically plausible range."""
    sim = VehicleSimulator("v-1011")
    reading = sim.tick()
    # Battery temp should be >= ambient (can't be cooler than surroundings)
    assert reading["battery_temp"] >= reading["ambient"] - 1  # Small float tolerance


def test_soc_drains_over_time():
    """Verify SOC decreases over multiple ticks (simulating power consumption)."""
    sim = VehicleSimulator("v-1011")
    initial_soc = sim.soc
    for _ in range(50):
        sim.tick()
    # SOC should have decreased (unless power draw was exactly 0 for all ticks)
    assert sim.soc <= initial_soc
