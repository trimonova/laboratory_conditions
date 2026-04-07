"""Read experimental data from HDF5 MAT files and apply Savitzky-Golay filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import savgol_filter


# Sensor coordinates in mm: {sensor_id: [x_mm, y_mm]}
SENSOR_COORDS_MM: dict[int, list[int]] = {
    0: [57, -127],
    1: [70, 0],
    2: [-57, 127],
    3: [0, 127],
    4: [121, -121],
    5: [65, 65],
    6: [0, 70],
    7: [0, -185],
    8: [127, 0],
    9: [0, -70],
    10: [0, 0],
    11: [-57, -127],
    12: [57, 127],
    13: [-121, 121],
}


@dataclass
class ExperimentData:
    """Container for experimental pressure data."""

    pressure: np.ndarray  # (n_samples, n_sensors) raw pressure in Pa
    filtered_pressures: dict[int, np.ndarray]  # {sensor_id: filtered_pressure}
    sensor_coords_rad: dict[int, tuple[float, float]]  # {sensor_id: (r, phi)}
    time_step_exp: float  # experimental time step in seconds (0.01s)


def load_experiment_data(mat_path: str | Path) -> ExperimentData:
    """Load experimental data from a MAT (HDF5) file.

    Args:
        mat_path: Path to the .mat file.

    Returns:
        ExperimentData with raw and filtered pressure arrays.
    """
    mat_path = Path(mat_path)

    with h5py.File(mat_path, "r") as f:
        pressure_raw = np.array(f.get("p")).transpose()

    # Convert sensor coordinates to polar (r in meters, phi in [0, 2pi])
    sensor_coords_rad: dict[int, tuple[float, float]] = {}
    for sensor_id, coords_mm in SENSOR_COORDS_MM.items():
        x = coords_mm[0] / 1000.0
        y = coords_mm[1] / 1000.0
        r = (x**2 + y**2) ** 0.5
        phi = np.arctan2(y, x)
        if phi < 0:
            phi += 2 * np.pi
        sensor_coords_rad[sensor_id] = (r, phi)

    # Apply Savitzky-Golay filter to each sensor
    filtered_pressures: dict[int, np.ndarray] = {}
    for i in range(pressure_raw.shape[1]):
        # Sensor 1 needs wider window due to noise
        window = 5051 if i == 1 else 151
        filtered_pressures[i] = savgol_filter(pressure_raw[:, i], polyorder=3, window_length=window)

    return ExperimentData(
        pressure=pressure_raw,
        filtered_pressures=filtered_pressures,
        sensor_coords_rad=sensor_coords_rad,
        time_step_exp=0.01,
    )
