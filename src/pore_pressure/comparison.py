"""Compare simulation results with experimental data."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pore_pressure.config import SimulationConfig
from pore_pressure.experiment_data import ExperimentData


def compare_with_experiment(
    Pres_distrib_in_Time: np.ndarray,
    config: SimulationConfig,
    exp_data: ExperimentData,
    sensor_ids: list[int] | None = None,
) -> None:
    """Plot simulation vs experiment pressure time series at sensor locations.

    Args:
        Pres_distrib_in_Time: Array of shape (n_timesteps, N_r_full, M_fi_full).
        config: Simulation configuration.
        exp_data: Experimental data.
        sensor_ids: Which sensors to plot. Default: [1, 2, 6, 7, 9, 10].
    """
    if sensor_ids is None:
        sensor_ids = [1, 2, 6, 7, 9, 10]

    # Map sensor coordinates to grid indices
    points_coords = {}
    for sensor_id in sensor_ids:
        r, phi = exp_data.sensor_coords_rad[sensor_id]
        n = int(round(r / config.delta_r))
        m = int(round(phi / config.delta_fi))
        points_coords[sensor_id] = (n, m)

    # Time arrays
    n_timesteps = len(Pres_distrib_in_Time)
    t_sim = [j * config.delta_t for j in range(n_timesteps)]
    n_exp_samples = len(list(exp_data.filtered_pressures.values())[0])
    t_exp = [i * exp_data.time_step_exp for i in range(n_exp_samples)]

    # Plot
    n_rows = (len(sensor_ids) + 1) // 2
    fig, axs = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
    axs = axs.flatten()

    for idx, sensor_id in enumerate(sensor_ids):
        n, m = points_coords[sensor_id]
        r_val, phi_val = exp_data.sensor_coords_rad[sensor_id]

        sim_pressure = Pres_distrib_in_Time[:, n, m]

        axs[idx].set_title(f"Sensor {sensor_id}: (r={r_val:.3f}, phi={phi_val:.2f})", y=0.75, loc="left")
        axs[idx].set_xlabel("Time, s")
        axs[idx].set_ylabel("Pressure, MPa")
        axs[idx].plot(t_exp, exp_data.filtered_pressures[sensor_id], label="Experiment")
        axs[idx].plot(t_sim, sim_pressure / 1e6, label="Simulation")
        axs[idx].legend()

    # Hide unused subplots
    for idx in range(len(sensor_ids), len(axs)):
        axs[idx].set_visible(False)

    plt.tight_layout()
    plt.show()
