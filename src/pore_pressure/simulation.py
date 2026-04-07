"""Main simulation loop for pore pressure during injection."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pore_pressure.config import SimulationConfig
from pore_pressure.fracture import replace_boundary
from pore_pressure.viscosity import find_viscosity
from pore_pressure.solver import PorePressure_in_Time
from pore_pressure.level_set import define_func_matrix
from pore_pressure.boundary import find_bound_coords_cell_center_2_parts
from pore_pressure.reconstruction import find_func_matrix_remake
from pore_pressure.plotting import (
    plot_contour,
    plot_final_results,
    plot_grad_fi_components,
    plot_velocity_components,
    print_timestep_stats,
)


@dataclass
class SimulationResult:
    """Container for simulation output arrays."""

    Pres_distrib_in_Time: list[np.ndarray] = field(default_factory=list)
    Func_matrix_in_Time: list[np.ndarray] = field(default_factory=list)
    velocity_in_Time: list[np.ndarray] = field(default_factory=list)
    bound_coords_rad_in_Time: list = field(default_factory=list)
    bound_coords_cart_in_Time: list = field(default_factory=list)
    Func_matrix_remake_in_Time: list[np.ndarray] = field(default_factory=list)
    viscosity_in_Time: list[np.ndarray] = field(default_factory=list)


def run_simulation(config: SimulationConfig, show_plots: bool = False) -> SimulationResult:
    """Run the full pore pressure simulation.

    Args:
        config: Simulation configuration.
        show_plots: If True, show intermediate plots (default: False for batch runs).

    Returns:
        SimulationResult with time series of all fields.
    """
    c = config  # shorthand

    # Initialize fracture geometry
    # Fracture passes through cell boundaries (not centers)
    Func_matrix_remake, Func_matrix_remake_in_cell_center, func_list_0, func_list_end = replace_boundary(
        c.M_1 * c.delta_fi_list[0],
        c.M_2 * c.delta_fi_list[0],
        c.oval_width,
        c.frac_length_1,
        c.frac_length_2,
        c.M_fi_full,
        c.N_r_full,
        c.coord_matrix_cart_cell,
        c.well_radius,
        c.delta_r,
        c.delta_fi_list,
        c.coord_matrix_cart,
    )

    if show_plots:
        plot_contour(c.X_matrix, c.Y_matrix, Func_matrix_remake, "Func_matrix (initial)")

    # Initial viscosity
    viscosity_matrix = find_viscosity(
        c.oil_viscosity,
        c.water_viscosity,
        Func_matrix_remake,
        c.coord_matrix_rad_cell,
        c.delta_r_list,
        c.delta_fi_list,
        c.N_r_full,
        c.M_fi_full,
    )

    if show_plots:
        plot_contour(c.X_matrix, c.Y_matrix, viscosity_matrix, "Viscosity (initial)")
        plot_contour(c.X_matrix, c.Y_matrix, Func_matrix_remake_in_cell_center, "Func_matrix_in_cell_center (initial)")

    # Working copy of pressure distribution
    Pres_distrib = copy.deepcopy(c.Pres_distrib)
    wells_frac_coords = list(c.wells_frac_coords)
    wells_coord: list = []

    result = SimulationResult()

    # Main time-stepping loop
    for t in range(c.num_steps):
        print(f"Timestep {t + 1}/{c.num_steps}")

        # Solve pressure
        Pres_distrib, A, B = PorePressure_in_Time(
            c.N_r_full,
            c.M_fi_full,
            Pres_distrib,
            c.c3_oil,
            c.c3_water,
            c.CP_dict,
            c.q,
            wells_frac_coords,
            wells_coord,
            c.delta_r_list,
            c.delta_fi_list,
            viscosity_matrix,
            c.porosity,
            c.C_total,
            c.permeability,
            c.delta_t,
            c.well_radius,
            c.M_1,
            c.M_2,
            c.N_1,
            c.N_2,
        )

        if show_plots:
            plot_contour(c.X_matrix, c.Y_matrix, Pres_distrib, f"Pressure (t={t + 1})")

        # Update level-set (advection)
        Func_matrix_in_cell_center, velocity, v_r, v_fi, grad_fi_distrib, velocity_distrib = define_func_matrix(
            Pres_distrib,
            Func_matrix_remake,
            c.permeability,
            c.delta_r_list,
            c.delta_fi_list,
            c.delta_t,
            c.well_radius,
            c.q,
            viscosity_matrix,
            c.N_r_full,
            c.M_fi_full,
            c.N_1,
            c.N_2,
            c.M_1,
            c.M_2,
            func_list_0,
            func_list_end,
            Func_matrix_remake_in_cell_center,
        )

        # Track displacement front
        bound_coords_rad_1, bound_coords_cart_1, bound_coords_rad_2, bound_coords_cart_2 = (
            find_bound_coords_cell_center_2_parts(
                Func_matrix_in_cell_center,
                c.coord_matrix_rad,
                c.delta_r_list,
                c.delta_fi_list,
                c.M_fi_full,
                c.N_r_full,
            )
        )

        # Reconstruct level-set from boundary
        Func_matrix_remake, Func_matrix_remake_in_cell_center, func_list_0, func_list_end = find_func_matrix_remake(
            c.coord_matrix_cart_cell,
            c.M_fi_full,
            c.N_r_full,
            bound_coords_cart_1,
            bound_coords_cart_2,
            c.coord_matrix_cart,
            c.well_radius,
            c.delta_r,
            c.delta_fi_list,
        )

        # Recompute viscosity
        viscosity_matrix = find_viscosity(
            c.oil_viscosity,
            c.water_viscosity,
            Func_matrix_remake,
            c.coord_matrix_rad_cell,
            c.delta_r_list,
            c.delta_fi_list,
            c.N_r_full,
            c.M_fi_full,
        )

        # Print min/max for all fields
        print_timestep_stats(
            t + 1,
            Pres_distrib,
            Func_matrix_in_cell_center,
            Func_matrix_remake_in_cell_center,
            viscosity_matrix,
            velocity,
            v_r,
            v_fi,
            velocity_distrib,
            grad_fi_distrib,
            bound_coords_rad_1,
            bound_coords_cart_1,
            bound_coords_rad_2,
            bound_coords_cart_2,
        )

        if show_plots:
            plot_contour(c.X_matrix_cell, c.Y_matrix_cell, Func_matrix_remake, f"Func_matrix_remake (t={t + 1})")
            plot_contour(c.X_matrix_cell, c.Y_matrix_cell, viscosity_matrix, f"Viscosity (t={t + 1})")
            plot_velocity_components(c.X_matrix_cell, c.Y_matrix_cell, velocity_distrib, t + 1)
            plot_grad_fi_components(c.X_matrix_cell, c.Y_matrix_cell, grad_fi_distrib, t + 1)

        # Store results
        result.Pres_distrib_in_Time.append(Pres_distrib.copy())
        result.Func_matrix_in_Time.append(Func_matrix_in_cell_center.copy())
        result.velocity_in_Time.append(velocity.copy())
        result.bound_coords_rad_in_Time.append(bound_coords_cart_1)
        result.bound_coords_cart_in_Time.append(bound_coords_cart_2)
        result.Func_matrix_remake_in_Time.append(Func_matrix_remake_in_cell_center.copy())
        result.viscosity_in_Time.append(viscosity_matrix.copy())

    # Show final plots
    if show_plots:
        plot_final_results(
            c.X_matrix,
            c.Y_matrix,
            c.X_matrix_cell,
            c.Y_matrix_cell,
            Pres_distrib,
            viscosity_matrix,
            Func_matrix_remake,
        )

    return result


def save_results(result: SimulationResult, output_dir: str | Path) -> None:
    """Save simulation results to .npy files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "Pres_distrib_in_Time_2", result.Pres_distrib_in_Time)
    np.save(output_dir / "viscosity_in_Time", result.viscosity_in_Time)
    np.save(output_dir / "Func_matrix_in_Time", result.Func_matrix_in_Time)
    np.save(output_dir / "Func_matrix_remake_in_Time", result.Func_matrix_remake_in_Time)
    np.save(output_dir / "velocity_in_Time", result.velocity_in_Time)
    np.save(output_dir / "bound_coords_cart_in_Time", result.bound_coords_cart_in_Time, allow_pickle=True)
    np.save(output_dir / "bound_coords_rad_in_Time", result.bound_coords_rad_in_Time, allow_pickle=True)
    print(f"Results saved to {output_dir}")
