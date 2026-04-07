"""Visualization functions for simulation results."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_contour(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, title: str) -> None:
    """Plot a filled contour with colorbar."""
    surf = plt.contourf(X, Y, Z, cmap=plt.get_cmap("jet"))
    plt.title(title)
    plt.colorbar(surf)
    plt.show()


def plot_radial_profiles(data: np.ndarray, columns: list[int], labels: list[str] | None = None) -> None:
    """Plot radial profiles for selected angular indices."""
    colors = ["blue", "orange", "green", "red", "black", "purple"]
    for i, col in enumerate(columns):
        label = labels[i] if labels else f"m={col}"
        plt.plot(data[:, col], color=colors[i % len(colors)], label=label)
    plt.legend()
    plt.show()


def print_field_stats(name: str, data) -> None:
    """Print min/max for a field."""
    arr = np.asarray(data)
    if arr.size == 0:
        print(f"  {name}: (empty)")
        return
    if arr.dtype == object:
        # list of tuples (boundary coords)
        all_vals = []
        for item in arr.flat:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                all_vals.extend(item)
            elif isinstance(item, (int, float, np.floating)):
                all_vals.append(item)
        if all_vals:
            print(f"  {name}: min={min(all_vals):.6e}, max={max(all_vals):.6e}, count={len(all_vals)}")
        else:
            print(f"  {name}: (no numeric data)")
        return
    print(f"  {name}: min={arr.min():.6e}, max={arr.max():.6e}")


def print_timestep_stats(
    t: int,
    Pres_distrib: np.ndarray,
    Func_matrix_in_cell_center: np.ndarray,
    Func_matrix_remake_in_cell_center: np.ndarray,
    viscosity_matrix: np.ndarray,
    velocity: np.ndarray,
    v_r: np.ndarray,
    v_fi: np.ndarray,
    velocity_distrib: np.ndarray,
    grad_fi_distrib: np.ndarray,
    bound_coords_rad_1: list,
    bound_coords_cart_1: list,
    bound_coords_rad_2: list,
    bound_coords_cart_2: list,
) -> None:
    """Print min/max statistics for all fields at a given timestep."""
    print(f"\n--- Timestep {t} field statistics ---")
    print_field_stats("Pres_distrib", Pres_distrib)
    print_field_stats("Func_matrix_in_cell_center", Func_matrix_in_cell_center)
    print_field_stats("Func_matrix_remake_in_cell_center", Func_matrix_remake_in_cell_center)
    print_field_stats("viscosity_matrix", viscosity_matrix)
    print_field_stats("velocity (v*grad)", velocity)
    print_field_stats("v_r", v_r)
    print_field_stats("v_fi", v_fi)
    print_field_stats("velocity_left", velocity_distrib[0])
    print_field_stats("velocity_right", velocity_distrib[1])
    print_field_stats("velocity_bot", velocity_distrib[2])
    print_field_stats("velocity_up", velocity_distrib[3])
    print_field_stats("grad_fi_left", grad_fi_distrib[0])
    print_field_stats("grad_fi_right", grad_fi_distrib[1])
    print_field_stats("grad_fi_bot", grad_fi_distrib[2])
    print_field_stats("grad_fi_up", grad_fi_distrib[3])
    print_field_stats("bound_coords_rad_1", bound_coords_rad_1)
    print_field_stats("bound_coords_cart_1", bound_coords_cart_1)
    print_field_stats("bound_coords_rad_2", bound_coords_rad_2)
    print_field_stats("bound_coords_cart_2", bound_coords_cart_2)
    print()


def plot_velocity_components(
    X_cell: np.ndarray,
    Y_cell: np.ndarray,
    velocity_distrib: np.ndarray,
    t: int,
) -> None:
    """Plot all 4 velocity face components as contour plots."""
    titles = ["velocity_left (v_r, left face)", "velocity_right (v_r, right face)",
              "velocity_bot (v_phi, bot face)", "velocity_up (v_phi, top face)"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    for i, ax in enumerate(axs.flat):
        surf = ax.contourf(X_cell, Y_cell, velocity_distrib[i], cmap="jet")
        ax.set_title(f"{titles[i]} (t={t})")
        fig.colorbar(surf, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_grad_fi_components(
    X_cell: np.ndarray,
    Y_cell: np.ndarray,
    grad_fi_distrib: np.ndarray,
    t: int,
) -> None:
    """Plot all 4 level-set gradient components as contour plots."""
    titles = ["grad_fi_left (d_phi/dr, backward)", "grad_fi_right (d_phi/dr, forward)",
              "grad_fi_bot (d_phi/d_fi, backward)", "grad_fi_up (d_phi/d_fi, forward)"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    for i, ax in enumerate(axs.flat):
        surf = ax.contourf(X_cell, Y_cell, grad_fi_distrib[i], cmap="jet")
        ax.set_title(f"{titles[i]} (t={t})")
        fig.colorbar(surf, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_final_results(
    X_matrix: np.ndarray,
    Y_matrix: np.ndarray,
    X_matrix_cell: np.ndarray,
    Y_matrix_cell: np.ndarray,
    Pres_distrib: np.ndarray,
    viscosity_matrix: np.ndarray,
    Func_matrix_remake: np.ndarray,
) -> None:
    """Plot final pressure, viscosity, and level-set fields."""
    plot_contour(X_matrix, Y_matrix, Pres_distrib, "pressure")
    plot_contour(X_matrix, Y_matrix, viscosity_matrix, "viscosity")
    plot_contour(X_matrix_cell, Y_matrix_cell, Func_matrix_remake, "Func_matrix")
