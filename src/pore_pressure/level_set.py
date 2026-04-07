"""Level-set function update using upwind scheme for advection."""

from __future__ import annotations

import numpy as np

from pore_pressure.velocity import find_velocity_grad_fi


def define_func_matrix(
    pressure_field: np.ndarray,
    func_matrix: np.ndarray,
    perm: float,
    delta_r_list: list[float],
    delta_fi_list: list[float],
    t_step: float,
    r_well: float,
    q: float,
    viscosity_matrix: np.ndarray,
    N_r_full: int,
    M_fi_full: int,
    N_1: int,
    N_2: int,
    M_1: int,
    M_2: int,
    func_list_0: list[float],
    func_list_end: list[float],
    func_matrix_in_cell_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update level-set function using upwind advection scheme.

    Returns (func_matrix_new, v_mult_grad, v_r, v_fi, grad_fi_distrib, velocity_distrib).
    """
    velocity_distrib, grad_fi_distrib, viscosity_distrib = find_velocity_grad_fi(
        pressure_field,
        viscosity_matrix,
        delta_r_list[0],
        delta_fi_list[0],
        perm,
        r_well,
        func_matrix_in_cell_center,
        q,
        N_1,
        N_2,
        M_1,
        M_2,
        func_list_0,
        func_list_end,
    )

    func_matrix_in_cell_center_new = np.zeros((N_r_full, M_fi_full))
    v_mult_grad_func_matrix = np.zeros((N_r_full, M_fi_full))
    v_r = np.zeros((N_r_full, M_fi_full))
    v_fi = np.zeros((N_r_full, M_fi_full))

    for m in range(M_fi_full):
        for n in range(N_r_full):
            v_r[n][m] = (velocity_distrib[0][n][m] + velocity_distrib[1][n][m]) / 2
            v_fi[n][m] = (velocity_distrib[2][n][m] + velocity_distrib[3][n][m]) / 2

            # Upwind scheme
            v_mult_grad_func_matrix[n][m] = (
                max(v_r[n][m], 0) * grad_fi_distrib[0][n][m]
                + min(v_r[n][m], 0) * grad_fi_distrib[1][n][m]
                + max(v_fi[n][m], 0) * grad_fi_distrib[2][n][m]
                + min(v_fi[n][m], 0) * grad_fi_distrib[3][n][m]
            )

            func_matrix_in_cell_center_new[n][m] = (
                func_matrix_in_cell_center[n][m] - t_step * v_mult_grad_func_matrix[n][m]
            )

    return func_matrix_in_cell_center_new, v_mult_grad_func_matrix, v_r, v_fi, grad_fi_distrib, velocity_distrib
