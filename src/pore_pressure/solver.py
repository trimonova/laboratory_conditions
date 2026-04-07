"""Sparse matrix pressure solver using implicit backward Euler finite differences."""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, hstack, vstack
from scipy.sparse.linalg import spsolve


def PorePressure_in_Time(
    N_r_full: int,
    M_fi_full: int,
    Pres_distrib: np.ndarray,
    c3_oil: float,
    c3_water: float,
    CP_dict: dict,
    q: float,
    wells_frac_coords: list,
    wells_coords: list,
    delta_r_list: list[float],
    delta_fi_list: list[float],
    viscosity_matrix: np.ndarray,
    fi: float,
    C_total: float,
    perm: float,
    delta_t: float,
    r_well: float,
    M_1: int,
    M_2: int,
    N_1: int,
    N_2: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve pressure equation for one timestep.

    Returns (Pres_end, A, B) where Pres_end is the new pressure field.
    """
    k_fluid = fi * C_total
    c3_fluid = k_fluid / delta_t

    B = np.zeros((N_r_full * M_fi_full, 1))
    A_full = None

    for m in range(M_fi_full):
        A = np.zeros((N_r_full, N_r_full))
        A_sym_right = np.zeros((N_r_full, N_r_full))
        A_sym_left = np.zeros((N_r_full, N_r_full))

        for n in range(N_r_full):
            delta_r_i = delta_r_list[n]
            delta_r_i_minus_0_5 = delta_r_i_plus_0_5 = delta_r_i
            r_i = n * delta_r_i + delta_r_i / 2 + r_well
            r_i_minus_0_5 = r_i - delta_r_i / 2
            r_i_plus_0_5 = r_i + delta_r_i / 2

            if n != N_r_full - 1:
                mu_i_plus_0_5 = (viscosity_matrix[n][m] + viscosity_matrix[n + 1][m]) / 2
            if n != 0:
                mu_i_minus_0_5 = (viscosity_matrix[n][m] + viscosity_matrix[n - 1][m]) / 2

            delta_fi_j = delta_fi_list[m]
            delta_fi_j_minus_0_5 = delta_fi_j_plus_0_5 = delta_fi_j

            if m == M_fi_full - 1:
                mu_j_plus_0_5 = (viscosity_matrix[n][m] + viscosity_matrix[n][0]) / 2
                mu_j_minus_0_5 = (viscosity_matrix[n][m] + viscosity_matrix[n][m - 1]) / 2
            else:
                mu_j_plus_0_5 = (viscosity_matrix[n][m] + viscosity_matrix[n][m + 1]) / 2
                mu_j_minus_0_5 = (viscosity_matrix[n][m] + viscosity_matrix[n][m - 1]) / 2

            A_sym_right[n][n] = perm / r_i / delta_fi_j / mu_j_plus_0_5 / r_i / delta_fi_j_plus_0_5
            A_sym_left[n][n] = perm / r_i / delta_fi_j / mu_j_minus_0_5 / r_i / delta_fi_j_minus_0_5

            if n == 0 and m == M_1 - 1:
                A_sym_right[n][n] = 0
                A[n][n + 1] = perm * r_i_plus_0_5 / r_i / delta_r_i / delta_r_i_plus_0_5 / mu_i_plus_0_5
                A[n][n] = -(A[n][n + 1] + A_sym_right[n][n] + A_sym_left[n][n] + c3_fluid)
            elif n == 0 and m == M_2 - 1:
                A_sym_right[n][n] = 0
                A[n][n + 1] = perm * r_i_plus_0_5 / r_i / delta_r_i / delta_r_i_plus_0_5 / mu_i_plus_0_5
                A[n][n] = -(A[n][n + 1] + A_sym_right[n][n] + A_sym_left[n][n] + c3_fluid)
            elif n == 0 and m == M_1:
                A_sym_left[n][n] = 0
                A[n][n + 1] = perm * r_i_plus_0_5 / r_i / delta_r_i / delta_r_i_plus_0_5 / mu_i_plus_0_5
                A[n][n] = -(A[n][n + 1] + A_sym_right[n][n] + A_sym_left[n][n] + c3_fluid)
            elif n == 0 and m == M_2:
                A_sym_left[n][n] = 0
                A[n][n + 1] = perm * r_i_plus_0_5 / r_i / delta_r_i / delta_r_i_plus_0_5 / mu_i_plus_0_5
                A[n][n] = -(A[n][n + 1] + A_sym_right[n][n] + A_sym_left[n][n] + c3_fluid)
            elif n == N_r_full - 1:
                A[n][n - 1] = perm * r_i_minus_0_5 / r_i / delta_r_i / delta_r_i_minus_0_5 / mu_i_minus_0_5
                A[n][n] = -(A[n][n - 1] + A_sym_right[n][n] + A_sym_left[n][n] + c3_fluid)
            elif m == M_1 and n <= N_1:
                A[n][n - 1] = perm * r_i_minus_0_5 / r_i / delta_r_i / delta_r_i_minus_0_5 / mu_i_minus_0_5
                A[n][n + 1] = perm * r_i_plus_0_5 / r_i / delta_r_i / delta_r_i_plus_0_5 / mu_i_plus_0_5
                A[n][n] = -(A[n][n + 1] + A[n][n - 1] + A_sym_right[n][n] + c3_fluid)
                A_sym_left[n][n] = 0
            elif m == M_2 and n <= N_2:
                A[n][n - 1] = perm * r_i_minus_0_5 / r_i / delta_r_i / delta_r_i_minus_0_5 / mu_i_minus_0_5
                A[n][n + 1] = perm * r_i_plus_0_5 / r_i / delta_r_i / delta_r_i_plus_0_5 / mu_i_plus_0_5
                A[n][n] = -(A[n][n + 1] + A[n][n - 1] + A_sym_right[n][n] + c3_fluid)
                A_sym_left[n][n] = 0
            elif m == M_1 - 1 and n <= N_1:
                A[n][n - 1] = perm * r_i_minus_0_5 / r_i / delta_r_i / delta_r_i_minus_0_5 / mu_i_minus_0_5
                A[n][n + 1] = perm * r_i_plus_0_5 / r_i / delta_r_i / delta_r_i_plus_0_5 / mu_i_plus_0_5
                A[n][n] = -(A[n][n + 1] + A[n][n - 1] + A_sym_left[n][n] + c3_fluid)
                A_sym_right[n][n] = 0
            elif m == M_2 - 1 and n <= N_2:
                A[n][n - 1] = perm * r_i_minus_0_5 / r_i / delta_r_i / delta_r_i_minus_0_5 / mu_i_minus_0_5
                A[n][n + 1] = perm * r_i_plus_0_5 / r_i / delta_r_i / delta_r_i_plus_0_5 / mu_i_plus_0_5
                A[n][n] = -(A[n][n + 1] + A[n][n - 1] + A_sym_left[n][n] + c3_fluid)
                A_sym_right[n][n] = 0
            elif n == 0:
                A[n][n + 1] = perm * r_i_plus_0_5 / r_i / delta_r_i / delta_r_i_plus_0_5 / mu_i_plus_0_5
                A[n][n] = -(A[n][n + 1] + A_sym_right[n][n] + A_sym_left[n][n] + c3_fluid)
            else:
                A[n][n - 1] = perm * r_i_minus_0_5 / r_i / delta_r_i / delta_r_i_minus_0_5 / mu_i_minus_0_5
                A[n][n + 1] = perm * r_i_plus_0_5 / r_i / delta_r_i / delta_r_i_plus_0_5 / mu_i_plus_0_5
                A[n][n] = -(A[n][n + 1] + A[n][n - 1] + A_sym_right[n][n] + A_sym_left[n][n] + c3_fluid)

        A_sym_right_coo = coo_matrix(A_sym_right)
        A_sym_left_coo = coo_matrix(A_sym_left)

        if m == 0:
            A_line_1 = hstack(
                [A, A_sym_right_coo, np.zeros((N_r_full, N_r_full * M_fi_full - 3 * N_r_full)), A_sym_left_coo]
            )
            A_full = coo_matrix(A_line_1)
        elif m == M_fi_full - 1:
            A_line_end = hstack(
                [A_sym_right_coo, np.zeros((N_r_full, N_r_full * M_fi_full - 3 * N_r_full)), A_sym_left_coo, A]
            )
            A_full = vstack([A_full, A_line_end])
        else:
            A_line = hstack(
                [
                    np.zeros((N_r_full, N_r_full * (m - 1))),
                    A_sym_left_coo,
                    A,
                    A_sym_right_coo,
                    np.zeros((N_r_full, N_r_full * M_fi_full - (3 + (m - 1)) * N_r_full)),
                ]
            )
            A_full = vstack([A_full, A_line])

    # Build RHS vector
    j = 0
    delta_fi_j = delta_fi_list[0]
    for m in range(M_fi_full):
        for n in range(N_r_full):
            delta_r_i = delta_r_list[n]
            r_i = n * delta_r_i + delta_r_i / 2 + r_well
            k_fluid = fi * C_total
            c3_fluid = k_fluid / delta_t
            B[j][0] = -c3_fluid * Pres_distrib[n][m]

            if m == M_1 and n <= N_1:
                B[j][0] = B[j][0] - 1 / r_i / delta_fi_j * q
            elif m == M_2 and n <= N_2:
                B[j][0] = B[j][0] - 1 / r_i / delta_fi_j * q
            elif m == M_1 - 1 and n <= N_1:
                B[j][0] = B[j][0] - 1 / r_i / delta_fi_j * q
            elif m == M_2 - 1 and n <= N_2:
                B[j][0] = B[j][0] - 1 / r_i / delta_fi_j * q

            j += 1

    # Apply well boundary conditions
    def sort_func(well_coord_couple):
        return well_coord_couple[1] * N_r_full + well_coord_couple[0]

    wells_frac_coords.sort(key=sort_func)

    wells_coords.sort(key=sort_func)
    wells_coords_reverse = wells_coords[::-1]

    for coord_couple in wells_coords_reverse:
        A_well_column_coo = A_full.getcol(coord_couple[1] * N_r_full + coord_couple[0])
        A_well_column = A_well_column_coo.toarray()
        for cell_number in range(len(A_well_column)):
            if A_well_column[cell_number] != 0:
                B[cell_number][0] = B[cell_number][0] - A_well_column[cell_number] * CP_dict[coord_couple]

    # Solve sparse system
    P_new = spsolve(A_full, B)

    P_new = P_new.reshape(N_r_full * M_fi_full, 1)
    Pres_end = np.zeros((N_r_full, M_fi_full))
    j = 0
    for m in range(M_fi_full):
        for n in range(N_r_full):
            Pres_end[n][m] = P_new[j][0]
            j += 1

    return Pres_end, A, B
