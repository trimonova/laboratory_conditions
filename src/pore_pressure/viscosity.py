"""Effective viscosity computation from oil-water area fractions."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from pore_pressure.geometry import find_area


def find_viscosity(
    mu_oil: float,
    mu_water: float,
    func_matrix: np.ndarray,
    coord_matrix: list,
    delta_r_list: list[float],
    delta_fi_list: list[float],
    N_r_full: int,
    M_fi_full: int,
) -> np.ndarray:
    """Compute effective viscosity matrix as average of left and right cell contributions.

    Returns the averaged viscosity matrix of shape (N_r_full, M_fi_full).
    """
    viscosity_matrix_1 = _compute_viscosity_half(
        mu_oil, mu_water, func_matrix, coord_matrix, delta_r_list, delta_fi_list, N_r_full, M_fi_full, direction="left"
    )
    viscosity_matrix_right = _compute_viscosity_half(
        mu_oil, mu_water, func_matrix, coord_matrix, delta_r_list, delta_fi_list, N_r_full, M_fi_full, direction="right"
    )
    return (viscosity_matrix_1 + viscosity_matrix_right) / 2


def _compute_viscosity_half(
    mu_oil: float,
    mu_water: float,
    func_matrix: np.ndarray,
    coord_matrix: list,
    delta_r_list: list[float],
    delta_fi_list: list[float],
    N_r_full: int,
    M_fi_full: int,
    direction: str,
) -> np.ndarray:
    """Compute viscosity for one half (left or right angular neighbor)."""
    viscosity_matrix = np.zeros((N_r_full, M_fi_full))

    for m in range(M_fi_full):
        for n in range(N_r_full - 1):
            oil_area_coords = []

            if direction == "left":
                if m != M_fi_full - 1:
                    func_matrix_cell = [
                        func_matrix[n][m],
                        func_matrix[n + 1][m],
                        func_matrix[n + 1][m + 1],
                        func_matrix[n][m + 1],
                        func_matrix[n][m],
                    ]
                    coord_matrix_cell = [
                        coord_matrix[n][m],
                        coord_matrix[n + 1][m],
                        coord_matrix[n + 1][m + 1],
                        coord_matrix[n][m + 1],
                        coord_matrix[n][m],
                    ]
                else:
                    func_matrix_cell = [
                        func_matrix[n][m],
                        func_matrix[n + 1][m],
                        func_matrix[n + 1][0],
                        func_matrix[n][0],
                        func_matrix[n][m],
                    ]
                    coord_matrix_cell = [
                        coord_matrix[n][m],
                        coord_matrix[n + 1][m],
                        (coord_matrix[n + 1][0][0], 2 * np.pi),
                        (coord_matrix[n][0][0], 2 * np.pi),
                        coord_matrix[n][m],
                    ]
                # Sign adjustments for angular interpolation
                sign_i1, sign_i3 = 1, -1
            else:  # right
                if m != 0:
                    func_matrix_cell = [
                        func_matrix[n][m],
                        func_matrix[n + 1][m],
                        func_matrix[n + 1][m - 1],
                        func_matrix[n][m - 1],
                        func_matrix[n][m],
                    ]
                    coord_matrix_cell = [
                        coord_matrix[n][m],
                        coord_matrix[n + 1][m],
                        coord_matrix[n + 1][m - 1],
                        coord_matrix[n][m - 1],
                        coord_matrix[n][m],
                    ]
                else:
                    func_matrix_cell = [
                        func_matrix[n][m],
                        func_matrix[n + 1][m],
                        func_matrix[n + 1][M_fi_full - 1],
                        func_matrix[n][M_fi_full - 1],
                        func_matrix[n][m],
                    ]
                    coord_matrix_cell = [
                        (coord_matrix[n][m][0], 2 * np.pi),
                        (coord_matrix[n + 1][m][0], 2 * np.pi),
                        coord_matrix[n + 1][M_fi_full - 1],
                        coord_matrix[n][M_fi_full - 1],
                        coord_matrix[n][m],
                    ]
                sign_i1, sign_i3 = -1, 1

            # Zero out tiny values
            for i in range(len(func_matrix_cell)):
                if abs(func_matrix_cell[i]) < 1e-10:
                    func_matrix_cell[i] = 0

            func_matrix_cell_signs = [np.sign(i) for i in func_matrix_cell]

            if (
                func_matrix_cell[0] > 0
                and func_matrix_cell[1] > 0
                and func_matrix_cell[2] > 0
                and func_matrix_cell[3] > 0
            ):
                viscosity_matrix[n][m] = mu_oil
            elif (
                func_matrix_cell[0] < 0
                and func_matrix_cell[1] < 0
                and func_matrix_cell[2] < 0
                and func_matrix_cell[3] < 0
            ):
                viscosity_matrix[n][m] = mu_water
            else:
                for i in range(4):
                    if (
                        func_matrix_cell_signs[i] != func_matrix_cell_signs[i + 1]
                        and func_matrix_cell_signs[i] != 0
                        and func_matrix_cell_signs[i + 1] != 0
                    ):
                        proport_coef = abs(func_matrix_cell[i] / func_matrix_cell[i + 1])
                        if i == 0:
                            oil_area_coords.append(
                                (
                                    coord_matrix_cell[i][0] + proport_coef * delta_r_list[n] / (proport_coef + 1),
                                    coord_matrix_cell[i][1],
                                )
                            )
                        if i == 1:
                            oil_area_coords.append(
                                (
                                    coord_matrix_cell[i][0],
                                    coord_matrix_cell[i][1]
                                    + sign_i1 * delta_fi_list[m] * proport_coef / (proport_coef + 1),
                                )
                            )
                        if i == 2:
                            oil_area_coords.append(
                                (
                                    coord_matrix_cell[i][0] - proport_coef * delta_r_list[n] / (proport_coef + 1),
                                    coord_matrix_cell[i][1],
                                )
                            )
                        if i == 3:
                            oil_area_coords.append(
                                (
                                    coord_matrix_cell[i][0],
                                    coord_matrix_cell[i][1]
                                    + sign_i3 * delta_fi_list[m] * proport_coef / (proport_coef + 1),
                                )
                            )

                for i in range(4):
                    if func_matrix_cell_signs[i] == 0:
                        oil_area_coords.append(coord_matrix_cell[i])
                    elif func_matrix_cell_signs[i] > 0:
                        oil_area_coords.append(coord_matrix_cell[i])

                if len(set(oil_area_coords)) > 2:
                    x_oil_area_coords = [i[0] * np.cos(i[1]) for i in oil_area_coords]
                    y_oil_area_coords = [i[0] * np.sin(i[1]) for i in oil_area_coords]
                    xy_oil_area_coords = list(zip(x_oil_area_coords, y_oil_area_coords))
                    hull = ConvexHull(xy_oil_area_coords).vertices
                    xy_oil_area_coords_sort = [xy_oil_area_coords[i] for i in hull]
                    rfi_oil_area_coords_sort = [oil_area_coords[i] for i in hull]
                    oil_area, cell_area = find_area(
                        xy_oil_area_coords_sort, rfi_oil_area_coords_sort, coord_matrix_cell
                    )

                    viscosity_matrix[n][m] = (
                        oil_area / cell_area * mu_oil + (cell_area - oil_area) / cell_area * mu_water
                    )
                else:
                    viscosity_matrix[n][m] = mu_water

        viscosity_matrix[N_r_full - 1][m] = mu_water

    return viscosity_matrix
