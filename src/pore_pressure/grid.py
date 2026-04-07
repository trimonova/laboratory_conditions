from __future__ import annotations

import copy

import numpy as np


def build_coord_matrices(
    N_r_full: int,
    M_fi_full: int,
    delta_r: float,
    delta_fi: float,
    r_well: float,
    cell_center: bool,
) -> tuple[list, list]:
    """Build coordinate matrices in polar and Cartesian coordinates.

    If cell_center=True, coordinates are at cell boundaries (no offset).
    If cell_center=False, coordinates are at cell centers (half-step offset).
    """
    offset_r = 0.0 if cell_center else delta_r / 2
    offset_fi = 0.0 if cell_center else delta_fi / 2

    coord_matrix_rad = []
    coord_matrix_cart = []
    for n in range(N_r_full):
        r = delta_r * n + r_well + offset_r
        coord_line_rad = []
        coord_line_cart = []
        for m in range(M_fi_full):
            fi = delta_fi * m + offset_fi
            coord_line_rad.append((r, fi))
            coord_line_cart.append((r * np.cos(fi), r * np.sin(fi)))
        coord_matrix_rad.append(coord_line_rad)
        coord_matrix_cart.append(coord_line_cart)

    return coord_matrix_rad, coord_matrix_cart


def build_xy_matrices(
    N_r_full: int,
    M_fi_full: int,
    delta_r: float,
    delta_fi: float,
    r_well: float,
    cell_center: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build X, Y coordinate matrices as numpy arrays."""
    offset_r = 0.0 if cell_center else delta_r / 2
    offset_fi = 0.0 if cell_center else delta_fi / 2

    X_matrix = np.zeros((N_r_full, M_fi_full))
    Y_matrix = np.zeros((N_r_full, M_fi_full))
    for n in range(N_r_full):
        r = delta_r * n + r_well + offset_r
        for m in range(M_fi_full):
            fi = delta_fi * m + offset_fi
            X_matrix[n][m] = r * np.cos(fi)
            Y_matrix[n][m] = r * np.sin(fi)

    return X_matrix, Y_matrix


def find_fracture_cell_indices(delta_fi_list: list[float], frac_angle: float, frac_angle_2: float) -> tuple[int, int]:
    """Find angular grid indices closest to fracture angles."""
    angle = 0.0
    M_1 = 0
    M_2 = 0
    for i in range(len(delta_fi_list) - 1):
        angle += delta_fi_list[i]
        if angle <= frac_angle <= (angle + delta_fi_list[i + 1]):
            if abs(frac_angle - angle) < abs(frac_angle - (angle + delta_fi_list[i + 1])):
                M_1 = copy.copy(i)
            else:
                M_1 = copy.copy(i + 1)
        if angle <= frac_angle_2 <= (angle + delta_fi_list[i + 1]):
            if abs(frac_angle - angle) < abs(frac_angle - (angle + delta_fi_list[i + 1])):
                M_2 = copy.copy(i)
            else:
                M_2 = copy.copy(i + 1)

    if M_1 == 0 or M_2 == 0:
        raise ValueError("Could not find M_1 or M_2 fracture angular indices")

    M_1 = M_1 + 1
    M_2 = M_2 + 1

    return M_1, M_2
