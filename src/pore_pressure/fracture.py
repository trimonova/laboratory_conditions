"""Initial fracture geometry initialization via half-ellipses."""

from __future__ import annotations

from math import cos, sin, pi

import numpy as np
import shapely.geometry as geom
from scipy.spatial import ConvexHull


def _find_oval(t_rot: float, b: float, a: float) -> np.ndarray:
    """Generate a half-oval (ellipse) rotated by t_rot."""
    # Element-wise addition (NOT concatenation) — matches original code behavior
    t = np.linspace(0, pi / 2, 50) + np.linspace(3 * pi / 2, 2 * pi, 50)
    Ell = np.array([a * np.cos(t), b * np.sin(t)])
    R_rot = np.array([[cos(t_rot), -sin(t_rot)], [sin(t_rot), cos(t_rot)]])

    Ell_rot = np.zeros((2, Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:, i] = np.dot(R_rot, Ell[:, i])

    return Ell_rot


def replace_boundary(
    frac_angle_1: float,
    frac_angle_2: float,
    oval_width: float,
    frac_length_1: float,
    frac_length_2: float,
    M_fi_full: int,
    N_r_full: int,
    coord_matrix: list,
    r_well: float,
    delta_r: float,
    delta_fi_list: list[float],
    coord_matrix_in_cell_center: list,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    """Initialize level-set function matrices based on fracture geometry.

    Returns (Func_matrix, Func_matrix_in_cell_center, func_list_0, func_list_end).
    """
    Func_matrix = np.zeros((N_r_full, M_fi_full))
    Func_matrix_in_cell_center = np.zeros((N_r_full, M_fi_full))

    u_1 = 0.008 * np.cos(frac_angle_1)
    v_1 = 0.008 * np.sin(frac_angle_1)
    oval_length_1 = frac_length_1 + 0.005
    half_oval_1 = _find_oval(frac_angle_1, oval_width, oval_length_1)
    x_y_bound_1 = list(zip(half_oval_1[0] + u_1, half_oval_1[1] + v_1))

    u_2 = 0.008 * np.cos(frac_angle_2)
    v_2 = 0.008 * np.sin(frac_angle_2)
    oval_length_2 = frac_length_2 + 0.005
    half_oval_2 = _find_oval(frac_angle_2, oval_width, oval_length_2)
    x_y_bound_2 = list(zip(half_oval_2[0] + u_2, half_oval_2[1] + v_2))

    hull_1 = ConvexHull(x_y_bound_1).vertices
    sort_bound_coords_1 = [x_y_bound_1[i] for i in hull_1]
    sort_bound_coords_1.append(sort_bound_coords_1[0])

    hull_2 = ConvexHull(x_y_bound_2).vertices
    sort_bound_coords_2 = [x_y_bound_2[i] for i in hull_2]
    sort_bound_coords_2.append(sort_bound_coords_2[0])

    line_1 = geom.LineString(sort_bound_coords_1)
    polygon_1 = geom.Polygon(sort_bound_coords_1)

    line_2 = geom.LineString(sort_bound_coords_2)
    polygon_2 = geom.Polygon(sort_bound_coords_2)

    func_list_0 = []
    func_list_end = []

    for m in range(M_fi_full):
        for n in range(N_r_full):
            point = geom.Point(coord_matrix[n][m])
            if polygon_1.contains(point):
                Func_matrix[n][m] = point.distance(line_1)
            elif polygon_2.contains(point):
                Func_matrix[n][m] = point.distance(line_2)
            else:
                point_dist_line_1 = point.distance(line_1)
                point_dist_line_2 = point.distance(line_2)
                Func_matrix[n][m] = -min([point_dist_line_1, point_dist_line_2])

    for m in range(M_fi_full):
        for n in range(N_r_full):
            point = geom.Point(coord_matrix_in_cell_center[n][m])
            if polygon_1.contains(point):
                Func_matrix_in_cell_center[n][m] = point.distance(line_1)
            elif polygon_2.contains(point):
                Func_matrix_in_cell_center[n][m] = point.distance(line_2)
            else:
                point_dist_line_1 = point.distance(line_1)
                point_dist_line_2 = point.distance(line_2)
                Func_matrix_in_cell_center[n][m] = -min([point_dist_line_1, point_dist_line_2])

        point_center = geom.Point(
            (r_well - delta_r / 2) * np.cos(m * delta_fi_list[m] + delta_fi_list[m] / 2),
            (r_well - delta_r / 2) * np.sin(m * delta_fi_list[m] + delta_fi_list[m] / 2),
        )
        point_dist_line_1 = point_center.distance(line_1)
        point_dist_line_2 = point_center.distance(line_2)
        func_list_0.append(-min([point_dist_line_1, point_dist_line_2]))

        point_end = geom.Point(
            (r_well + (N_r_full - 1) * delta_r + delta_r / 2) * np.cos(m * delta_fi_list[m] + delta_fi_list[m] / 2),
            (r_well + (N_r_full - 1) * delta_r + delta_r / 2) * np.sin(m * delta_fi_list[m] + delta_fi_list[m] / 2),
        )
        point_dist_line_1 = point_end.distance(line_1)
        point_dist_line_2 = point_end.distance(line_2)
        func_list_end.append(-min([point_dist_line_1, point_dist_line_2]))

    return Func_matrix, Func_matrix_in_cell_center, func_list_0, func_list_end
