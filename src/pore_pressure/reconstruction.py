"""Level-set reinitialization from boundary coordinates."""

from __future__ import annotations

import numpy as np
import shapely.geometry as geom
from scipy.spatial import ConvexHull


def find_func_matrix_remake(
    coord_matrix: list,
    M_fi_full: int,
    N_r_full: int,
    bound_coords_cart_1: list[tuple[float, float]],
    bound_coords_cart_2: list[tuple[float, float]],
    coord_matrix_in_cell_center: list,
    r_well: float,
    delta_r: float,
    delta_fi_list: list[float],
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    """Reconstruct the level-set function from boundary coordinates.

    Returns (Func_matrix_remake, Func_matrix_remake_in_cell_center, func_list_0, func_list_end).
    """
    Func_matrix_remake = np.zeros((N_r_full, M_fi_full))
    Func_matrix_remake_in_cell_center = np.zeros((N_r_full, M_fi_full))

    hull_1 = ConvexHull(bound_coords_cart_1).vertices
    hull_2 = ConvexHull(bound_coords_cart_2).vertices
    sort_bound_coords_1 = [bound_coords_cart_1[i] for i in hull_1]
    sort_bound_coords_2 = [bound_coords_cart_2[i] for i in hull_2]

    sort_bound_coords_1.append(sort_bound_coords_1[0])
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
                Func_matrix_remake[n][m] = point.distance(line_1)
            elif polygon_2.contains(point):
                Func_matrix_remake[n][m] = point.distance(line_2)
            else:
                point_dist_line_1 = point.distance(line_1)
                point_dist_line_2 = point.distance(line_2)
                Func_matrix_remake[n][m] = -min([point_dist_line_1, point_dist_line_2])

    for m in range(M_fi_full):
        for n in range(N_r_full):
            point = geom.Point(coord_matrix_in_cell_center[n][m])
            if polygon_1.contains(point):
                Func_matrix_remake_in_cell_center[n][m] = point.distance(line_1)
            elif polygon_2.contains(point):
                Func_matrix_remake_in_cell_center[n][m] = point.distance(line_2)
            else:
                point_dist_line_1 = point.distance(line_1)
                point_dist_line_2 = point.distance(line_2)
                Func_matrix_remake_in_cell_center[n][m] = -min([point_dist_line_1, point_dist_line_2])

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

    return Func_matrix_remake, Func_matrix_remake_in_cell_center, func_list_0, func_list_end
