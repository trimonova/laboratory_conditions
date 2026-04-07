"""Oil area computation in grid cells using Shapely polygons with circular segment correction."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def find_area(
    xy_oil_area_coords_sort: list[tuple[float, float]],
    rfi_oil_area_coords_sort: list[tuple[float, float]],
    coord_matrix_cell: list[tuple[float, float]],
) -> tuple[float, float]:
    """Compute oil area and cell area with circular segment correction.

    Returns (oil_area, cell_area) both rounded to 19 decimal places.
    """
    xy_oil_area_coords_sort.append(xy_oil_area_coords_sort[0])
    rfi_oil_area_coords_sort.append(rfi_oil_area_coords_sort[0])
    pgon_oil = Polygon(xy_oil_area_coords_sort)
    oil_area = pgon_oil.area

    r_min = coord_matrix_cell[0][0]
    r_max = coord_matrix_cell[1][0]
    delta_fi = abs(coord_matrix_cell[1][1] - coord_matrix_cell[2][1])

    for i in range(len(rfi_oil_area_coords_sort) - 1):
        if rfi_oil_area_coords_sort[i][0] == rfi_oil_area_coords_sort[i + 1][0]:
            r = rfi_oil_area_coords_sort[i][0]
            delta_alpha = abs(rfi_oil_area_coords_sort[i][1] - rfi_oil_area_coords_sort[i + 1][1])
            S_segment = 0.5 * r**2 * (delta_alpha - np.sin(delta_alpha))
            if rfi_oil_area_coords_sort[i][0] == r_max:
                oil_area = oil_area + S_segment
            if rfi_oil_area_coords_sort[i][0] == r_min:
                if (oil_area - S_segment) > 0:
                    oil_area = oil_area - S_segment

    cell_area = 0.5 * delta_fi * (r_max**2 - r_min**2)

    return round(oil_area, 19), round(cell_area, 19)
