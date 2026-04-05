import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
import shapely.geometry as geom
from scipy.spatial import ConvexHull

import numpy as np
from matplotlib import pyplot as plt
from math import pi, cos, sin

def replace_boundary(M_fi_full, N_r_full, coord_matrix, r_well, r_bound_init):

    bound_coords = []
    for fi in range(360):
        bound_coords.append(
            ((r_well + r_bound_init) * np.cos(fi * np.pi / 180), (r_well + r_bound_init) * np.sin(fi * np.pi / 180)))

    Func_matrix = np.zeros((N_r_full, M_fi_full))

    line_1 = geom.LineString(bound_coords)
    polygon_1 = geom.Polygon(bound_coords)

    for m in range(M_fi_full):
        for n in range(N_r_full):
            point = geom.Point(coord_matrix[n][m])
            if polygon_1.contains(point):
                Func_matrix[n][m] = point.distance(line_1)
            else:
                Func_matrix[n][m] = -point.distance(line_1)


    return Func_matrix


