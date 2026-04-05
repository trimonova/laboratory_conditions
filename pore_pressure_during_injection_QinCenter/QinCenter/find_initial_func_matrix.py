import numpy as np
import shapely.geometry as geom
def find_initial_func_matrix(r_well, coord_matrix, M_fi_full, N_r_full):
    r_bound = 0.001
    x_y_bound = []
    Func_matrix = np.zeros((N_r_full, M_fi_full))
    for fi in range(360):
        x_y_bound.append(((r_well+r_bound)*np.cos(fi*np.pi/180), (r_well + r_bound) * np.sin(fi * np.pi / 180)))

    line = geom.LineString(x_y_bound)
    polygon = geom.Polygon(x_y_bound)

    for m in range(M_fi_full):
        for n in range(N_r_full):
            point = geom.Point(coord_matrix[n][m])
            if polygon.contains(point):
                Func_matrix[n][m] = point.distance(line)
            else:
                Func_matrix[n][m] = -point.distance(line)
    return Func_matrix
