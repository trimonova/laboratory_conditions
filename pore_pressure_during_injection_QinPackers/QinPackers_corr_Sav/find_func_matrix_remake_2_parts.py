import numpy as np
import shapely.geometry as geom
import matplotlib.pyplot as plt
#from bound_coord_sort import grahamscan, jarvismarch
from scipy.spatial import ConvexHull
#from matplotlib.figures import plot_line_issimple
def find_func_matrix_remake(coord_matrix, M_fi_full, N_r_full, bound_coords_cart_1, bound_coords_cart_2, coord_matrix_in_cell_center, r_well, delta_r, delta_fi_list):
    Func_matrix_remake = np.zeros((N_r_full, M_fi_full))
    Func_matrix_remake_in_cell_center = np.zeros((N_r_full, M_fi_full))
    hull_1 = ConvexHull(bound_coords_cart_1).vertices
    hull_2 = ConvexHull(bound_coords_cart_2).vertices
    sort_bound_coords_1 = [bound_coords_cart_1[i] for i in hull_1]
    sort_bound_coords_2 = [bound_coords_cart_2[i] for i in hull_2]

    #print(sort_bound_coords)

    sort_bound_coords_1.append(sort_bound_coords_1[0])
    sort_bound_coords_2.append(sort_bound_coords_2[0])
    #print(len(bound_coords), len(sort_bound_coords))
    #print(bound_coords)
    print("tadam")


    for elem in sort_bound_coords_1:
        plt.scatter(elem[0], elem[1], color='red')
    for elem in sort_bound_coords_2:
        plt.scatter(elem[0], elem[1], color='blue')
    plt.Circle((0, 0), 0.008, color='black')
    plt.show()

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

        point_center = geom.Point((r_well - delta_r / 2) * np.cos(m * delta_fi_list[m] + delta_fi_list[m] / 2),
                                  (r_well - delta_r / 2) * np.sin(m * delta_fi_list[m] + delta_fi_list[m] / 2))
        # point_center = geom.Point((r_well - delta_r)*np.cos(m*delta_fi_list[m]), (r_well - delta_r)*np.sin(m*delta_fi_list[m]))
        point_dist_line_1 = point_center.distance(line_1)
        point_dist_line_2 = point_center.distance(line_2)
        func_list_0.append(-min([point_dist_line_1, point_dist_line_2]))

        point_end = geom.Point(
            (r_well + (N_r_full - 1) * delta_r + delta_r / 2) * np.cos(m * delta_fi_list[m] + delta_fi_list[m] / 2),
            (r_well + (N_r_full - 1) * delta_r + delta_r / 2) * np.sin(m * delta_fi_list[m] + delta_fi_list[m] / 2))
        # point_end = geom.Point((r_well + n * delta_r + delta_r) * np.cos(m * delta_fi_list[m]),
        #                       (r_well + n * delta_r + delta_r) * np.sin(m * delta_fi_list[m]))
        point_dist_line_1 = point_end.distance(line_1)
        point_dist_line_2 = point_end.distance(line_2)
        func_list_end.append(-min([point_dist_line_1, point_dist_line_2]))

    return Func_matrix_remake, Func_matrix_remake_in_cell_center, func_list_0, func_list_end
    return Func_matrix_remake
