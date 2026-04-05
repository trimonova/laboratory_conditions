import numpy as np
import shapely.geometry as geom
import matplotlib.pyplot as plt
#from bound_coord_sort import grahamscan, jarvismarch
from scipy.spatial import ConvexHull
#from matplotlib.figures import plot_line_issimple
def find_func_matrix_remake(coord_matrix, M_fi_full, N_r_full, bound_coords, coord_matrix_in_cell_center):
    Func_matrix_remake = np.zeros((N_r_full, M_fi_full))
    Func_matrix_remake_in_cell_center = np.zeros((N_r_full, M_fi_full))
    hull = ConvexHull(bound_coords).vertices
    sort_bound_coords = [bound_coords[i] for i in hull]
    print(sort_bound_coords)

    sort_bound_coords.append(sort_bound_coords[0])
    print(len(bound_coords), len(sort_bound_coords))
    #print(bound_coords)
    print("tadam")
    for elem in sort_bound_coords:
        plt.scatter(elem[0], elem[1])
    plt.Circle((0, 0), 0.008, color='r')
    plt.show()

    line = geom.LineString(sort_bound_coords)
    polygon = geom.Polygon(sort_bound_coords)

    x_sort = [i[0] for i in sort_bound_coords]
    y_sort = [i[1] for i in sort_bound_coords]

    # plt.plot(x_sort, y_sort)
    # plt.title('sort_bound_coords')
    # plt.show()
    #print(sort_bound_coords[50], sort_bound_coords[51], sort_bound_coords[52],)

    x = [i[0] for i in bound_coords]
    y = [i[1] for i in bound_coords]

    # plt.plot(x, y)
    # plt.title('bound_coords')
    # plt.show()

    # for coord_pair in bound_coords:
    #     plt.scatter(coord_pair[0], coord_pair[1])
    # plt.show()


    # plt.plot(line)
    # plt.title('line')
    # plt.show()


    # plt.plot(polygon)
    # plt.show()
    #
    for m in range(M_fi_full):
        for n in range(N_r_full):
            point = geom.Point(coord_matrix[n][m])
            if polygon.contains(point):
                Func_matrix_remake[n][m] = point.distance(line)
            else:
                Func_matrix_remake[n][m] = -point.distance(line)
    for m in range(M_fi_full):
        for n in range(N_r_full):
            point = geom.Point(coord_matrix_in_cell_center[n][m])
            if polygon.contains(point):
                Func_matrix_remake_in_cell_center[n][m] = point.distance(line)
            else:
                Func_matrix_remake_in_cell_center[n][m] = -point.distance(line)

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

    return Func_matrix_remake, Func_matrix_remake_in_cell_center
    return Func_matrix_remake
