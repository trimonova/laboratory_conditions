import numpy as np
import shapely.geometry as geom
import matplotlib.pyplot as plt
#from bound_coord_sort import grahamscan, jarvismarch
from scipy.spatial import ConvexHull
#from matplotlib.figures import plot_line_issimple
def find_func_matrix_remake(coord_matrix, M_fi_full, N_r_full, bound_coords):
    Func_matrix_remake = np.zeros((N_r_full, M_fi_full))
    hull = ConvexHull(bound_coords).vertices
    sort_bound_coords = [bound_coords[i] for i in hull]
    print(sort_bound_coords)

    sort_bound_coords.append(sort_bound_coords[0])
    print(len(bound_coords), len(sort_bound_coords))
    #print(bound_coords)

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
    return Func_matrix_remake
