import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
import shapely.geometry as geom
from scipy.spatial import ConvexHull

import numpy as np
from matplotlib import pyplot as plt
from math import pi, cos, sin

def find_oval(t_rot, b, a):
    u = 0.008*np.cos(t_rot)       #x-position of the center
    v = 0.008*np.sin(t_rot)
    # u = 0
    # v = 0 #y-position of the center
    # a=0.1       #radius on the x-axis
    # b=0.005     #radius on the y-axis
    # t_rot=pi/4 #rotation angle

    t = np.linspace(0, pi/2, 50) + np.linspace(3*pi/2, 2*pi, 50)
    Ell = np.array([a*np.cos(t) , b*np.sin(t)])
         #u,v removed to keep the same center location
    R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])
         #2-D rotation matrix

    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])

    plt.plot( u+Ell[0,:] , v+Ell[1,:] )     #initial ellipse

    plt.plot( u+Ell_rot[0,:] , v+Ell_rot[1,:],'darkorange' )    #rotated ellipse
    plt.grid(color='lightgray',linestyle='--')
    plt.show()

    return Ell_rot

def replace_boundary(frac_angle_1, frac_angle_2, oval_width, frac_length_1, frac_length_2, M_fi_full, N_r_full, coord_matrix, r_well, delta_r, delta_fi_list, coord_matrix_in_cell_center):
    Func_matrix = np.zeros((N_r_full, M_fi_full))
    Func_matrix_in_cell_center = np.zeros((N_r_full, M_fi_full))
    u_1 = 0.008 * np.cos(frac_angle_1)  # x-position of the center
    v_1 = 0.008 * np.sin(frac_angle_1)
    # u_1 = 0
    # v_1 = 0
    oval_length_1 = frac_length_1 + 0.005
    half_oval_1 = find_oval(frac_angle_1, oval_width, oval_length_1)
    x_y_bound_1 = list(zip(half_oval_1[0]+u_1, half_oval_1[1]+v_1))

    u_2 = 0.008 * np.cos(frac_angle_2)  # x-position of the center
    v_2 = 0.008 * np.sin(frac_angle_2)
    # u_2 = 0
    # v_2 = 0
    oval_length_2 = frac_length_2 + 0.005
    half_oval_2 = find_oval(frac_angle_2, oval_width, oval_length_2)
    x_y_bound_2 = list(zip(half_oval_2[0]+u_2, half_oval_2[1]+v_2))

    plt.plot( half_oval_1[0,:]+u_1, half_oval_1[1,:]+v_1)    #rotated ellipse
    plt.plot( half_oval_2[0,:] + u_2, half_oval_2[1,:] + v_2)    #rotated ellipse

    plt.grid(color='lightgray',linestyle='--')
    plt.show()



    hull_1 = ConvexHull(x_y_bound_1).vertices
    sort_bound_coords_1 = [x_y_bound_1[i] for i in hull_1]
    print(sort_bound_coords_1)

    sort_bound_coords_1.append(sort_bound_coords_1[0])
    print(len(x_y_bound_1), len(sort_bound_coords_1))

    hull_2 = ConvexHull(x_y_bound_2).vertices
    sort_bound_coords_2 = [x_y_bound_2[i] for i in hull_2]
    print(sort_bound_coords_2)

    sort_bound_coords_2.append(sort_bound_coords_2[0])
    print(len(x_y_bound_2), len(sort_bound_coords_2))

    # print(bound_coords)
    for elem in sort_bound_coords_1:
        plt.scatter(elem[0], elem[1])
    for elem in sort_bound_coords_2:
        plt.scatter(elem[0], elem[1])
    plt.Circle((0, 0), 0.008, color='r')
    plt.show()

    line_1 = geom.LineString(sort_bound_coords_1)
    polygon_1 = geom.Polygon(sort_bound_coords_1)

    line_2 = geom.LineString(sort_bound_coords_2)
    polygon_2 = geom.Polygon(sort_bound_coords_2)

    import shapely as _shapely

    # --- Векторное вычисление level-set (Shapely 2.0) ---
    # Вместо двойного цикла for m / for n с поточечными Point/contains/distance
    # создаём все точки разом и вызываем операции один раз на весь массив.

    def _compute_level_set(coord_mat):
        x = np.array([[coord_mat[n][m][0] for m in range(M_fi_full)] for n in range(N_r_full)])
        y = np.array([[coord_mat[n][m][1] for m in range(M_fi_full)] for n in range(N_r_full)])
        pts = _shapely.points(x.ravel(), y.ravel())
        mask_1 = _shapely.contains(polygon_1, pts)
        mask_2 = _shapely.contains(polygon_2, pts)
        dist_1 = _shapely.distance(line_1, pts)
        dist_2 = _shapely.distance(line_2, pts)
        return np.where(mask_1, dist_1, np.where(mask_2, dist_2, -np.minimum(dist_1, dist_2))).reshape(N_r_full, M_fi_full)

    Func_matrix = _compute_level_set(coord_matrix)
    Func_matrix_in_cell_center = _compute_level_set(coord_matrix_in_cell_center)

    # Ghost-точки у скважины и на внешней границе
    angles = np.array([m * delta_fi_list[m] + delta_fi_list[m] / 2 for m in range(M_fi_full)])

    r_0 = r_well - delta_r / 2
    pts_0 = _shapely.points(r_0 * np.cos(angles), r_0 * np.sin(angles))
    dist_0_1 = _shapely.distance(line_1, pts_0)
    dist_0_2 = _shapely.distance(line_2, pts_0)
    func_list_0 = (-np.minimum(dist_0_1, dist_0_2)).tolist()

    r_end = r_well + (N_r_full - 1) * delta_r + delta_r / 2
    pts_end = _shapely.points(r_end * np.cos(angles), r_end * np.sin(angles))
    dist_end_1 = _shapely.distance(line_1, pts_end)
    dist_end_2 = _shapely.distance(line_2, pts_end)
    func_list_end = (-np.minimum(dist_end_1, dist_end_2)).tolist()

    return Func_matrix, Func_matrix_in_cell_center, func_list_0, func_list_end

if __name__ == '__main__':

    delta_r = 0.0001
    delta_r_fine = 0.0001
    R_for_fine = 0.02
    R = 0.215
    r_well = 0.0075
    N_r_fine = round(R_for_fine / delta_r_fine)
    delta_r_list = [delta_r_fine] * N_r_fine + [delta_r] * round((R - r_well - R_for_fine) / delta_r)
    N_r_full = len(delta_r_list)

    delta_fi = np.pi / 180  # угол-шаг в радианах
    delta_fi_fine = np.pi / 180
    fi_for_fine = np.pi / 6
    M_fi_fine = round(fi_for_fine / delta_fi_fine)

    frac_angle = np.pi / 6
    frac_angle_2 = np.pi + np.pi/6

    delta_fi_list_first = [delta_fi] * round((frac_angle - fi_for_fine) / delta_fi) + [delta_fi_fine] * (M_fi_fine * 2) + [
        delta_fi] * round((frac_angle_2 - frac_angle - 2 * fi_for_fine) / delta_fi) + [delta_fi_fine] * (M_fi_fine * 2) + [
                              delta_fi] * (round((2 * np.pi - frac_angle_2 - fi_for_fine) / delta_fi))
    angle_lack = round((2 * np.pi - sum(delta_fi_list_first)) / delta_fi)
    # delta_fi_list = [delta_fi]*round((frac_angle-fi_for_fine)/delta_fi) + [delta_fi_fine]*(M_fi_fine*2) + [delta_fi] * round((frac_angle_2 - frac_angle - 2 * fi_for_fine) / delta_fi) + [delta_fi_fine] * (M_fi_fine*2) + [delta_fi] * (round((2*np.pi - frac_angle_2 - fi_for_fine)/delta_fi)+angle_lack)
    delta_fi_list = [delta_fi] * round(2 * np.pi / delta_fi)
    M_fi_full = len(delta_fi_list)

    coord_matrix_rad = []
    coord_matrix_cart = []
    Y_list = []
    X_list = []
    for n in range(len(delta_r_list)):
        coord_line_rad = []
        coord_line_cart = []
        r = sum(delta_r_list[0:n]) + r_well
        for m in range(len(delta_fi_list)):
            fi = sum(delta_fi_list[0:m])
            coord_line_rad.append((r, fi))
            coord_line_cart.append((r * np.cos(fi), r * np.sin(fi)))
            X_list.append(r * np.cos(fi))
            Y_list.append(r * np.sin(fi))
            # coord_matrix_rad[n][m] = (r, fi)
            # coord_matrix_cart[n][m] = (r*np.cos(fi), r*np.sin(fi))
        coord_matrix_rad.append(coord_line_rad)
        coord_matrix_cart.append(coord_line_cart)
        #print(min(coord_matrix_cart), max(coord_matrix_cart))


    fig = plt.figure()
    func_matrix = replace_boundary(frac_angle, frac_angle_2, 0.003, 0.02, M_fi_full, N_r_full, coord_matrix_cart)

    xi = np.linspace(min(X_list),max(X_list), 500)
    yi = np.linspace(min(Y_list), max(Y_list), 500)
    xig, yig = np.meshgrid(xi, yi)
    Func_i = interpolate.griddata((X_list,Y_list), func_matrix.flat, (xig, yig), method='cubic')
    surf = plt.contourf(xig, yig, Func_i, cmap=cm.jet, antialiased=True, vmin=np.nanmin(Func_i), vmax=np.nanmax(Func_i))
    plt.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


