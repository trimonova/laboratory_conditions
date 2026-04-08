import numpy as np
import shapely
import shapely.geometry as geom
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def _compute_level_set_vectorized(coord_matrix, N_r_full, M_fi_full, polygon_1, polygon_2, line_1, line_2):
    """Вычисление level-set для сетки точек — векторная версия Shapely 2.0.

    Вместо поточечного цикла (N_r_full * M_fi_full вызовов Point/contains/distance)
    создаём все точки разом и вызываем contains/distance один раз на весь массив.
    """
    # Извлекаем x, y координаты из coord_matrix в плоские массивы
    x_coords = np.array([[coord_matrix[n][m][0] for m in range(M_fi_full)] for n in range(N_r_full)])
    y_coords = np.array([[coord_matrix[n][m][1] for m in range(M_fi_full)] for n in range(N_r_full)])

    # Создаём все точки разом (один вызов вместо N*M)
    pts = shapely.points(x_coords.ravel(), y_coords.ravel())

    # Проверяем принадлежность полигонам (один вызов вместо N*M)
    mask_1 = shapely.contains(polygon_1, pts)
    mask_2 = shapely.contains(polygon_2, pts)

    # Вычисляем расстояния до линий (два вызова вместо 2*N*M)
    dist_1 = shapely.distance(line_1, pts)
    dist_2 = shapely.distance(line_2, pts)

    # Собираем результат: внутри полигона — положительное, снаружи — отрицательное
    result = np.where(mask_1, dist_1, np.where(mask_2, dist_2, -np.minimum(dist_1, dist_2)))

    return result.reshape(N_r_full, M_fi_full)


def find_func_matrix_remake(coord_matrix, M_fi_full, N_r_full, bound_coords_cart_1, bound_coords_cart_2, coord_matrix_in_cell_center, r_well, delta_r, delta_fi_list):
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

    # Векторное вычисление level-set для обеих сеток
    Func_matrix_remake = _compute_level_set_vectorized(
        coord_matrix, N_r_full, M_fi_full, polygon_1, polygon_2, line_1, line_2)
    Func_matrix_remake_in_cell_center = _compute_level_set_vectorized(
        coord_matrix_in_cell_center, N_r_full, M_fi_full, polygon_1, polygon_2, line_1, line_2)

    # Ghost-точки у скважины и на внешней границе (M_fi_full точек каждая)
    angles = np.array([m * delta_fi_list[m] + delta_fi_list[m] / 2 for m in range(M_fi_full)])

    r_0 = r_well - delta_r / 2
    pts_0 = shapely.points(r_0 * np.cos(angles), r_0 * np.sin(angles))
    dist_0_1 = shapely.distance(line_1, pts_0)
    dist_0_2 = shapely.distance(line_2, pts_0)
    func_list_0 = (-np.minimum(dist_0_1, dist_0_2)).tolist()

    r_end = r_well + (N_r_full - 1) * delta_r + delta_r / 2
    pts_end = shapely.points(r_end * np.cos(angles), r_end * np.sin(angles))
    dist_end_1 = shapely.distance(line_1, pts_end)
    dist_end_2 = shapely.distance(line_2, pts_end)
    func_list_end = (-np.minimum(dist_end_1, dist_end_2)).tolist()

    return Func_matrix_remake, Func_matrix_remake_in_cell_center, func_list_0, func_list_end
