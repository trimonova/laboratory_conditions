from shapely.geometry import Polygon
import numpy as np
from scipy.spatial import ConvexHull
from QinPackers.find_area import find_area

# x = [0, 0, 20, 10]
# y = [0, 10, 10, 0]
# pgon = Polygon(zip(x, y)) # Assuming the OP's x,y coordinates
# print(pgon.area)
# print(np.sign(0), np.sign(-5), np.sign(6))

def find_viscosity(mu_oil, mu_water, func_matrix, coord_matrix, delta_r_list, delta_fi_list, N_r_full, M_fi_full):
    viscosity_matrix = np.zeros((N_r_full, M_fi_full))
    for m in range(M_fi_full):
        for n in range(N_r_full-1):
            oil_area_coords = []  # координаты области с маслом в одной ячейке

            if m != M_fi_full-1:
                func_matrix_cell = [func_matrix[n][m], func_matrix[n + 1][m], func_matrix[n + 1][m + 1],
                                    func_matrix[n][m + 1], func_matrix[n][m]]
                coord_matrix_cell = [coord_matrix[n][m], coord_matrix[n + 1][m], coord_matrix[n + 1][m + 1],
                                     coord_matrix[n][m + 1], coord_matrix[n][m]]
            else:
                func_matrix_cell = [func_matrix[n][m], func_matrix[n + 1][m], func_matrix[n + 1][0],
                                    func_matrix[n][0], func_matrix[n][m]]
                coord_matrix_cell = [coord_matrix[n][m], coord_matrix[n + 1][m], (coord_matrix[n + 1][0][0], 2*np.pi),
                                     (coord_matrix[n][0][0], 2*np.pi) , coord_matrix[n][m]]

            for i in range(len(func_matrix_cell)):
                if abs(func_matrix_cell[i]) < 10**(-10):
                    func_matrix_cell[i] = 0

            func_matrix_cell_signs = [np.sign(i) for i in func_matrix_cell]

            if func_matrix_cell[0] > 0  and func_matrix_cell[1] > 0 and func_matrix_cell[2] > 0 and func_matrix_cell[3] > 0:
                viscosity_matrix[n][m] = mu_oil
            elif func_matrix_cell[0] < 0  and func_matrix_cell[1] < 0 and func_matrix_cell[2] < 0 and func_matrix_cell[3] < 0:
                viscosity_matrix[n][m] = mu_water
            else:
                rad = sum(delta_r_list[0:n+1])
                delta = rad/np.cos(delta_fi_list[m]) - rad
                min_coef_proport = delta/(delta_r_list[n] - delta)
                #print('min_coef_proport', min_coef_proport)
                for i in range(4):
                    if func_matrix_cell_signs[i] != func_matrix_cell_signs[i+1] and func_matrix_cell_signs[i] != 0 and func_matrix_cell_signs[i+1] != 0:
                        # значит граница вытеснения пересекает эту сторону. Найдем ее координаты
                        proport_coef = abs(func_matrix_cell[i] / func_matrix_cell[i + 1])
                        #print('proport_coef', proport_coef)
                        if i == 0:
                            #delta_x = coord_matrix_cell[i+1][0]
                            oil_area_coords.append((coord_matrix_cell[i][0]+proport_coef*delta_r_list[n]/(proport_coef+1), coord_matrix_cell[i][1]))
                            #print(i, 'delta r', delta_r_list[n], oil_area_coords[-1])
                        if i == 1:
                            oil_area_coords.append((coord_matrix_cell[i][0], coord_matrix_cell[i][1]+delta_fi_list[m]*proport_coef/(proport_coef+1)))
                            #print(i, 'delta fi', delta_fi_list[m], oil_area_coords[-1])
                        if i == 2:
                            oil_area_coords.append((coord_matrix_cell[i][0]-proport_coef*delta_r_list[n]/(proport_coef+1), coord_matrix_cell[i][1]))
                            #print(i, 'delta r', delta_r_list[n], oil_area_coords[-1])
                        if i == 3:
                            oil_area_coords.append(
                                (coord_matrix_cell[i][0], coord_matrix_cell[i][1] - delta_fi_list[m]*proport_coef / (proport_coef + 1)))
                            #print(i, 'delta fi', delta_fi_list[m], oil_area_coords[-1])
                for i in range(4):
                    if func_matrix_cell_signs[i] == 0:
                        oil_area_coords.append((coord_matrix_cell[i]))
                    elif func_matrix_cell_signs[i] > 0:
                        oil_area_coords.append((coord_matrix_cell[i]))


                if len(set(oil_area_coords)) > 2:
                    x_oil_area_coords = [i[0] * np.cos(i[1]) for i in oil_area_coords]
                    y_oil_area_coords = [i[0] * np.sin(i[1]) for i in oil_area_coords]
                    xy_oil_area_coords = [(x_oil_area_coords[i], y_oil_area_coords[i]) for i in
                                          range(len(x_oil_area_coords))]
                    #print(func_matrix_cell_signs)
                    #print(func_matrix_cell)
                    #print(coord_matrix_cell)
                    #print(oil_area_coords)
                    hull = ConvexHull(xy_oil_area_coords).vertices
                    xy_oil_area_coords_sort = [xy_oil_area_coords[i] for i in hull]
                    rfi_oil_area_coords_sort = [oil_area_coords[i] for i in hull]
                    oil_area, cell_area = find_area(xy_oil_area_coords_sort, rfi_oil_area_coords_sort, coord_matrix_cell)

                    x_cell_area_coords = [i[0] * np.cos(i[1]) for i in coord_matrix_cell]
                    y_cell_area_coords = [i[0] * np.sin(i[1]) for i in coord_matrix_cell]
                    # pgon_oil = Polygon(xy_oil_area_coords_sort)  # Assuming the OP's x,y coordinates
                    # oil_area = pgon_oil.area
                    # pgon_cell = Polygon(zip(x_cell_area_coords, y_cell_area_coords))  # Assuming the OP's x,y coordinates
                    # cell_area = pgon_cell.area
                    viscosity_matrix[n][m] = oil_area/cell_area * mu_oil + (cell_area-oil_area)/cell_area * mu_water
                    #print(n, m, 'oil area-', oil_area, 'cell area-', cell_area)

                    if viscosity_matrix[n][m] > 0.2 or viscosity_matrix[n][m] < 0.002:
                        print(coord_matrix_cell, cell_area)
                        print(oil_area_coords, oil_area)
                        print(func_matrix_cell)
                        print(viscosity_matrix[n][m], n, m, func_matrix_cell)
                        raise ValueError
                else:
                    viscosity_matrix[n][m] = mu_water
            # if n == 50 and m == 4:
            #     print('tadam', oil_area, cell_area, oil_area_coords, coord_matrix_cell, viscosity_matrix[n][m])
            #     print('oil', x_oil_area_coords, y_oil_area_coords)
            #     print('cell', x_cell_area_coords, y_cell_area_coords)
    for m in range(M_fi_full):
        viscosity_matrix[N_r_full-1][m] = mu_water

    return viscosity_matrix




