from shapely.geometry import Polygon
import numpy as np
from scipy.spatial import ConvexHull
def find_area(xy_oil_area_coords_sort, rfi_oil_area_coords_sort, coord_matrix_cell):

    xy_oil_area_coords_sort.append(xy_oil_area_coords_sort[0])
    rfi_oil_area_coords_sort.append(rfi_oil_area_coords_sort[0])
    pgon_oil = Polygon(xy_oil_area_coords_sort)  # Assuming the OP's x,y coordinates
    oil_area = pgon_oil.area # площадь многоугольника
    #print('oil_area 2', oil_area)
    r_min = coord_matrix_cell[0][0]
    r_max = coord_matrix_cell[1][0]
    #print('r_min', r_min)
    #print('r_max', r_max)
    delta_fi = abs(coord_matrix_cell[1][1]-coord_matrix_cell[2][1])
    for i in range(len(rfi_oil_area_coords_sort)-1):
        if rfi_oil_area_coords_sort[i][0] == rfi_oil_area_coords_sort[i+1][0]:
            r = rfi_oil_area_coords_sort[i][0]
            delta_alpha = abs(rfi_oil_area_coords_sort[i][1] - rfi_oil_area_coords_sort[i+1][1])
            S_segment = 0.5*r**2*(delta_alpha - np.sin(delta_alpha))
            #print('S_segment=', S_segment)
            if rfi_oil_area_coords_sort[i][0] == r_max:
                oil_area = oil_area + S_segment
                #print('tut1')
            if rfi_oil_area_coords_sort[i][0] == r_min:
                #print('tut2')
                if (oil_area-S_segment) > 0:
                    oil_area = oil_area - S_segment

                else:
                    print('oil_area - S_segment < 0', rfi_oil_area_coords_sort, coord_matrix_cell, S_segment)
    cell_area = 0.5*delta_fi*(r_max**2 - r_min**2)

    return round(oil_area, 19), round(cell_area, 19)


