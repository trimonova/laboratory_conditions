from input_parameters import N_r_full, M_fi_full, T_exp_dir, c3_oil, c3_water, CP_dict, Pres_distrib
from input_parameters import wells_coord, delta_r_list, delta_fi_list, porosity, C_total, perm, \
    delta_t, r_well, frac_angle, frac_angle_2, frac_length_1, frac_length_2, mu_oil, mu_water, \
    coord_matrix_rad, coord_matrix_cart, q, M_1, M_2, N_1, N_2, wells_frac_coords, delta_r, delta_fi, X_matrix, Y_matrix
from input_parameters import coord_matrix_rad_cell, coord_matrix_cart_cell, X_matrix_cell, Y_matrix_cell
from QinPackers_corr_Sav.newFuncMatrix_fix_5 import define_func_matrix
import copy
from QinPackers_corr_Sav.find_viscosity_2 import find_viscosity
from QinPackers_corr_Sav.find_bound_coords_cell_center_2_parts import find_bound_coords_cell_center_2_parts

from QinPackers_corr_Sav.find_bound_coords import find_bound_coords
#from QinPackers_corr_Sav.find_func_matrix_remake import find_func_matrix_remake
from QinPackers_corr_Sav.find_func_matrix_remake_2_parts import find_func_matrix_remake
from QinPackers_corr_Sav.start_to_do_replacement import replace_boundary
from QinPackers_corr_Sav.find_pore_pressure_QinPackers_new_Savenkov_correct_2_0 import PorePressure_in_Time
#from QinPackers_corr_Sav.find_velocity_2 import fi_distrib_in_cell_center

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, Wedge, Polygon
import matplotlib.lines as mlines
from matplotlib.collections import PatchCollection

#from laboratory_conditions.pore_pressure_during_injection_QinPackers.result_folder.calculation_2.find_pore_pressure_during_injection_QinPackers import \
#    Func_matrix

#Pres_distrib_in_time = np.load('..//../pore_pressure_before_injection/Pres_distrib_before_fracturing_delta_r_0.0005_delta_fi_0.017.npy')
#Pres_distrib = Pres_distrib_in_time[-1]

print(M_1, M_2)
T_exp = 50
R = 0.215

# чтобы трещина проходила через границы ячеек, а не через их центры
Func_matrix_remake, Func_matrix_remake_in_cell_center, func_list_0, func_list_end = replace_boundary((M_1)*delta_fi_list[0], (M_2)*delta_fi_list[0], 0.005, frac_length_1, frac_length_2, M_fi_full, N_r_full, coord_matrix_cart_cell, r_well, delta_r, delta_fi_list, coord_matrix_cart)

plt.plot(func_list_end)
plt.show()
plt.plot(func_list_0)
plt.show()
# чтобы трещина проходила через центры ячеек
# Func_matrix_remake = replace_boundary((M_1)*delta_fi_list[0], (M_2)*delta_fi_list[0], 0.001, frac_length_1, frac_length_2, M_fi_full, N_r_full, coord_matrix_cart)

fig, ax = plt.subplots()
surf = plt.contourf(X_matrix, Y_matrix, Func_matrix_remake, cmap=plt.get_cmap('jet'))
plt.title('Func_matrix')
for n in range(N_1):
    plt.scatter((n*delta_r_list[0]+r_well+delta_r/2)*np.cos((M_1)*delta_fi_list[0]), (n*delta_r_list[0]+r_well+delta_r/2)*np.sin((M_1)*delta_fi_list[0]))
for n in range(N_2):
    plt.scatter((n*delta_r_list[0]+r_well)*np.cos(M_2*delta_fi_list[0]), (n*delta_r_list[0]+r_well)*np.sin(M_2*delta_fi_list[0]))

plt.colorbar(surf)
plt.show()

#Func_matrix_remake_in_cell_center = fi_distrib_in_cell_center(Func_matrix_remake, X_matrix_cell, Y_matrix_cell, X_matrix, Y_matrix)

# for n in range(N_1):
#     plt.scatter((n*delta_r_list[0]+r_well)*np.cos(M_1*delta_fi_list[0]), (n*delta_r_list[0]+r_well)*np.sin(M_1*delta_fi_list[0]))
# for n in range(N_2):
#     plt.scatter((n*delta_r_list[0]+r_well)*np.cos(M_2*delta_fi_list[0]), (n*delta_r_list[0]+r_well)*np.sin(M_2*delta_fi_list[0]))
#
# # plt.title('Func_matrix')
# plt.colorbar(surf)
# # plt.show()

viscosity_matrix = find_viscosity(mu_oil, mu_water, Func_matrix_remake, coord_matrix_rad_cell, delta_r_list,
                                 delta_fi_list, N_r_full, M_fi_full)
print(max(viscosity_matrix.flat), min(viscosity_matrix.flat))

surf_visc = plt.contourf(X_matrix, Y_matrix, viscosity_matrix, cmap=plt.get_cmap('jet'))
plt.title('Viscosity_matrix')
plt.colorbar(surf_visc)

plt.show()

surf_func = plt.contourf(X_matrix, Y_matrix, Func_matrix_remake_in_cell_center, cmap=plt.get_cmap('jet'))
plt.title('func_matrix_in_cell_center')
plt.colorbar(surf_func)
plt.show()


for i in range(len(delta_fi_list)):
    x, y = np.array([[0, (R-r_well)*np.cos(sum(delta_fi_list[0:i]))], [0, (R-r_well)*np.sin(sum(delta_fi_list[0:i]))]])
    line = mlines.Line2D(x, y, lw=0.5)
    ax.add_line(line)

patches = []
for i in range(len(delta_r_list)):
    circle = Wedge((0,0), r_well+sum(delta_r_list[0:i]), 0, 360, width=0.00005)
    patches.append(circle)

p = PatchCollection(patches)
ax.add_collection(p)

plt.title('viscosity_matrix')
plt.colorbar(surf_visc)
plt.show()

plt.plot(viscosity_matrix[:, 45], 'blue')
plt.plot(viscosity_matrix[:, 46], 'yellow')
plt.plot(viscosity_matrix[:, 47], 'green')
plt.plot(viscosity_matrix[:, 44], 'orange')
plt.plot(viscosity_matrix[:, 43], 'black')
plt.show()

plt.plot(Func_matrix_remake[:, 45], 'blue')
plt.plot(Func_matrix_remake[:, 46], 'yellow')
plt.plot(Func_matrix_remake[:, 47], 'green')
plt.plot(Func_matrix_remake[:, 44], 'orange')
plt.plot(Func_matrix_remake[:, 43], 'black')
plt.show()


Pres_distrib_in_Time_2 = []
Func_matrix_in_Time = []
velocity_in_Time = []
bound_coords_rad_in_Time = []
bound_coords_cart_in_Time = []
Func_matrix_remake_in_Time = []
viscosity_in_Time = []

for t in range(T_exp):
    print(t)
    Pres_distrib, A, B = PorePressure_in_Time(N_r_full, M_fi_full, Pres_distrib, c3_oil, c3_water, CP_dict, q, wells_frac_coords,
                                              wells_coord, delta_r_list, delta_fi_list, viscosity_matrix, porosity,
                                              C_total, perm, delta_t,
                                              r_well, M_1, M_2, N_1, N_2)
    surf = plt.contourf(X_matrix, Y_matrix, Pres_distrib, cmap=plt.get_cmap('jet'))
    plt.title('pressure')
    plt.colorbar(surf)
    plt.show()

    plt.plot(Pres_distrib[:, 45])
    plt.plot(Pres_distrib[:, 44])
    plt.plot(Pres_distrib[:, 46])
    plt.plot(Pres_distrib[:, 43])
    #plt.plot(Pres_distrib[:, 42])
    plt.show()


    Func_matrix_in_cell_center, velocity, v_r, v_fi, grad_fi_distrib, velocity_distrib = define_func_matrix(Pres_distrib, Func_matrix_remake, perm, delta_r_list, delta_fi_list,
                                               delta_t, r_well, q, viscosity_matrix,  N_r_full, M_fi_full, N_1, N_2, M_1, M_2, func_list_0, func_list_end, Func_matrix_remake_in_cell_center)



    surf = plt.contourf(X_matrix_cell, Y_matrix_cell, velocity_distrib[0], cmap=plt.get_cmap('jet'))
    plt.title('velocity_left')
    plt.colorbar(surf)
    plt.show()

    surf = plt.contourf(X_matrix_cell, Y_matrix_cell, velocity_distrib[1], cmap=plt.get_cmap('jet'))
    plt.title('velocity_right')
    plt.colorbar(surf)
    plt.show()

    surf = plt.contourf(X_matrix_cell, Y_matrix_cell, velocity_distrib[2], cmap=plt.get_cmap('jet'))
    plt.title('velocity_bot')
    plt.colorbar(surf)
    plt.show()

    surf = plt.contourf(X_matrix_cell, Y_matrix_cell, velocity_distrib[3], cmap=plt.get_cmap('jet'))
    plt.title('velocity_top')
    plt.colorbar(surf)
    plt.show()

    surf = plt.contourf(X_matrix_cell, Y_matrix_cell, v_r, cmap=plt.get_cmap('jet'))
    plt.title('velocity_r')
    plt.colorbar(surf)
    plt.show()

    surf = plt.contourf(X_matrix_cell, Y_matrix_cell, v_fi, cmap=plt.get_cmap('jet'))
    plt.title('velocity_fi')
    plt.colorbar(surf)
    plt.show()

    surf = plt.contourf(X_matrix_cell, Y_matrix_cell, velocity, cmap=plt.get_cmap('jet'))
    plt.title('velocity')
    plt.colorbar(surf)
    plt.show()

    plt.plot(velocity[:, 45])
    plt.plot(velocity[:, 44])
    plt.plot(velocity[:, 46])
    plt.plot(velocity[:, 43])
    plt.plot(velocity[:, 47])
    plt.plot(velocity[:, 42])
    #plt.plot(Pres_distrib[:, 42])
    plt.show()


    #plt.plot(velocity[:, 47])
    #plt.plot(velocity[:, 44])
    # plt.plot(Pres_distrib[:, 42])


    surf = plt.contourf(X_matrix_cell, Y_matrix_cell, grad_fi_distrib[0], cmap=plt.get_cmap('jet'))
    plt.title('grad_fi_distrib_0')
    plt.colorbar(surf)
    plt.show()

    surf = plt.contourf(X_matrix_cell, Y_matrix_cell, grad_fi_distrib[1], cmap=plt.get_cmap('jet'))
    plt.title('grad_fi_distrib_1')
    plt.colorbar(surf)
    plt.show()

    surf = plt.contourf(X_matrix_cell, Y_matrix_cell, grad_fi_distrib[2], cmap=plt.get_cmap('jet'))
    plt.title('grad_fi_distrib_2')
    plt.colorbar(surf)
    plt.show()

    surf = plt.contourf(X_matrix_cell, Y_matrix_cell, grad_fi_distrib[3], cmap=plt.get_cmap('jet'))
    plt.title('grad_fi_distrib_3')
    plt.colorbar(surf)
    plt.show()

    surf = plt.contourf(X_matrix_cell, Y_matrix_cell, Func_matrix_remake_in_cell_center, cmap=plt.get_cmap('jet'))
    plt.title('Func_matrix')
    plt.colorbar(surf)
    plt.show()

    bound_coords_rad_1, bound_coords_cart_1, bound_coords_rad_2, bound_coords_cart_2 = find_bound_coords_cell_center_2_parts(Func_matrix_in_cell_center, coord_matrix_rad, delta_r_list,
                                                                    delta_fi_list, M_fi_full, N_r_full)

    Func_matrix_remake, Func_matrix_remake_in_cell_center, func_list_0, func_list_end = find_func_matrix_remake(coord_matrix_cart_cell, M_fi_full, N_r_full, bound_coords_cart_1, bound_coords_cart_2, coord_matrix_cart, r_well, delta_r, delta_fi_list)

    viscosity_matrix = find_viscosity(mu_oil, mu_water, Func_matrix_remake, coord_matrix_rad_cell, delta_r_list,
                                      delta_fi_list, N_r_full, M_fi_full)

    surf = plt.contourf(X_matrix_cell, Y_matrix_cell, Func_matrix_remake, cmap=plt.get_cmap('jet'))
    plt.title('Func_matrix_remake')
    plt.colorbar(surf)
    plt.show()

    surf = plt.contourf(X_matrix_cell, Y_matrix_cell, viscosity_matrix, cmap=plt.get_cmap('jet'))
    plt.title('viscosity_matrix')
    plt.colorbar(surf)
    plt.show()

    Pres_distrib_in_Time_2.append(Pres_distrib)
    Func_matrix_in_Time.append(Func_matrix_in_cell_center)
    velocity_in_Time.append(velocity)
    bound_coords_rad_in_Time.append(bound_coords_cart_1)
    bound_coords_cart_in_Time.append(bound_coords_cart_2)
    Func_matrix_remake_in_Time.append(Func_matrix_remake_in_cell_center)
    viscosity_in_Time.append(viscosity_matrix)

np.save('result_folder/Pres_distrib_in_Time_2', Pres_distrib_in_Time_2)
np.save('result_folder/viscosity_in_Time', viscosity_in_Time)
np.save('result_folder/Func_matrix_in_Time', Func_matrix_in_Time)
np.save('result_folder/Func_matrix_remake_in_Time', Func_matrix_remake_in_Time)
np.save('result_folder/velocity_in_Time', velocity_in_Time)
np.save('result_folder/bound_coords_cart_in_Time', bound_coords_cart_in_Time, allow_pickle=True)
np.save('result_folder/bound_coords_rad_in_Time', bound_coords_rad_in_Time, allow_pickle=True)




surf = plt.contourf(X_matrix, Y_matrix, Pres_distrib, cmap=plt.get_cmap('jet'))
plt.title('pressure')
plt.colorbar(surf)
plt.show()

surf = plt.contourf(X_matrix, Y_matrix, viscosity_matrix, cmap=plt.get_cmap('jet'))
plt.title('viscosity')
plt.colorbar(surf)
plt.show()

surf = plt.contourf(X_matrix_cell, Y_matrix_cell, Func_matrix_remake, cmap=plt.get_cmap('jet'))
plt.title('Func_matrix')
plt.colorbar(surf)
plt.show()
