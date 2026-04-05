from input_parameters import N_r_full, M_fi_full, T_exp_dir, c3_oil, c3_water, CP_dict
from input_parameters import wells_coord, delta_r_list, delta_fi_list, porosity, C_total, perm, \
    delta_t, r_well, frac_angle, frac_angle_2, frac_length_1, frac_length_2, mu_oil, mu_water, \
    coord_matrix_rad, coord_matrix_cart, q, M_1, M_2, N_1, N_2, wells_frac_coords, delta_r, delta_fi, X_matrix, Y_matrix

from QinCenter_attempt_2_Sav.newFuncMatrix_fix_5 import define_func_matrix
import copy
from QinCenter_attempt_2_Sav.find_viscosity import find_viscosity

from QinCenter_attempt_2_Sav.find_bound_coords import find_bound_coords
from QinCenter_attempt_2_Sav.find_func_matrix_remake import find_func_matrix_remake
from QinCenter_attempt_2_Sav.start_to_do_replacement import replace_boundary
#from QinCenter_attempt_2_Sav.find_pore_pressure_QinCenter_new_Savenkov_correct import PorePressure_in_Time
from QinCenter.find_pore_pressure_QinCenter import PorePressure_in_Time
import numpy as np
import matplotlib.pyplot as plt

Pres_distrib_in_time = np.load('../../pore_pressure_before_injection/Pres_distrib_before_fracturing_delta_r_0.0005_delta_fi_0.017.npy')
Pres_distrib = Pres_distrib_in_time[-1]

T_exp = 30


Func_matrix_remake = replace_boundary(M_fi_full, N_r_full, coord_matrix_cart, r_well, r_bound_init=0.005)

surf = plt.contourf(X_matrix, Y_matrix, Func_matrix_remake, cmap=plt.get_cmap('jet'))
plt.title('Func_matrix')
plt.colorbar(surf)
plt.show()

viscosity_matrix = find_viscosity(mu_oil, mu_water, Func_matrix_remake, coord_matrix_rad, delta_r_list,
                                 delta_fi_list, N_r_full, M_fi_full)
print(max(viscosity_matrix.flat), min(viscosity_matrix.flat))

surf = plt.contourf(X_matrix, Y_matrix, viscosity_matrix, cmap=plt.get_cmap('jet'))
plt.title('viscosity_matrix')
plt.colorbar(surf)
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

    Func_matrix, velocity = define_func_matrix(Pres_distrib, Func_matrix_remake, perm, delta_r_list, delta_fi_list,
                                               delta_t, r_well, q, viscosity_matrix,  N_r_full, M_fi_full)
    bound_coords_rad_new, bound_coords_cart_new = find_bound_coords(Func_matrix, coord_matrix_rad, delta_r_list,
                                                                    delta_fi_list, M_fi_full, N_r_full)

    Func_matrix_remake = find_func_matrix_remake(coord_matrix_cart, M_fi_full, N_r_full, bound_coords_cart_new)
    #Func_matrix_remake = Func_matrix
    viscosity_matrix = find_viscosity(mu_oil, mu_water, Func_matrix_remake, coord_matrix_rad, delta_r_list,
                                      delta_fi_list, N_r_full, M_fi_full)

    Pres_distrib_in_Time_2.append(Pres_distrib)
    Func_matrix_in_Time.append(Func_matrix)
    velocity_in_Time.append(velocity)
    bound_coords_rad_in_Time.append(bound_coords_rad_new)
    bound_coords_cart_in_Time.append(bound_coords_cart_new)
    Func_matrix_remake_in_Time.append(Func_matrix_remake)
    viscosity_in_Time.append(viscosity_matrix)

#Func_matrix_remake = find_func_matrix_remake(coord_matrix_cart, M_fi_full, N_r_full, bound_coords_cart_new)
np.save('Pres_distrib_in_Time_2', Pres_distrib_in_Time_2)
np.save('viscosity_in_Time', viscosity_in_Time)
np.save('Func_matrix_in_Time', Func_matrix_in_Time)
np.save('Func_matrix_remake_in_Time', Func_matrix_remake_in_Time)
np.save('velocity_in_Time', velocity_in_Time)
np.save('bound_coords_cart_in_Time', bound_coords_cart_in_Time, allow_pickle=True)
np.save('bound_coords_rad_in_Time', bound_coords_rad_in_Time, allow_pickle=True)




surf = plt.contourf(X_matrix, Y_matrix, Pres_distrib, cmap=plt.get_cmap('jet'))
plt.title('pressure')
plt.colorbar(surf)
plt.show()

surf = plt.contourf(X_matrix, Y_matrix, viscosity_matrix, cmap=plt.get_cmap('jet'))
plt.title('viscosity')
plt.colorbar(surf)
plt.show()

surf = plt.contourf(X_matrix, Y_matrix, Func_matrix_remake, cmap=plt.get_cmap('jet'))
plt.title('Func_matrix')
plt.colorbar(surf)
plt.show()
