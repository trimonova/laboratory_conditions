from input_parameters_2 import N_r_full, M_fi_full, T_exp_dir, c3_oil, c3_water, CP_dict, Pres_distrib
from input_parameters_2 import wells_coord, delta_r_list, delta_fi_list, porosity, C_total, perm, \
    delta_t, r_well, frac_angle, frac_angle_2, frac_length_1, frac_length_2, mu_oil, mu_water, \
    coord_matrix_rad, coord_matrix_cart, q, M_1, M_2, N_1, N_2, wells_frac_coords, delta_r, delta_fi, X_matrix, Y_matrix

from QinCenter_attempt_2_Sav.newFuncMatrix_fix_5 import define_func_matrix
import copy
from QinCenter_attempt_2_Sav.find_viscosity import find_viscosity

from QinCenter_attempt_2_Sav.find_bound_coords import find_bound_coords
from QinCenter_attempt_2_Sav.find_func_matrix_remake import find_func_matrix_remake
from QinCenter_attempt_2_Sav.start_to_do_replacement import replace_boundary
from QinCenter_attempt_2_Sav.find_pore_pressure_QinCenter_new_Savenkov_correct import PorePressure_in_Time as PorePressure_in_Time_Sav
from QinCenter.find_pore_pressure_QinCenter import PorePressure_in_Time as PorePressure_in_Time_first
import numpy as np
import matplotlib.pyplot as plt

Pres_distrib_in_time = np.load('../../pore_pressure_before_injection/Pres_distrib_before_fracturing_delta_r_0.0005_delta_fi_0.017.npy')
#Pres_distrib = Pres_distrib_in_time[-1]

T_exp = 30


Func_matrix_remake = replace_boundary(M_fi_full, N_r_full, coord_matrix_cart, r_well, r_bound_init=0.05)

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
    Pres_distrib_first, A_first, B_first, A_full_first = PorePressure_in_Time_first(N_r_full, M_fi_full, Pres_distrib, c3_oil, c3_water, CP_dict, q, wells_frac_coords,
                                              wells_coord, delta_r_list, delta_fi_list, viscosity_matrix, porosity,
                                              C_total, perm, delta_t,
                                              r_well, M_1, M_2, N_1, N_2)

    print(np.min(abs(A_full_first.data)), np.max(abs(A_full_first.data)))
    print(np.min(abs(B_first)), np.max(abs(B_first)))

    Pres_distrib_Sav, A_Sav, B_Sav, A_full_Sav = PorePressure_in_Time_Sav(N_r_full, M_fi_full, Pres_distrib, c3_oil, c3_water, CP_dict, q, wells_frac_coords,
                                              wells_coord, delta_r_list, delta_fi_list, viscosity_matrix, porosity,
                                              C_total, perm, delta_t,
                                              r_well, M_1, M_2, N_1, N_2)
    print(np.min(abs(A_full_Sav.data)), np.max(abs(A_full_Sav.data)))
    print(np.min(abs(B_Sav)), np.max(abs(B_Sav)))

    B_relate = B_first/B_Sav
    A_full_relate = A_full_first.data/A_full_Sav.data

    B_relate = np.zeros_like(B_first)

    for number_elem in range(len(B_first)):
        A_list = []
        B_relate[number_elem] = B_first[number_elem]/B_Sav[number_elem]


        A_row_coo_first = A_full_first.getrow(number_elem)
        A_row_array_first = A_row_coo_first.toarray()

        A_row_coo_Sav = A_full_Sav.getrow(number_elem)
        A_row_array_Sav = A_row_coo_Sav.toarray()

        for number_elem_col in range(len(A_row_array_Sav[0])):
            if A_row_array_Sav[0][number_elem_col] != 0:
                A_list.append(A_row_array_first[0][number_elem_col]/A_row_array_Sav[0][number_elem_col])

        if number_elem == 0:
            A_full_relate = np.array((A_list))
            print(np.shape(A_full_relate), np.shape(np.array((A_list))))
            print(number_elem)
        else:
            print(np.array(A_list))
            print(A_full_relate)
            A_full_relate = np.vstack((A_full_relate, np.array(A_list)[0]))





    surf = plt.contourf(X_matrix, Y_matrix, Pres_distrib_first, cmap=plt.get_cmap('jet'))
    plt.title('pressure')
    plt.colorbar(surf)
    plt.show()

    surf = plt.contourf(X_matrix, Y_matrix, Pres_distrib_Sav, cmap=plt.get_cmap('jet'))
    plt.title('pressure')
    plt.colorbar(surf)
    plt.show()