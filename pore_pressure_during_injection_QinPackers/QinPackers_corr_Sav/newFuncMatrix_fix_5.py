import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from QinPackers_corr_Sav.find_velocity_2 import find_velocity_grad_fi

def define_func_matrix(pressure_field, func_matrix, perm, delta_r_list, delta_fi_list, t_step, r_well, q,
                       viscosity_matrix, N_r_full, M_fi_full, N_1, N_2, M_1, M_2, func_list_0, func_list_end, func_matrix_in_cell_center):
    velocity_distrib, grad_fi_distrib, viscosity_distrib = find_velocity_grad_fi(pressure_field, viscosity_matrix, delta_r_list[0], delta_fi_list[0], perm, r_well, func_matrix_in_cell_center, q, N_1, N_2, M_1, M_2, func_list_0, func_list_end)

    # Скорости в центрах ячеек (среднее граней) — numpy-операции вместо цикла
    v_r = (velocity_distrib[0] + velocity_distrib[1]) / 2
    v_fi = (velocity_distrib[2] + velocity_distrib[3]) / 2

    # Upwind-схема: max(v,0)*grad_backward + min(v,0)*grad_forward — numpy вместо цикла
    v_mult_grad_func_matrix = (
        np.maximum(v_r, 0) * grad_fi_distrib[0] + np.minimum(v_r, 0) * grad_fi_distrib[1]
        + np.maximum(v_fi, 0) * grad_fi_distrib[2] + np.minimum(v_fi, 0) * grad_fi_distrib[3]
    )

    # Обновление level-set
    func_matrix_in_cell_center_new = func_matrix_in_cell_center - t_step * v_mult_grad_func_matrix

    return func_matrix_in_cell_center_new, v_mult_grad_func_matrix, v_r, v_fi, grad_fi_distrib, velocity_distrib