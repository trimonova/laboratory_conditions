import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate



def define_func_matrix(pressure_field, func_matrix, perm, delta_r_list, delta_fi_list, t_step, r_well, q,
                       viscosity_matrix, N_r_full, M_fi_full):
    v_mult_grad_func_matrix = np.zeros((N_r_full, M_fi_full))
    grad_p = np.gradient(pressure_field, delta_r_list[0], delta_fi_list[0])
    grad_func_matrix = np.gradient(func_matrix, delta_r_list[0], delta_fi_list[0])
    grad_p_simple = np.gradient(pressure_field)
    grad_func_matrix_simple = np.gradient(func_matrix)
    func_matrix_new = np.zeros((N_r_full, M_fi_full))
    for m in range(M_fi_full):
        for n in range(N_r_full):
            #print(grad_p[0][n][m], grad_func_matrix[0][n][m])
            v_mult_grad_func_matrix[n][m] = -perm/viscosity_matrix[n][m]*(grad_p[0][n][m]*grad_func_matrix[0][n][m] + 1/(sum(delta_r_list[0:n])+r_well)**2 * grad_p[1][n][m] * grad_func_matrix[1][n][m])
            func_matrix_new[n][m] = func_matrix[n][m] - t_step * v_mult_grad_func_matrix[n][m]


    return func_matrix_new, v_mult_grad_func_matrix