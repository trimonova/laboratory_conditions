import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from QinCenter_attempt_2_Sav.find_velocity import find_velocity_grad_fi

def define_func_matrix(pressure_field, func_matrix, perm, delta_r_list, delta_fi_list, t_step, r_well, q,
                       viscosity_matrix, N_r_full, M_fi_full):
    velocity_distrib, grad_fi_distrib, viscosity_distrib = find_velocity_grad_fi(pressure_field, viscosity_matrix, delta_r_list[0], delta_fi_list[0], perm, r_well, func_matrix, q)

    func_matrix_new = np.zeros((N_r_full, M_fi_full))
    v_mult_grad_func_matrix = np.zeros((N_r_full, M_fi_full))
    for m in range(M_fi_full):
        for n in range(N_r_full):
            v_r = (velocity_distrib[0][n][m] + velocity_distrib[1][n][m])/2
            #v_r = velocity_distrib[1][n][m]
            v_fi = (velocity_distrib[2][n][m] + velocity_distrib[3][n][m])/2
            #v_r = velocity_distrib[0][n][m]
            #v_fi = velocity_distrib[2][n][m]

            v_mult_grad_func_matrix[n][m] = max(v_r, 0)*grad_fi_distrib[0][n][m] + min(v_r, 0)*grad_fi_distrib[1][n][m] + \
                                            max(v_fi, 0)*grad_fi_distrib[2][n][m] + min(v_fi, 0)*grad_fi_distrib[3][n][m]
            #v_mult_grad_func_matrix[n][m] = -v_r
            #v_mult_grad_func_matrix[n][m] = max(v_r, 0) * grad_fi_distrib[0][n][m] + min(v_r, 0) * \
                                            #grad_fi_distrib[1][n][m]
            func_matrix_new[n][m] = func_matrix[n][m] - t_step * v_mult_grad_func_matrix[n][m]


    X = np.zeros((N_r_full, M_fi_full))
    Y = np.zeros((N_r_full, M_fi_full))
    for m in range(M_fi_full):
        for n in range(N_r_full):
            X[n][m] = (r_well + sum(delta_r_list[0:n + 1])) * np.cos(sum(delta_fi_list[0:m]))
            Y[n][m] = (r_well + sum(delta_r_list[0:n + 1])) * np.sin(sum(delta_fi_list[0:m]))

    X_list = [i for i in X.flat]
    Y_list = [i for i in Y.flat]
    Func_matrix_list = [l for l in func_matrix_new.flat]

    xi = np.linspace(min(X_list), max(X_list), 700)
    yi = np.linspace(min(Y_list), max(Y_list), 700)
    xig, yig = np.meshgrid(xi, yi)
    bound_i = interpolate.griddata((X_list, Y_list), Func_matrix_list, (xig, yig), method='cubic')

    # fig, axs = plt.subplots(1, 2, figsize=(11, 10))


    # surf_1 = plt.contourf(xig, yig, bound_i, cmap=plt.get_cmap('jet'))
    # plt.colorbar(surf_1)
    #
    # plt.show()

    return func_matrix_new, v_mult_grad_func_matrix