import numpy as np
from scipy.interpolate import interp1d


def find_velocity_grad_fi(Pres_distrib, viscosity_distrib, delta_r, delta_fi, perm, r_well, fi_distrib, q_well, N_1, N_2, M_1, M_2, func_list_0, func_list_end):
    N_r_full = np.shape(Pres_distrib)[0]
    M_fi_full = np.shape(Pres_distrib)[1]

    velocity_distrib_left = np.zeros_like(Pres_distrib)
    velocity_distrib_right = np.zeros_like(Pres_distrib)
    velocity_distrib_bot = np.zeros_like(Pres_distrib)
    velocity_distrib_up = np.zeros_like(Pres_distrib)

    grad_fi_distrib_left = np.zeros_like(Pres_distrib)
    grad_fi_distrib_right = np.zeros_like(Pres_distrib)
    grad_fi_distrib_bot = np.zeros_like(Pres_distrib)
    grad_fi_distrib_up = np.zeros_like(Pres_distrib)

    viscosity_distrib_left = np.zeros_like(Pres_distrib)
    viscosity_distrib_right = np.zeros_like(Pres_distrib)
    viscosity_distrib_bot = np.zeros_like(Pres_distrib)
    viscosity_distrib_up = np.zeros_like(Pres_distrib)

    r_line = []
    for n in range(np.shape(Pres_distrib)[0]):
        r_line.append(r_well + delta_r / 2 + delta_r * n)

    # for m in range(np.shape(Pres_distrib)[1]):
    #     fi_line = list(fi_distrib[:, m])
    #     f_interp = interp1d(r_line, fi_line)
        # f_0 = f_interp(r_well - delta_r/2)
        # f_end = f_interp(r_well + delta_r * len(fi_line) + delta_r/2)
        # fi_line_end = list(f_0).append(fi_line)
        # fi_line_end.append(f_end)



    for m in range(M_fi_full):
        for n in range(0, N_r_full):

            fi_line = list(fi_distrib[:, m])
            f_interp = interp1d(r_line, fi_line, fill_value='extrapolate')
            f_0 = f_interp(r_well - delta_r / 2)
            f_end = f_interp(r_well + delta_r * len(fi_line) + delta_r / 2)

            r = r_well + n*delta_r + delta_r/2
            if n == 0:
                visc_mid_right = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n + 1][m]) ** (-1)
                visc_mid_right = (viscosity_distrib[n][m] + viscosity_distrib[n + 1][m])/2
                viscosity_distrib_right[n][m] = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n + 1][m]) ** (-1)
                velocity_distrib_left[n][m] = 0
                velocity_distrib_right[n][m] = -perm / visc_mid_right * (
                        Pres_distrib[n + 1][m] - Pres_distrib[n][m]) / delta_r
                grad_fi_distrib_left[n][m] = (fi_distrib[n][m] - func_list_0[m])/delta_r

                grad_fi_distrib_right[n][m] = (fi_distrib[n+1][m] - fi_distrib[n][m])/delta_r

            elif n == np.shape(Pres_distrib)[0]-1:
                #visc_mid_left = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n - 1][m]) ** (-1)
                visc_mid_left = (viscosity_distrib[n][m] + viscosity_distrib[n - 1][m]) / 2
                viscosity_distrib_left[n][m] = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n - 1][m]) ** (-1)
                velocity_distrib_right[n][m] = 0
                velocity_distrib_left[n][m] = -perm / visc_mid_left * (
                        Pres_distrib[n][m] - Pres_distrib[n - 1][m]) / delta_r
                grad_fi_distrib_left[n][m] = (fi_distrib[n][m] - fi_distrib[n-1][m]) / delta_r

                grad_fi_distrib_right[n][m] = (func_list_end[m] - fi_distrib[n][m]) / delta_r

            else:
                #visc_mid_left = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n - 1][m]) ** (-1)
                #visc_mid_right = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n + 1][m]) ** (-1)
                visc_mid_right = (viscosity_distrib[n][m] + viscosity_distrib[n + 1][m]) / 2
                visc_mid_left = (viscosity_distrib[n][m] + viscosity_distrib[n - 1][m]) / 2
                viscosity_distrib_right[n][m] = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n + 1][m]) ** (-1)
                viscosity_distrib_left[n][m] = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n - 1][m]) ** (-1)
                velocity_distrib_left[n][m] = -perm / visc_mid_left * (
                            Pres_distrib[n][m] - Pres_distrib[n - 1][m]) / delta_r
                velocity_distrib_right[n][m] = -perm / visc_mid_right * (
                            Pres_distrib[n + 1][m] - Pres_distrib[n][m]) / delta_r

                grad_fi_distrib_left[n][m] = (fi_distrib[n][m] - fi_distrib[n - 1][m]) / delta_r
                grad_fi_distrib_right[n][m] = (fi_distrib[n + 1][m] - fi_distrib[n][m]) / delta_r

            if m == np.shape(Pres_distrib)[1]-1:
                #visc_mid_up = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n][0]) ** (-1)
                #visc_mid_bot = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n][m - 1]) ** (-1)
                visc_mid_up = (viscosity_distrib[n][m] + viscosity_distrib[n][0])/2
                visc_mid_bot = (viscosity_distrib[n][m] + viscosity_distrib[n][m - 1])/2
                viscosity_distrib_up[n][m] = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n][0]) ** (-1)
                viscosity_distrib_bot[n][m] = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n][m - 1]) ** (-1)

                velocity_distrib_up[n][m] = -perm / visc_mid_up * (
                            Pres_distrib[n][0] - Pres_distrib[n][m]) / delta_fi / r
                grad_fi_distrib_up[n][m] = (fi_distrib[n][0] - fi_distrib[n][m]) / delta_fi / r
                velocity_distrib_bot[n][m] = -perm / visc_mid_bot * (
                            Pres_distrib[n][m] - Pres_distrib[n][m - 1]) / delta_fi / r
                grad_fi_distrib_bot[n][m] = (fi_distrib[n][m] - fi_distrib[n][m - 1]) / delta_fi / r

            elif (m == M_1 + 1 and n <= N_1):
                visc_mid_bot = (viscosity_distrib[n][m] + viscosity_distrib[n][m - 1]) / 2
                visc_mid_up = (viscosity_distrib[n][m] + viscosity_distrib[n][m + 1]) / 2
                velocity_distrib_up[n][m] = perm / visc_mid_up * (
                        Pres_distrib[n][m + 1] - Pres_distrib[n][m]) / delta_fi / r
                grad_fi_distrib_up[n][m] = (fi_distrib[n][m + 1] - fi_distrib[n][m]) / delta_fi / r
                velocity_distrib_bot[n][m] = -q_well
                grad_fi_distrib_bot[n][m] = (fi_distrib[n][m] - fi_distrib[n][m - 1]) / delta_fi / r


            elif m == M_2 + 1 and n <= N_2:
                visc_mid_bot = (viscosity_distrib[n][m] + viscosity_distrib[n][m - 1]) / 2
                visc_mid_up = (viscosity_distrib[n][m] + viscosity_distrib[n][m + 1]) / 2
                velocity_distrib_up[n][m] = perm / visc_mid_up * (
                        Pres_distrib[n][m + 1] - Pres_distrib[n][m]) / delta_fi / r
                grad_fi_distrib_up[n][m] = (fi_distrib[n][m + 1] - fi_distrib[n][m]) / delta_fi / r
                velocity_distrib_bot[n][m] = -q_well
                grad_fi_distrib_bot[n][m] = (fi_distrib[n][m] - fi_distrib[n][m - 1]) / delta_fi / r

                # B[j][0] = B[j][0] - 1 / (sum(delta_r_list[0:n+1]) + r_well) / delta_fi_list[m] * (viscosity_matrix[n][m]*q/perm)
            elif m == M_1 and n <= N_1:
                visc_mid_bot = (viscosity_distrib[n][m] + viscosity_distrib[n][m - 1]) / 2
                visc_mid_up = (viscosity_distrib[n][m] + viscosity_distrib[n][m + 1]) / 2
                velocity_distrib_up[n][m] = q_well
                grad_fi_distrib_up[n][m] = (fi_distrib[n][m + 1] - fi_distrib[n][m]) / delta_fi / r
                velocity_distrib_bot[n][m] = -perm / visc_mid_bot * (
                        Pres_distrib[n][m] - Pres_distrib[n][m-1]) / delta_fi / r
                grad_fi_distrib_bot[n][m] = (fi_distrib[n][m] - fi_distrib[n][m - 1]) / delta_fi / r

                # B[j][0] = B[j][0] - 1 / (sum(delta_r_list[0:n+1]) + r_well) / delta_fi_list[m] * (viscosity_matrix[n][m]*q/perm)
            elif m == M_2 and n <= N_2:
                visc_mid_bot = (viscosity_distrib[n][m] + viscosity_distrib[n][m - 1]) / 2
                visc_mid_up = (viscosity_distrib[n][m] + viscosity_distrib[n][m + 1]) / 2
                velocity_distrib_up[n][m] = q_well
                grad_fi_distrib_up[n][m] = (fi_distrib[n][m + 1] - fi_distrib[n][m]) / delta_fi / r
                velocity_distrib_bot[n][m] = -perm / visc_mid_bot * (
                        Pres_distrib[n][m] - Pres_distrib[n][m-1]) / delta_fi / r
                grad_fi_distrib_bot[n][m] = (fi_distrib[n][m] - fi_distrib[n][m - 1]) / delta_fi / r

                # B[j][0] = B[j][0] - 1 / (sum(delta_r_list[0:n+1]) + r_well) / delta_fi_list[m] * (viscosity_matrix[n][m]*q/perm)
            # if m == M_1 and n == N_1:
            #     # c1 = 1 / delta_r_list[n] ** 2
            #     # c2 = 1 / 2 / delta_r_list[n]
            #     # B[j][0] = B[j][0] - (c1 - c2 / (sum(delta_r_list[0:n+1]) + r_well))*delta_r_list[n]*viscosity_matrix[n][m]*q/perm
            #
            # if m == M_2 and n == N_2:
            #     # c1 = 1 / delta_r_list[n] ** 2
            #     # c2 = 1 / 2 / delta_r_list[n]
            #     # B[j][0] = B[j][0] - (c1 - c2 / (sum(delta_r_list[0:n+1]) + r_well))*delta_r_list[n]*viscosity_matrix[n][m]*q/perm
            #     B[j][0] = B[j][0] - r_i_minus_0_5 / r_i / delta_r_i * q



            else:
                #visc_mid_bot = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n][m-1]) ** (-1)
                #visc_mid_up = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n][m+1]) ** (-1)
                visc_mid_bot = (viscosity_distrib[n][m] + viscosity_distrib[n][m - 1])/2
                visc_mid_up = (viscosity_distrib[n][m] + viscosity_distrib[n][m + 1])/2

                viscosity_distrib_bot[n][m] = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n][m - 1]) ** (-1)
                viscosity_distrib_up[n][m] = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n][m + 1]) ** (-1)

                velocity_distrib_up[n][m] = -perm / visc_mid_up * (
                            Pres_distrib[n][m + 1] - Pres_distrib[n][m]) / delta_fi / r
                grad_fi_distrib_up[n][m] = (fi_distrib[n][m + 1] - fi_distrib[n][m]) / delta_fi / r
                velocity_distrib_bot[n][m] = -perm / visc_mid_bot * (
                            Pres_distrib[n][m] - Pres_distrib[n][m - 1]) / delta_fi / r
                grad_fi_distrib_bot[n][m] = (fi_distrib[n][m] - fi_distrib[n][m - 1]) / delta_fi / r

            #velocity_distrib_left[n][m] = -perm/visc_mid_left * (Pres_distrib[n][m] - Pres_distrib[n-1][m])/delta_r
            #velocity_distrib_right[n][m] = -perm / visc_mid_right * (Pres_distrib[n+1][m] - Pres_distrib[n][m]) / delta_r
            #velocity_distrib_bot[n][m] = -perm / visc_mid_bot * (Pres_distrib[n][m] - Pres_distrib[n][m-1]) / delta_fi/r


            # grad_fi_distrib_left[n][m] = (fi_distrib[n][m] - fi_distrib[n-1][m])/delta_r
            # grad_fi_distrib_right[n][m] = (fi_distrib[n+1][m] - fi_distrib[n][m]) / delta_r
            # grad_fi_distrib_bot[n][m] = (fi_distrib[n][m] - fi_distrib[n][m-1]) / delta_fi/r
            # grad_fi_distrib_up[n][m] = (fi_distrib[n][m+1] - fi_distrib[n][m]) / delta_fi/r

            velocity_distrib = np.array([velocity_distrib_left, velocity_distrib_right, velocity_distrib_bot, velocity_distrib_up])
            grad_fi_distrib = np.array([grad_fi_distrib_left, grad_fi_distrib_right, grad_fi_distrib_bot, grad_fi_distrib_up])
            viscosity_distrib_total = np.array([viscosity_distrib_left, viscosity_distrib_right, viscosity_distrib_bot, viscosity_distrib_up])
    return velocity_distrib, grad_fi_distrib, viscosity_distrib_total
