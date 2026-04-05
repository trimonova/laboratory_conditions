import numpy as np
from scipy.interpolate import interp1d


def find_velocity_grad_fi(Pres_distrib, viscosity_distrib, delta_r, delta_fi, perm, r_well, fi_distrib, q_well):

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



    for (n, m), value in np.ndenumerate(Pres_distrib):

        fi_line = list(fi_distrib[:, m])
        f_interp = interp1d(r_line, fi_line, fill_value='extrapolate')
        f_0 = f_interp(r_well - delta_r / 2)
        f_end = f_interp(r_well + delta_r * len(fi_line) + delta_r / 2)

        r = r_well + n*delta_r + delta_r/2
        if n == 0:
            visc_mid_right = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n + 1][m]) ** (-1)
            #visc_mid_right = (viscosity_distrib[n][m] + viscosity_distrib[n + 1][m])/2
            viscosity_distrib_right[n][m] = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n + 1][m]) ** (-1)
            velocity_distrib_left[n][m] = q_well
            velocity_distrib_right[n][m] = -perm / visc_mid_right * (
                    Pres_distrib[n + 1][m] - Pres_distrib[n][m]) / delta_r
            grad_fi_distrib_left[n][m] = (fi_distrib[n][m] - f_0)/delta_r

            grad_fi_distrib_right[n][m] = (fi_distrib[n+1][m] - fi_distrib[n][m])/delta_r

        elif n == np.shape(Pres_distrib)[0]-1:
            visc_mid_left = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n - 1][m]) ** (-1)
            #visc_mid_left = (viscosity_distrib[n][m] + viscosity_distrib[n - 1][m]) / 2
            viscosity_distrib_left[n][m] = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n - 1][m]) ** (-1)
            velocity_distrib_right[n][m] = 0
            velocity_distrib_left[n][m] = -perm / visc_mid_left * (
                    Pres_distrib[n][m] - Pres_distrib[n - 1][m]) / delta_r
            grad_fi_distrib_left[n][m] = (fi_distrib[n][m] - fi_distrib[n-1][m]) / delta_r

            grad_fi_distrib_right[n][m] = (f_end - fi_distrib[n][m]) / delta_r

        else:
            visc_mid_left = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n - 1][m]) ** (-1)
            visc_mid_right = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n + 1][m]) ** (-1)
            #visc_mid_right = (viscosity_distrib[n][m] + viscosity_distrib[n + 1][m]) / 2
            #visc_mid_left = (viscosity_distrib[n][m] + viscosity_distrib[n - 1][m]) / 2
            viscosity_distrib_right[n][m] = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n + 1][m]) ** (-1)
            viscosity_distrib_left[n][m] = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n - 1][m]) ** (-1)
            velocity_distrib_left[n][m] = -perm / visc_mid_left * (
                        Pres_distrib[n][m] - Pres_distrib[n - 1][m]) / delta_r
            velocity_distrib_right[n][m] = -perm / visc_mid_right * (
                        Pres_distrib[n + 1][m] - Pres_distrib[n][m]) / delta_r

            grad_fi_distrib_left[n][m] = (fi_distrib[n][m] - fi_distrib[n - 1][m]) / delta_r
            grad_fi_distrib_right[n][m] = (fi_distrib[n + 1][m] - fi_distrib[n][m]) / delta_r

        if m == np.shape(Pres_distrib)[1]-1:
            visc_mid_up = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n][0]) ** (-1)
            visc_mid_bot = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n][m - 1]) ** (-1)
            #visc_mid_up = (viscosity_distrib[n][m] + viscosity_distrib[n][0])/2
            #visc_mid_bot = (viscosity_distrib[n][m] + viscosity_distrib[n][m - 1])/2
            viscosity_distrib_up[n][m] = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n][0]) ** (-1)
            viscosity_distrib_bot[n][m] = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n][m - 1]) ** (-1)

            velocity_distrib_up[n][m] = -perm / visc_mid_up * (
                        Pres_distrib[n][0] - Pres_distrib[n][m]) / delta_fi / r
            grad_fi_distrib_up[n][m] = (fi_distrib[n][0] - fi_distrib[n][m]) / delta_fi / r
            velocity_distrib_bot[n][m] = -perm / visc_mid_bot * (
                        Pres_distrib[n][m] - Pres_distrib[n][m - 1]) / delta_fi / r
            grad_fi_distrib_bot[n][m] = (fi_distrib[n][m] - fi_distrib[n][m - 1]) / delta_fi / r


        else:
            visc_mid_bot = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n][m-1]) ** (-1)
            visc_mid_up = 2 * (1 / viscosity_distrib[n][m] + 1 / viscosity_distrib[n][m+1]) ** (-1)
            #visc_mid_bot = (viscosity_distrib[n][m] + viscosity_distrib[n][m - 1])/2
            #visc_mid_up = (viscosity_distrib[n][m] + viscosity_distrib[n][m + 1])/2

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
