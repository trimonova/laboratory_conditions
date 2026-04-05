import numpy as np
from scipy.sparse import coo_matrix, linalg, hstack, vstack, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def PorePressure_in_Time(N_r_full, M_fi_full, Pres_distrib, c3_oil, c3_water, CP_dict, q, wells_frac_coords, wells_coords,
                         delta_r_list, delta_fi_list, viscosity_matrix, fi, C_total, perm, delta_t,
                         r_well, M_1, M_2, N_1, N_2):

    # пластовое давление во всей области на нулевом временном шаге
    B = np.zeros((N_r_full*M_fi_full, 1))
    for m in range(M_fi_full):

        A = np.zeros((N_r_full, N_r_full))

        A_sym_right = np.zeros((N_r_full, N_r_full))
        A_sym_left = np.zeros((N_r_full, N_r_full))

        for n in range(0, N_r_full):
            delta_r_i = delta_r_list[n]
            delta_r_i_minus_0_5 = delta_r_i_plus_0_5 = delta_r_i
            r_i = n*delta_r_i + delta_r_i/2 + r_well
            r_i_minus_0_5 = r_i - delta_r_i/2
            r_i_plus_0_5 = r_i + delta_r_i/2
            if n != N_r_full-1:
                mu_i_plus_0_5 = (viscosity_matrix[n][m] + viscosity_matrix[n+1][m])/2
            if n != 0:
                mu_i_minus_0_5 = (viscosity_matrix[n][m] + viscosity_matrix[n - 1][m]) / 2
            delta_fi_j = delta_fi_list[m]
            delta_fi_j_minus_0_5 = delta_fi_j_plus_0_5 = delta_fi_j
            if m == M_fi_full-1:
                mu_j_plus_0_5 = (viscosity_matrix[n][m] + viscosity_matrix[n][0]) / 2
                mu_j_minus_0_5 = (viscosity_matrix[n][m] + viscosity_matrix[n][m-1]) / 2
            else:
                mu_j_plus_0_5 = (viscosity_matrix[n][m] + viscosity_matrix[n][m+1]) / 2
                mu_j_minus_0_5 = (viscosity_matrix[n][m] + viscosity_matrix[n][m-1]) / 2
            k_fluid = fi * C_total
            #c3_fluid = k_fluid/delta_t

            A_sym_right[n][n] = perm/r_i/delta_fi_j/mu_j_plus_0_5/r_i/delta_fi_j_plus_0_5
            A_sym_left[n][n] = perm/r_i/delta_fi_j/mu_j_minus_0_5/r_i/delta_fi_j_minus_0_5

            if n == 0:
                A[n][n + 1] = perm*r_i_plus_0_5/r_i/delta_r_i/delta_r_i_plus_0_5/mu_i_plus_0_5
                A[n][n] = -(A[n][n + 1] + A_sym_right[n][n] + A_sym_left[n][n])
            elif n == N_r_full-1:
                A[n][n] = -(A[n][n - 1] + A_sym_right[n][n] + A_sym_left[n][n])
                A[n][n - 1] = perm*r_i_minus_0_5/r_i/delta_r_i/delta_r_i_minus_0_5/mu_i_minus_0_5
            else:

                A[n][n - 1] = perm*r_i_minus_0_5/r_i/delta_r_i/delta_r_i_minus_0_5/mu_i_minus_0_5
                A[n][n + 1] = perm*r_i_plus_0_5/r_i/delta_r_i/delta_r_i_plus_0_5/mu_i_plus_0_5
                A[n][n] = -(A[n][n + 1] + A[n][n - 1] + A_sym_right[n][n] + A_sym_left[n][n])


        A_sym_right_coo = coo_matrix(A_sym_right)
        A_sym_left_coo = coo_matrix(A_sym_left)

        if m == 0:
            A_line_1 = hstack([A, A_sym_right_coo, np.zeros((N_r_full, N_r_full * M_fi_full - 3 * N_r_full)), A_sym_left_coo])
            A_full = coo_matrix(A_line_1)
        elif m == M_fi_full-1:
            A_line_end = hstack(
                    [A_sym_right_coo, np.zeros((N_r_full, N_r_full * M_fi_full - 3 * N_r_full)), A_sym_left_coo, A])
            A_full = vstack([A_full, A_line_end])
        else:
            A_line = hstack([np.zeros((N_r_full, N_r_full * (m - 1))), A_sym_left_coo, A, A_sym_right_coo,
                                 np.zeros((N_r_full, N_r_full * M_fi_full - (3 + (m - 1)) * N_r_full))])
            A_full = vstack([A_full, A_line])

    j = 0
    for m in range(M_fi_full):
        for n in range(N_r_full):
            k_fluid = fi * C_total
            c3_fluid = k_fluid / delta_t
            B[j][0] = -c3_fluid * Pres_distrib[n][m]
            if n == 0:
                r_i = delta_r_list[n] + r_well
                r_i_minus_0_5 = r_i - delta_r_list[n] / 2
                B[j][0] = B[j][0] - r_i_minus_0_5/r_i/delta_r_i*q
            j += 1


    def sort_func(well_coord_couple):
        return (well_coord_couple[1]) * N_r_full + well_coord_couple[0]

    wells_frac_coords.sort(key=sort_func)
    wells_frac_coords_reverse = wells_frac_coords[:: -1]

    wells_coords.sort(key=sort_func)
    wells_coords_reverse = wells_coords[:: -1]


    for coord_couple in wells_coords_reverse:
        A_well_column_coo = A_full.getcol((coord_couple[1])*N_r_full + coord_couple[0])
        A_well_column = A_well_column_coo.toarray()
        for cell_number in range(len(A_well_column)):
            if A_well_column[cell_number] != 0:
                B[cell_number][0] = B[cell_number] - A_well_column[cell_number]*CP_dict[coord_couple]


    #A_full = A_full.toarray()
    # for (n,m) in bound_coord_cell:
    #     print(n,m)
    #     if m != 0 and m != M_fi_full-1:
    #         print(A_full[(m)*N_r_full + n][(m)*N_r_full + n])
    #         print(A_full[(m) * N_r_full + n][(m) * N_r_full + n-1])
    #         print(A_full[(m) * N_r_full + n][(m) * N_r_full + n+1])
    #         print(A_full[(m)*N_r_full + n][(m-1)*N_r_full + n])
    #         print(A_full[(m)*N_r_full + n][(m+1)*N_r_full + n])
    #         print(B[m*N_r_full + n][0])

    #A_full = coo_matrix(A_full)

    for coord_couple in wells_coords_reverse:
        # A_well_column_coo = A_full.getcol((coord_couple[1]-1)*N_r_full + coord_couple[0])
        # A_well_column = A_well_column_coo.toarray()
        # for cell_number in range(len(A_well_column)):
        #     if A_well_column[cell_number] != 0:
        #         B[cell_number] = B[cell_number] - A_well_column[cell_number]*CP_dict[coord_couple]

        A_full = A_full.tocsr()
        all_cols = np.arange(A_full.shape[1])
        cols_to_keep = np.where(np.logical_not(np.in1d(all_cols, [(coord_couple[1])*N_r_full + coord_couple[0]])))[0]
        A_full = A_full[:, cols_to_keep]
        A_full = A_full[cols_to_keep, :]

        B = np.delete(B, (coord_couple[1]) * N_r_full + coord_couple[0], axis=0)

    # for b in range(B.shape[0]):
    #     if B[b][0] < -50157142857143:
    #         print(B[b][0], b)

    P_new = spsolve(A_full, B)
    print('P_new', min(P_new), max(P_new))
    P_new_along_the_fracture_1 = []
    P_new_along_the_fracture_2 = []
    P_new_along_the_fracture_3 = []
    P_new_along_the_fracture_4 = []
    P_new_along_the_fracture_5 = []
    P_new_along_the_fracture_6 = []



    # plt.plot(P_new_along_the_fracture_1)
    # plt.show()
    for coord_couple in wells_coords:
        N = coord_couple[0]
        M = coord_couple[1]
        #P_new = np.insert(P_new, (M)*N_r_full + N, P_new[(M-1)*N_r_full + N] + q*viscosity_matrix[N][M-1]*delta_fi_list[M-1]*(sum(delta_r_list[0:N]) + r_well)/perm)
        #P_new = np.insert(P_new, (M)*N_r_full + N, P_new[(M-1)*N_r_full + N])
        #P_new = np.insert(P_new, (M)*N_r_full + N, 100000)
        P_new = np.insert(P_new, (M)*N_r_full + N, CP_dict[coord_couple])


    print('P_new_insert', min(P_new), max(P_new))

    P_new = P_new.reshape(N_r_full*M_fi_full, 1)
    Pres_end = np.zeros((N_r_full, M_fi_full))
    j = 0
    for m in range(M_fi_full):
        for n in range(N_r_full):
            Pres_end[n][m] = P_new[j][0]
            j += 1

    return Pres_end, A, B