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

        for n in range(1, N_r_full - 1):
            k_fluid = viscosity_matrix[n][m] * fi * C_total / perm
            c3_fluid = k_fluid/delta_t
            c1 = 1 / delta_r_list[n] ** 2
            c2 = 1 / 2 / delta_r_list[n]
            A[n][n - 1] = c1 - c2 / (sum(delta_r_list[0:n+1])+r_well)

            A[n][n] = -2 * c1 - c3_fluid - 2 / (sum(delta_r_list[0:n+1])+r_well) ** 2 / delta_fi_list[m] ** 2

            A[n][n + 1] = c1 + c2 / (sum(delta_r_list[0:n+1])+r_well)

        k_fluid = viscosity_matrix[0][m] * fi * C_total / perm
        c3_fluid = k_fluid / delta_t
        c1 = 1 / delta_r_list[0] ** 2
        c2 = 1 / 2 / delta_r_list[0]

        A[0][0] = -c1 - c3_fluid - 2/(delta_r_list[0]+r_well)**2/delta_fi_list[m]**2 - c2/(delta_r_list[0]+r_well)

        A[0][1] = c1 + c2 / (delta_r_list[0]+r_well)

        c1 = 1 / delta_r_list[N_r_full - 1] ** 2
        c2 = 1 / 2 / delta_r_list[N_r_full - 1]
        A[N_r_full - 1][N_r_full - 1] = -2 * c1 - c3_water + c1 + c2 / (sum(delta_r_list[0:N_r_full])+r_well) - 2/(sum(delta_r_list[0:N_r_full])+r_well)**2/delta_fi_list[m]**2
        A[N_r_full - 1][N_r_full - 2] = c1 - c2 / (sum(delta_r_list[0:N_r_full])+r_well)

        c4 = 1 / delta_fi_list[m] ** 2
        A_sym_right = np.zeros((N_r_full, N_r_full))
        A_sym_left = np.zeros((N_r_full, N_r_full))
        for n in range(0, N_r_full):
            A_sym_right[n][n] = c4 / (sum(delta_r_list[0:n+1])+r_well) ** 2
            A_sym_left[n][n] = c4 / (sum(delta_r_list[0:n+1]) + r_well) ** 2

            if (m == M_1 + 1 and n <= N_1):
                A[n][n] = A[n][n] + 1 / (sum(delta_r_list[0:n+1]) + r_well) ** 2 / delta_fi_list[m] ** 2
                A_sym_left[n][n] = 0
            elif (m == M_2 + 1 and n <= N_2):
                A[n][n] = A[n][n] + 1 / (sum(delta_r_list[0:n+1]) + r_well) ** 2 / delta_fi_list[m] ** 2
                A_sym_left[n][n] = 0
            elif (m == M_1 - 1 and n <= N_1):
                A[n][n] = A[n][n] + 1 / (sum(delta_r_list[0:n+1]) + r_well) ** 2 / delta_fi_list[m] ** 2
                A_sym_right[n][n] = 0
            elif (m == M_2 - 1 and n <= N_2):
                A[n][n] = A[n][n] + 1 / (sum(delta_r_list[0:n+1]) + r_well) ** 2 / delta_fi_list[m] ** 2
                A_sym_right[n][n] = 0
            elif m == M_1 and n == N_1:
                c1 = 1 / delta_r_list[n] ** 2
                c2 = 1 / 2 / delta_r_list[n]
                A[n-1][n] = 0
                A[n][n] = A[n][n] + (c1 - c2 / (sum(delta_r_list[0:n+1]) + r_well))
            elif m == M_2 and n == N_2:
                c1 = 1 / delta_r_list[n] ** 2
                c2 = 1 / 2 / delta_r_list[n]
                A[n-1][n] = 0
                A[n][n] = A[n][n] + (c1 - c2 / (sum(delta_r_list[0:n+1]) + r_well))


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
            k_fluid = viscosity_matrix[n][m] * fi * C_total / perm
            c3_fluid = k_fluid / delta_t
            B[j][0] = -c3_fluid * Pres_distrib[n][m]
            if (m == M_1 + 1 and n < N_1):
                B[j][0] = B[j][0] - 1 / (sum(delta_r_list[0:n+1]) + r_well) / delta_fi_list[m] * (viscosity_matrix[n][m]*q/perm)
            elif m == M_2 + 1 and n < N_2:
                B[j][0] = B[j][0] - 1 / (sum(delta_r_list[0:n+1]) + r_well) / delta_fi_list[m] * (viscosity_matrix[n][m]*q/perm)
            elif m == M_1 - 1 and n < N_1:
                B[j][0] = B[j][0] - 1 / (sum(delta_r_list[0:n+1]) + r_well) / delta_fi_list[m] * (viscosity_matrix[n][m]*q/perm)
            elif m == M_2 - 1 and n < N_2:
                B[j][0] = B[j][0] - 1 / (sum(delta_r_list[0:n+1]) + r_well) / delta_fi_list[m] * (viscosity_matrix[n][m]*q/perm)
            if m == M_1 and n == N_1:
                c1 = 1 / delta_r_list[n] ** 2
                c2 = 1 / 2 / delta_r_list[n]
                B[j][0] = B[j][0] - (c1 - c2 / (sum(delta_r_list[0:n+1]) + r_well))*delta_r_list[n]*viscosity_matrix[n][m]*q/perm
            if m == M_2 and n == N_2:
                c1 = 1 / delta_r_list[n] ** 2
                c2 = 1 / 2 / delta_r_list[n]
                B[j][0] = B[j][0] - (c1 - c2 / (sum(delta_r_list[0:n+1]) + r_well))*delta_r_list[n]*viscosity_matrix[n][m]*q/perm

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

    for coord_couple in wells_frac_coords_reverse:
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
    for coord_couple in wells_frac_coords:
        N = coord_couple[0]
        M = coord_couple[1]
        if coord_couple in wells_coords:
            P_new = np.insert(P_new, (M)*N_r_full + N, CP_dict[coord_couple])
        else:
            P_new = np.insert(P_new, (M)*N_r_full + N, P_new[(M-1)*N_r_full + N] + q*viscosity_matrix[N][M-1]*delta_fi_list[M-1]*(sum(delta_r_list[0:N]) + r_well)/perm)
        #P_new = np.insert(P_new, (M)*N_r_full + N, P_new[(M-1)*N_r_full + N])
        #P_new = np.insert(P_new, (M)*N_r_full + N, 100000)
        #P_new = np.insert(P_new, (coord_couple[1] - 1) * N_r_full + coord_couple[0], CP_dict[coord_couple])

    for coord_couple in wells_frac_coords:
        N = coord_couple[0]
        M = coord_couple[1]
        print(N, M)
        P_new_along_the_fracture_1.append(P_new[(M-1) * N_r_full + N] + q*viscosity_matrix[N][M]*delta_fi_list[M]*(sum(delta_r_list[0:N]) + r_well)/perm)
        P_new_along_the_fracture_2.append(P_new[(M-1) * N_r_full + N])
        P_new_along_the_fracture_3.append(P_new[(M) * N_r_full + N])
        P_new_along_the_fracture_4.append(P_new[(M+1) * N_r_full + N])

    print('P_new_insert', min(P_new), max(P_new))

    P_new = P_new.reshape(N_r_full*M_fi_full, 1)
    Pres_end = np.zeros((N_r_full, M_fi_full))
    j = 0
    for m in range(M_fi_full):
        for n in range(N_r_full):
            Pres_end[n][m] = P_new[j][0]
            j += 1

    return Pres_end, A, B