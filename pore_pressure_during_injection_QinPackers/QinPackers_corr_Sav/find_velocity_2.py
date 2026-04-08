import numpy as np


def fi_distrib_in_cell_center(fi_distrib, X_matrix_cell, Y_matrix_cell, X_matrix_cell_center, Y_matrix_cell_center):
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((X_matrix_cell, Y_matrix_cell), fi_distrib)
    return interp(X_matrix_cell_center, Y_matrix_cell_center)


def find_velocity_grad_fi(Pres_distrib, viscosity_distrib, delta_r, delta_fi, perm, r_well, fi_distrib, q_well, N_1, N_2, M_1, M_2, func_list_0, func_list_end):
    N_r_full = Pres_distrib.shape[0]
    M_fi_full = Pres_distrib.shape[1]

    P = Pres_distrib
    V = viscosity_distrib
    F = fi_distrib
    fl0 = np.array(func_list_0)
    fle = np.array(func_list_end)

    # Радиус в центре каждой ячейки: r[n] = r_well + n*dr + dr/2
    r_arr = r_well + np.arange(N_r_full) * delta_r + delta_r / 2  # shape (N_r_full,)
    r_2d = r_arr[:, np.newaxis]  # shape (N_r_full, 1) для broadcasting

    # ====================================================================
    # 1. Вязкость на гранях (среднее арифметическое соседей)
    # ====================================================================
    # Радиальное направление: mu_{i+1/2} = (V[n] + V[n+1]) / 2
    mu_right = np.zeros_like(P)
    mu_right[:-1, :] = (V[:-1, :] + V[1:, :]) / 2

    mu_left = np.zeros_like(P)
    mu_left[1:, :] = (V[1:, :] + V[:-1, :]) / 2

    # Угловое направление: mu_{j+1/2} = (V[m] + V[m+1]) / 2 с периодичностью
    mu_up = (V + np.roll(V, -1, axis=1)) / 2   # V[n,m] + V[n,m+1]
    mu_bot = (V + np.roll(V, 1, axis=1)) / 2    # V[n,m] + V[n,m-1]

    # ====================================================================
    # 2. Скорости Дарси на гранях — общий случай (внутренние ячейки)
    # ====================================================================
    # v_right = -(k/mu) * (P[n+1] - P[n]) / dr
    velocity_distrib_right = np.zeros_like(P)
    velocity_distrib_right[:-1, :] = -perm / mu_right[:-1, :] * (P[1:, :] - P[:-1, :]) / delta_r

    # v_left = (k/mu) * (P[n-1] - P[n]) / dr
    velocity_distrib_left = np.zeros_like(P)
    velocity_distrib_left[1:, :] = perm / mu_left[1:, :] * (P[:-1, :] - P[1:, :]) / delta_r

    # v_up = -(k/mu) * (P[m+1] - P[m]) / (dfi * r)
    P_up = np.roll(P, -1, axis=1)  # P[n, m+1] с периодичностью
    velocity_distrib_up = -perm / mu_up * (P_up - P) / delta_fi / r_2d

    # v_bot = (k/mu) * (P[m-1] - P[m]) / (dfi * r)
    P_bot = np.roll(P, 1, axis=1)  # P[n, m-1] с периодичностью
    velocity_distrib_bot = perm / mu_bot * (P_bot - P) / delta_fi / r_2d

    # ====================================================================
    # 3. Градиенты level-set — общий случай
    # ====================================================================
    # grad_right = (F[n+1] - F[n]) / dr
    grad_fi_distrib_right = np.zeros_like(P)
    grad_fi_distrib_right[:-1, :] = (F[1:, :] - F[:-1, :]) / delta_r

    # grad_left = (F[n] - F[n-1]) / dr
    grad_fi_distrib_left = np.zeros_like(P)
    grad_fi_distrib_left[1:, :] = (F[1:, :] - F[:-1, :]) / delta_r

    # grad_up = (F[m+1] - F[m]) / (dfi * r)
    F_up = np.roll(F, -1, axis=1)
    grad_fi_distrib_up = (F_up - F) / delta_fi / r_2d

    # grad_bot = (F[m] - F[m-1]) / (dfi * r)
    F_bot = np.roll(F, 1, axis=1)
    grad_fi_distrib_bot = (F - F_bot) / delta_fi / r_2d

    # ====================================================================
    # 4. Граничные условия: n=0 (стенка скважины)
    # ====================================================================
    velocity_distrib_left[0, :] = 0
    # v_right[0] уже вычислен правильно
    # grad_left[0] использует ghost-ячейку func_list_0
    grad_fi_distrib_left[0, :] = (F[0, :] - fl0) / delta_r
    # grad_right[0] уже правильный

    # ====================================================================
    # 5. Граничные условия: n=N_r_full-1 (внешняя граница)
    # ====================================================================
    velocity_distrib_right[-1, :] = 0
    # v_left[-1] уже вычислен правильно
    # grad_right[-1] использует ghost func_list_end
    grad_fi_distrib_right[-1, :] = (fle - F[-1, :]) / delta_r
    # grad_left[-1] уже правильный

    # ====================================================================
    # 6. Граничные условия: трещины (m=M_1, M_1-1, M_2, M_2-1, n<=N_1/N_2)
    # Трещина — источник потока в угловом направлении.
    # На правой стороне трещины (m=M_1, m=M_2): v_bot = q, v_up = Дарси, A_sym_left = 0
    # На левой стороне (m=M_1-1, m=M_2-1): v_up = -q, v_bot = Дарси, A_sym_right = 0
    # ====================================================================

    # Трещина 1: m = M_1 (правая сторона), n = 0..N_1
    frac_n_1 = slice(0, N_1 + 1)
    velocity_distrib_bot[frac_n_1, M_1] = q_well
    # v_up пересчитываем через Дарси (уже правильно из общего случая)

    # n=0 на правой стороне трещины: v_left = 0 (уже задано)

    # Трещина 1: m = M_1-1 (левая сторона), n = 0..N_1
    velocity_distrib_up[frac_n_1, M_1 - 1] = -q_well
    # v_bot пересчитывается через Дарси (уже правильно)

    # Трещина 2: m = M_2, n = 0..N_2
    frac_n_2 = slice(0, N_2 + 1)
    velocity_distrib_bot[frac_n_2, M_2] = q_well

    # Трещина 2: m = M_2-1, n = 0..N_2
    velocity_distrib_up[frac_n_2, M_2 - 1] = -q_well

    # ====================================================================
    # 7. Сборка результатов
    # ====================================================================
    velocity_distrib_result = np.array([velocity_distrib_left, velocity_distrib_right,
                                        velocity_distrib_bot, velocity_distrib_up])
    grad_fi_distrib_result = np.array([grad_fi_distrib_left, grad_fi_distrib_right,
                                       grad_fi_distrib_bot, grad_fi_distrib_up])
    viscosity_distrib_total = np.array([np.zeros_like(P), np.zeros_like(P),
                                        np.zeros_like(P), np.zeros_like(P)])

    return velocity_distrib_result, grad_fi_distrib_result, viscosity_distrib_total
