import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# если трещина проходит вдоль границ ячеек, а не через их центры
def PorePressure_in_Time(N_r_full, M_fi_full, Pres_distrib, c3_oil, c3_water, CP_dict, q, wells_frac_coords, wells_coords,
                         delta_r_list, delta_fi_list, viscosity_matrix, fi, C_total, perm, delta_t,
                         r_well, M_1, M_2, N_1, N_2):
    k_fluid = fi * C_total
    c3_fluid = k_fluid / delta_t
    V = viscosity_matrix
    N = N_r_full
    M = M_fi_full

    # ====================================================================
    # Геометрия сетки — 1D массивы по n
    # ====================================================================
    dr = np.array(delta_r_list)               # (N,)
    dfi = np.array(delta_fi_list)             # (M,)
    r_i = np.arange(N) * dr + dr / 2 + r_well  # центры ячеек (N,)
    r_i_minus = r_i - dr / 2                  # левые грани
    r_i_plus = r_i + dr / 2                   # правые грани

    # ====================================================================
    # Вязкость на гранях — 2D массивы (N, M)
    # ====================================================================
    # Радиальные грани: mu_{i+1/2}, mu_{i-1/2}
    mu_r_plus = np.zeros((N, M))   # (V[n] + V[n+1]) / 2
    mu_r_plus[:-1, :] = (V[:-1, :] + V[1:, :]) / 2

    mu_r_minus = np.zeros((N, M))  # (V[n] + V[n-1]) / 2
    mu_r_minus[1:, :] = (V[1:, :] + V[:-1, :]) / 2

    # Угловые грани: mu_{j+1/2}, mu_{j-1/2} (с периодичностью)
    mu_fi_plus = (V + np.roll(V, -1, axis=1)) / 2   # V[n,m]+V[n,m+1]
    mu_fi_minus = (V + np.roll(V, 1, axis=1)) / 2    # V[n,m]+V[n,m-1]

    # ====================================================================
    # Коэффициенты матрицы — 2D массивы (N, M)
    # ====================================================================
    # coef_right[n,m] = A[n][n+1] = perm * r_{i+1/2} / r_i / dr^2 / mu_{i+1/2}
    coef_right = np.zeros((N, M))
    coef_right[:-1, :] = perm * r_i_plus[:-1, None] / r_i[:-1, None] / dr[:-1, None]**2 / mu_r_plus[:-1, :]

    # coef_left[n,m] = A[n][n-1] = perm * r_{i-1/2} / r_i / dr^2 / mu_{i-1/2}
    coef_left = np.zeros((N, M))
    coef_left[1:, :] = perm * r_i_minus[1:, None] / r_i[1:, None] / dr[1:, None]**2 / mu_r_minus[1:, :]

    # sym_right[n,m] = A_sym_right = perm / (r_i * dfi)^2 / mu_{j+1/2}
    sym_right = perm / (r_i[:, None] * dfi[None, :])**2 / mu_fi_plus

    # sym_left[n,m] = A_sym_left = perm / (r_i * dfi)^2 / mu_{j-1/2}
    sym_left = perm / (r_i[:, None] * dfi[None, :])**2 / mu_fi_minus

    # ====================================================================
    # Граничные условия: n=0 (стенка скважины) — нет левого соседа
    # ====================================================================
    coef_left[0, :] = 0   # уже 0 из инициализации

    # ====================================================================
    # Граничные условия: n=N-1 (внешняя граница) — нет правого соседа
    # ====================================================================
    coef_right[-1, :] = 0  # уже 0

    # ====================================================================
    # Граничные условия: трещины
    # m=M_1, n<=N_1: sym_left = 0 (трещина — правая сторона)
    # m=M_1-1, n<=N_1: sym_right = 0 (трещина — левая сторона)
    # m=M_2, n<=N_2: sym_left = 0
    # m=M_2-1, n<=N_2: sym_right = 0
    # n=0, m=M_1: sym_left = 0 (уже 0 от n=M_1 условия)
    # n=0, m=M_1-1: sym_right = 0
    # n=0, m=M_2: sym_left = 0
    # n=0, m=M_2-1: sym_right = 0
    # ====================================================================
    sym_left[:N_1 + 1, M_1] = 0
    sym_left[:N_2 + 1, M_2] = 0
    sym_right[:N_1 + 1, M_1 - 1] = 0
    sym_right[:N_2 + 1, M_2 - 1] = 0

    # n=0 + трещина: дополнительные обнуления
    # n=0, m=M_1-1: sym_right = 0 (уже задано выше через [:N_1+1])
    # n=0, m=M_2-1: sym_right = 0 (уже задано)
    # n=0, m=M_1: sym_left = 0 (уже задано)
    # n=0, m=M_2: sym_left = 0 (уже задано)

    # ====================================================================
    # Диагональный элемент: A[n][n] = -(coef_right + coef_left + sym_right + sym_left + c3_fluid)
    # ====================================================================
    coef_diag = -(coef_right + coef_left + sym_right + sym_left + c3_fluid)

    # ====================================================================
    # Сборка COO-массивов для разреженной матрицы
    # Глобальная нумерация: idx = m * N + n
    # ====================================================================
    n_idx = np.arange(N)        # [0, 1, ..., N-1]
    m_idx = np.arange(M)        # [0, 1, ..., M-1]
    nn, mm = np.meshgrid(n_idx, m_idx, indexing='ij')  # (N, M)
    global_idx = mm * N + nn  # (N, M) — глобальный индекс каждой ячейки

    rows_list = []
    cols_list = []
    vals_list = []

    # Диагональ: (n,m) -> (n,m)
    rows_list.append(global_idx.ravel())
    cols_list.append(global_idx.ravel())
    vals_list.append(coef_diag.ravel())

    # Под-диагональ (радиально): (n,m) -> (n-1,m) — coef_left
    mask_left = n_idx > 0
    gl_from = global_idx[mask_left, :]
    gl_to = global_idx[mask_left, :] - 1  # n-1 в том же слое m
    vals_left = coef_left[mask_left, :]
    # Убираем нулевые элементы
    nonzero = vals_left != 0
    rows_list.append(gl_from[nonzero])
    cols_list.append(gl_to[nonzero])
    vals_list.append(vals_left[nonzero])

    # Над-диагональ (радиально): (n,m) -> (n+1,m) — coef_right
    mask_right = n_idx < N - 1
    gl_from = global_idx[mask_right, :]
    gl_to = global_idx[mask_right, :] + 1  # n+1 в том же слое m
    vals_right = coef_right[mask_right, :]
    nonzero = vals_right != 0
    rows_list.append(gl_from[nonzero])
    cols_list.append(gl_to[nonzero])
    vals_list.append(vals_right[nonzero])

    # sym_right: (n,m) -> (n, m+1) с периодичностью
    m_next = (mm + 1) % M
    gl_to_right = m_next * N + nn
    nonzero = sym_right != 0
    rows_list.append(global_idx[nonzero])
    cols_list.append(gl_to_right[nonzero])
    vals_list.append(sym_right[nonzero])

    # sym_left: (n,m) -> (n, m-1) с периодичностью
    m_prev = (mm - 1) % M
    gl_to_left = m_prev * N + nn
    nonzero = sym_left != 0
    rows_list.append(global_idx[nonzero])
    cols_list.append(gl_to_left[nonzero])
    vals_list.append(sym_left[nonzero])

    # Сборка разреженной матрицы
    total_size = N * M
    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_vals = np.concatenate(vals_list)
    A_full = csr_matrix((all_vals, (all_rows, all_cols)), shape=(total_size, total_size))

    # ====================================================================
    # Правая часть B — numpy-массивы вместо цикла
    # ====================================================================
    delta_fi_j = dfi[0]  # все dfi одинаковые в текущей конфигурации

    # B = -c3_fluid * P (для всех ячеек)
    # Глобальная нумерация: j = m*N + n, порядок — сначала n внутри m
    B_2d = -c3_fluid * Pres_distrib  # (N, M)

    # Источники на трещинах: B -= 1/(r_i * dfi) * q
    source = -q / (r_i[:, None] * delta_fi_j)  # (N, 1)

    # m=M_1, n=0..N_1
    B_2d[:N_1 + 1, M_1] += source[:N_1 + 1, 0]
    # m=M_2, n=0..N_2
    B_2d[:N_2 + 1, M_2] += source[:N_2 + 1, 0]
    # m=M_1-1, n=0..N_1
    B_2d[:N_1 + 1, M_1 - 1] += source[:N_1 + 1, 0]
    # m=M_2-1, n=0..N_2
    B_2d[:N_2 + 1, M_2 - 1] += source[:N_2 + 1, 0]

    # Переводим в глобальный вектор (порядок: m=0: n=0..N-1, m=1: n=0..N-1, ...)
    B = B_2d.T.reshape(-1, 1)  # transpose потому что нумерация j = m*N + n

    # ====================================================================
    # Граничные условия скважин (wells_coords — обычно пустой список)
    # ====================================================================
    def sort_func(well_coord_couple):
        return well_coord_couple[1] * N + well_coord_couple[0]

    wells_frac_coords.sort(key=sort_func)
    wells_coords.sort(key=sort_func)

    for coord_couple in reversed(wells_coords):
        col_idx = coord_couple[1] * N + coord_couple[0]
        A_well_column = A_full.getcol(col_idx).toarray()
        for cell_number in range(len(A_well_column)):
            if A_well_column[cell_number] != 0:
                B[cell_number][0] -= A_well_column[cell_number] * CP_dict[coord_couple]

    # ====================================================================
    # Решение СЛАУ
    # ====================================================================
    P_new = spsolve(A_full, B)

    # Преобразование обратно в 2D массив (N, M)
    Pres_end = P_new.reshape(M, N).T  # из глобальной нумерации (m*N+n) в [n, m]

    return Pres_end, np.zeros(1), B
