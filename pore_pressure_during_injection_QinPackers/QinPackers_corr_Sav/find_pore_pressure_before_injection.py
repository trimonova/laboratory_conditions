import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Решение уравнения пьезопроводности в цилиндрических координатах
# Однофазная фильтрация, граничные условия Дирихле на скважинах
# Оптимизированная версия (COO-сборка, numpy-векторизация)

def PorePressure_before_injection(N_r_full, M_fi_full, Pres_distrib,
                                   CP_dict, wells_coords,
                                   delta_r_list, delta_fi_list, mu, porosity,
                                   C_total, perm, delta_t, r_well):
    N = N_r_full
    M = M_fi_full

    # c3 вычисляется здесь: phi * C_total / dt
    # Пространственные члены уже содержат perm/mu, поэтому c3 НЕ содержит mu/perm
    c3_fluid = porosity * C_total / delta_t

    # ====================================================================
    # Геометрия сетки
    # ====================================================================
    dr = np.array(delta_r_list)
    dfi = np.array(delta_fi_list)
    r_i = np.cumsum(dr) - dr / 2 + r_well  # центры ячеек
    r_i_minus = r_i - dr / 2
    r_i_plus = r_i + dr / 2

    # ====================================================================
    # Коэффициенты (постоянная вязкость)
    # ====================================================================
    # Радиальные
    coef_right = np.zeros((N, M))
    coef_right[:-1, :] = perm / mu * r_i_plus[:-1, None] / r_i[:-1, None] / dr[:-1, None]**2

    coef_left = np.zeros((N, M))
    coef_left[1:, :] = perm / mu * r_i_minus[1:, None] / r_i[1:, None] / dr[1:, None]**2

    # Угловые
    sym_right = perm / mu / (r_i[:, None] * dfi[None, :])**2
    sym_left = perm / mu / (r_i[:, None] * dfi[None, :])**2

    # ====================================================================
    # Граничные условия: n=0 (стенка скважины) — нулевой поток
    # ====================================================================
    coef_left[0, :] = 0

    # ====================================================================
    # Граничные условия: n=N-1 (внешняя граница) — нулевой поток
    # ====================================================================
    coef_right[-1, :] = 0

    # ====================================================================
    # Диагональ
    # ====================================================================
    coef_diag = -(coef_right + coef_left + sym_right + sym_left + c3_fluid)

    # ====================================================================
    # Сборка COO
    # ====================================================================
    n_idx = np.arange(N)
    m_idx = np.arange(M)
    nn, mm = np.meshgrid(n_idx, m_idx, indexing='ij')
    global_idx = mm * N + nn

    rows_list = []
    cols_list = []
    vals_list = []

    # Диагональ
    rows_list.append(global_idx.ravel())
    cols_list.append(global_idx.ravel())
    vals_list.append(coef_diag.ravel())

    # coef_left: (n,m) -> (n-1,m)
    mask_left = n_idx > 0
    gl_from = global_idx[mask_left, :]
    gl_to = global_idx[mask_left, :] - 1
    vals_l = coef_left[mask_left, :]
    nz = vals_l != 0
    rows_list.append(gl_from[nz]); cols_list.append(gl_to[nz]); vals_list.append(vals_l[nz])

    # coef_right: (n,m) -> (n+1,m)
    mask_right = n_idx < N - 1
    gl_from = global_idx[mask_right, :]
    gl_to = global_idx[mask_right, :] + 1
    vals_r = coef_right[mask_right, :]
    nz = vals_r != 0
    rows_list.append(gl_from[nz]); cols_list.append(gl_to[nz]); vals_list.append(vals_r[nz])

    # sym_right: (n,m) -> (n, m+1)
    m_next = (mm + 1) % M
    gl_to_right = m_next * N + nn
    nz = sym_right != 0
    rows_list.append(global_idx[nz]); cols_list.append(gl_to_right[nz]); vals_list.append(sym_right[nz])

    # sym_left: (n,m) -> (n, m-1)
    m_prev = (mm - 1) % M
    gl_to_left = m_prev * N + nn
    nz = sym_left != 0
    rows_list.append(global_idx[nz]); cols_list.append(gl_to_left[nz]); vals_list.append(sym_left[nz])

    total_size = N * M
    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_vals = np.concatenate(vals_list)
    A_full = csr_matrix((all_vals, (all_rows, all_cols)), shape=(total_size, total_size))

    # ====================================================================
    # Правая часть: B = -c3_fluid * P_old
    # ====================================================================
    B_2d = -c3_fluid * Pres_distrib
    B = B_2d.T.reshape(-1, 1)

    # ====================================================================
    # Граничные условия Дирихле (скважины с заданным давлением)
    # Удаляем строки/столбцы скважин, переносим в RHS
    # ====================================================================
    well_indices = []
    for coord_couple in wells_coords:
        col_idx = coord_couple[1] * N + coord_couple[0]
        well_indices.append((col_idx, CP_dict[coord_couple]))

    # Сортируем по убыванию индекса для корректного удаления
    well_indices.sort(key=lambda x: x[0], reverse=True)

    for col_idx, P_well in well_indices:
        # Переносим вклад скважины в RHS
        A_col = A_full.getcol(col_idx).toarray().ravel()
        for i in range(len(A_col)):
            if A_col[i] != 0:
                B[i, 0] -= A_col[i] * P_well

    # Удаляем строки и столбцы скважин
    all_indices = np.arange(total_size)
    well_idx_list = sorted([wi[0] for wi in well_indices])
    keep = np.ones(total_size, dtype=bool)
    keep[well_idx_list] = False

    A_reduced = A_full[keep][:, keep]
    B_reduced = B[keep]

    # Решение
    P_reduced = spsolve(A_reduced, B_reduced)

    # Вставка давлений скважин обратно
    P_full = np.zeros(total_size)
    P_full[keep] = P_reduced
    for col_idx, P_well in well_indices:
        P_full[col_idx] = P_well

    # Преобразование в 2D
    Pres_end = P_full.reshape(M, N).T

    return Pres_end
