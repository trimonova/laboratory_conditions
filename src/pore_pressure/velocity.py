"""Velocity field computation using Darcy's law with boundary conditions at fractures."""

from __future__ import annotations

import numpy as np


def find_velocity_grad_fi(
    Pres_distrib: np.ndarray,
    viscosity_distrib: np.ndarray,
    delta_r: float,
    delta_fi: float,
    perm: float,
    r_well: float,
    fi_distrib: np.ndarray,
    q_well: float,
    N_1: int,
    N_2: int,
    M_1: int,
    M_2: int,
    func_list_0: list[float],
    func_list_end: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute velocity and level-set gradient fields at cell faces.

    Returns (velocity_distrib, grad_fi_distrib, viscosity_distrib_total),
    each of shape (4, N_r_full, M_fi_full) for [left, right, bot, up] directions.
    """
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

    for m in range(M_fi_full):
        for n in range(N_r_full):
            r = r_well + n * delta_r + delta_r / 2

            if n == 0 and m == M_1 - 1:
                _handle_n0_frac_left(
                    n,
                    m,
                    Pres_distrib,
                    viscosity_distrib,
                    fi_distrib,
                    func_list_0,
                    perm,
                    delta_r,
                    delta_fi,
                    r,
                    q_well,
                    velocity_distrib_left,
                    velocity_distrib_right,
                    velocity_distrib_bot,
                    velocity_distrib_up,
                    grad_fi_distrib_left,
                    grad_fi_distrib_right,
                    grad_fi_distrib_bot,
                    grad_fi_distrib_up,
                )

            elif n == 0 and m == M_2 - 1:
                _handle_n0_frac_left(
                    n,
                    m,
                    Pres_distrib,
                    viscosity_distrib,
                    fi_distrib,
                    func_list_0,
                    perm,
                    delta_r,
                    delta_fi,
                    r,
                    q_well,
                    velocity_distrib_left,
                    velocity_distrib_right,
                    velocity_distrib_bot,
                    velocity_distrib_up,
                    grad_fi_distrib_left,
                    grad_fi_distrib_right,
                    grad_fi_distrib_bot,
                    grad_fi_distrib_up,
                )

            elif n == 0 and m == M_1:
                _handle_n0_frac_right(
                    n,
                    m,
                    Pres_distrib,
                    viscosity_distrib,
                    fi_distrib,
                    func_list_0,
                    perm,
                    delta_r,
                    delta_fi,
                    r,
                    q_well,
                    velocity_distrib_left,
                    velocity_distrib_right,
                    velocity_distrib_bot,
                    velocity_distrib_up,
                    grad_fi_distrib_left,
                    grad_fi_distrib_right,
                    grad_fi_distrib_bot,
                    grad_fi_distrib_up,
                )

            elif n == 0 and m == M_2:
                _handle_n0_frac_right(
                    n,
                    m,
                    Pres_distrib,
                    viscosity_distrib,
                    fi_distrib,
                    func_list_0,
                    perm,
                    delta_r,
                    delta_fi,
                    r,
                    q_well,
                    velocity_distrib_left,
                    velocity_distrib_right,
                    velocity_distrib_bot,
                    velocity_distrib_up,
                    grad_fi_distrib_left,
                    grad_fi_distrib_right,
                    grad_fi_distrib_bot,
                    grad_fi_distrib_up,
                )

            elif m == M_1 and n <= N_1:
                _handle_frac_right(
                    n,
                    m,
                    Pres_distrib,
                    viscosity_distrib,
                    fi_distrib,
                    perm,
                    delta_r,
                    delta_fi,
                    r,
                    q_well,
                    velocity_distrib_left,
                    velocity_distrib_right,
                    velocity_distrib_bot,
                    velocity_distrib_up,
                    grad_fi_distrib_left,
                    grad_fi_distrib_right,
                    grad_fi_distrib_bot,
                    grad_fi_distrib_up,
                )

            elif m == M_2 and n <= N_2:
                _handle_frac_right(
                    n,
                    m,
                    Pres_distrib,
                    viscosity_distrib,
                    fi_distrib,
                    perm,
                    delta_r,
                    delta_fi,
                    r,
                    q_well,
                    velocity_distrib_left,
                    velocity_distrib_right,
                    velocity_distrib_bot,
                    velocity_distrib_up,
                    grad_fi_distrib_left,
                    grad_fi_distrib_right,
                    grad_fi_distrib_bot,
                    grad_fi_distrib_up,
                )

            elif m == M_1 - 1 and n <= N_1:
                _handle_frac_left(
                    n,
                    m,
                    Pres_distrib,
                    viscosity_distrib,
                    fi_distrib,
                    perm,
                    delta_r,
                    delta_fi,
                    r,
                    q_well,
                    velocity_distrib_left,
                    velocity_distrib_right,
                    velocity_distrib_bot,
                    velocity_distrib_up,
                    grad_fi_distrib_left,
                    grad_fi_distrib_right,
                    grad_fi_distrib_bot,
                    grad_fi_distrib_up,
                )

            elif m == M_2 - 1 and n <= N_2:
                _handle_frac_left(
                    n,
                    m,
                    Pres_distrib,
                    viscosity_distrib,
                    fi_distrib,
                    perm,
                    delta_r,
                    delta_fi,
                    r,
                    q_well,
                    velocity_distrib_left,
                    velocity_distrib_right,
                    velocity_distrib_bot,
                    velocity_distrib_up,
                    grad_fi_distrib_left,
                    grad_fi_distrib_right,
                    grad_fi_distrib_bot,
                    grad_fi_distrib_up,
                )

            elif n == 0 and m == M_fi_full - 1:
                visc_mid_right = (viscosity_distrib[n][m] + viscosity_distrib[n + 1][m]) / 2
                velocity_distrib_left[n][m] = 0
                velocity_distrib_right[n][m] = (
                    -perm / visc_mid_right * (Pres_distrib[n + 1][m] - Pres_distrib[n][m]) / delta_r
                )
                grad_fi_distrib_left[n][m] = (fi_distrib[n][m] - func_list_0[m]) / delta_r
                grad_fi_distrib_right[n][m] = (fi_distrib[n + 1][m] - fi_distrib[n][m]) / delta_r
                visc_mid_bot = (viscosity_distrib[n][m] + viscosity_distrib[n][m - 1]) / 2
                visc_mid_up = (viscosity_distrib[n][m] + viscosity_distrib[n][0]) / 2
                velocity_distrib_up[n][m] = (
                    -perm / visc_mid_up * (Pres_distrib[n][0] - Pres_distrib[n][m]) / delta_fi / r
                )
                grad_fi_distrib_up[n][m] = (fi_distrib[n][0] - fi_distrib[n][m]) / delta_fi / r
                velocity_distrib_bot[n][m] = (
                    perm / visc_mid_bot * (Pres_distrib[n][m - 1] - Pres_distrib[n][m]) / delta_fi / r
                )
                grad_fi_distrib_bot[n][m] = (fi_distrib[n][m] - fi_distrib[n][m - 1]) / delta_fi / r

            elif n == 0:
                visc_mid_right = (viscosity_distrib[n][m] + viscosity_distrib[n + 1][m]) / 2
                velocity_distrib_left[n][m] = 0
                velocity_distrib_right[n][m] = (
                    -perm / visc_mid_right * (Pres_distrib[n + 1][m] - Pres_distrib[n][m]) / delta_r
                )
                grad_fi_distrib_left[n][m] = (fi_distrib[n][m] - func_list_0[m]) / delta_r
                grad_fi_distrib_right[n][m] = (fi_distrib[n + 1][m] - fi_distrib[n][m]) / delta_r
                visc_mid_bot = (viscosity_distrib[n][m] + viscosity_distrib[n][m - 1]) / 2
                visc_mid_up = (viscosity_distrib[n][m + 1] + viscosity_distrib[n][m]) / 2
                velocity_distrib_up[n][m] = (
                    -perm / visc_mid_up * (Pres_distrib[n][m + 1] - Pres_distrib[n][m]) / delta_fi / r
                )
                grad_fi_distrib_up[n][m] = (fi_distrib[n][m + 1] - fi_distrib[n][m]) / delta_fi / r
                velocity_distrib_bot[n][m] = (
                    perm / visc_mid_bot * (Pres_distrib[n][m - 1] - Pres_distrib[n][m]) / delta_fi / r
                )
                grad_fi_distrib_bot[n][m] = (fi_distrib[n][m] - fi_distrib[n][m - 1]) / delta_fi / r

            elif n == N_r_full - 1 and m == M_fi_full - 1:
                visc_mid_left = (viscosity_distrib[n][m] + viscosity_distrib[n - 1][m]) / 2
                velocity_distrib_right[n][m] = 0
                velocity_distrib_left[n][m] = (
                    perm / visc_mid_left * (Pres_distrib[n - 1][m] - Pres_distrib[n][m]) / delta_r
                )
                grad_fi_distrib_right[n][m] = (func_list_end[m] - fi_distrib[n][m]) / delta_r
                grad_fi_distrib_left[n][m] = (fi_distrib[n][m] - fi_distrib[n - 1][m]) / delta_r
                visc_mid_bot = (viscosity_distrib[n][m] + viscosity_distrib[n][m - 1]) / 2
                visc_mid_up = (viscosity_distrib[n][m] + viscosity_distrib[n][0]) / 2
                velocity_distrib_up[n][m] = (
                    -perm / visc_mid_up * (Pres_distrib[n][0] - Pres_distrib[n][m]) / delta_fi / r
                )
                grad_fi_distrib_up[n][m] = (fi_distrib[n][0] - fi_distrib[n][m]) / delta_fi / r
                velocity_distrib_bot[n][m] = (
                    perm / visc_mid_bot * (Pres_distrib[n][m - 1] - Pres_distrib[n][m]) / delta_fi / r
                )
                grad_fi_distrib_bot[n][m] = (fi_distrib[n][m] - fi_distrib[n][m - 1]) / delta_fi / r

            elif n == N_r_full - 1:
                visc_mid_left = (viscosity_distrib[n][m] + viscosity_distrib[n - 1][m]) / 2
                velocity_distrib_right[n][m] = 0
                velocity_distrib_left[n][m] = (
                    perm / visc_mid_left * (Pres_distrib[n - 1][m] - Pres_distrib[n][m]) / delta_r
                )
                grad_fi_distrib_right[n][m] = (func_list_end[m] - fi_distrib[n][m]) / delta_r
                grad_fi_distrib_left[n][m] = (fi_distrib[n][m] - fi_distrib[n - 1][m]) / delta_r
                visc_mid_bot = (viscosity_distrib[n][m] + viscosity_distrib[n][m - 1]) / 2
                visc_mid_up = (viscosity_distrib[n][m + 1] + viscosity_distrib[n][m]) / 2
                velocity_distrib_up[n][m] = (
                    -perm / visc_mid_up * (Pres_distrib[n][m + 1] - Pres_distrib[n][m]) / delta_fi / r
                )
                grad_fi_distrib_up[n][m] = (fi_distrib[n][m + 1] - fi_distrib[n][m]) / delta_fi / r
                velocity_distrib_bot[n][m] = (
                    perm / visc_mid_bot * (Pres_distrib[n][m - 1] - Pres_distrib[n][m]) / delta_fi / r
                )
                grad_fi_distrib_bot[n][m] = (fi_distrib[n][m] - fi_distrib[n][m - 1]) / delta_fi / r

            elif m == M_fi_full - 1:
                visc_mid_left = (viscosity_distrib[n][m] + viscosity_distrib[n - 1][m]) / 2
                visc_mid_right = (viscosity_distrib[n][m] + viscosity_distrib[n + 1][m]) / 2
                velocity_distrib_right[n][m] = (
                    -perm / visc_mid_right * (Pres_distrib[n + 1][m] - Pres_distrib[n][m]) / delta_r
                )
                velocity_distrib_left[n][m] = (
                    perm / visc_mid_left * (Pres_distrib[n - 1][m] - Pres_distrib[n][m]) / delta_r
                )
                grad_fi_distrib_right[n][m] = (fi_distrib[n + 1][m] - fi_distrib[n][m]) / delta_r
                grad_fi_distrib_left[n][m] = (fi_distrib[n][m] - fi_distrib[n - 1][m]) / delta_r
                visc_mid_bot = (viscosity_distrib[n][m] + viscosity_distrib[n][m - 1]) / 2
                visc_mid_up = (viscosity_distrib[n][0] + viscosity_distrib[n][m]) / 2
                velocity_distrib_up[n][m] = (
                    -perm / visc_mid_up * (Pres_distrib[n][0] - Pres_distrib[n][m]) / delta_fi / r
                )
                grad_fi_distrib_up[n][m] = (fi_distrib[n][0] - fi_distrib[n][m]) / delta_fi / r
                velocity_distrib_bot[n][m] = (
                    perm / visc_mid_bot * (Pres_distrib[n][m - 1] - Pres_distrib[n][m]) / delta_fi / r
                )
                grad_fi_distrib_bot[n][m] = (fi_distrib[n][m] - fi_distrib[n][m - 1]) / delta_fi / r

            else:
                visc_mid_left = (viscosity_distrib[n][m] + viscosity_distrib[n - 1][m]) / 2
                visc_mid_right = (viscosity_distrib[n][m] + viscosity_distrib[n + 1][m]) / 2
                velocity_distrib_right[n][m] = (
                    -perm / visc_mid_right * (Pres_distrib[n + 1][m] - Pres_distrib[n][m]) / delta_r
                )
                velocity_distrib_left[n][m] = (
                    perm / visc_mid_left * (Pres_distrib[n - 1][m] - Pres_distrib[n][m]) / delta_r
                )
                grad_fi_distrib_right[n][m] = (fi_distrib[n + 1][m] - fi_distrib[n][m]) / delta_r
                grad_fi_distrib_left[n][m] = (fi_distrib[n][m] - fi_distrib[n - 1][m]) / delta_r
                visc_mid_bot = (viscosity_distrib[n][m] + viscosity_distrib[n][m - 1]) / 2
                visc_mid_up = (viscosity_distrib[n][m + 1] + viscosity_distrib[n][m]) / 2
                velocity_distrib_up[n][m] = (
                    -perm / visc_mid_up * (Pres_distrib[n][m + 1] - Pres_distrib[n][m]) / delta_fi / r
                )
                grad_fi_distrib_up[n][m] = (fi_distrib[n][m + 1] - fi_distrib[n][m]) / delta_fi / r
                velocity_distrib_bot[n][m] = (
                    perm / visc_mid_bot * (Pres_distrib[n][m - 1] - Pres_distrib[n][m]) / delta_fi / r
                )
                grad_fi_distrib_bot[n][m] = (fi_distrib[n][m] - fi_distrib[n][m - 1]) / delta_fi / r

    velocity_distrib = np.array(
        [velocity_distrib_left, velocity_distrib_right, velocity_distrib_bot, velocity_distrib_up]
    )
    grad_fi_distrib = np.array([grad_fi_distrib_left, grad_fi_distrib_right, grad_fi_distrib_bot, grad_fi_distrib_up])
    viscosity_distrib_total = np.array(
        [viscosity_distrib_left, viscosity_distrib_right, viscosity_distrib_bot, viscosity_distrib_up]
    )

    return velocity_distrib, grad_fi_distrib, viscosity_distrib_total


# --- Helper functions for boundary condition cases ---


def _handle_n0_frac_left(n, m, Pres, visc, fi, func_list_0, perm, dr, dfi, r, q, vl, vr, vb, vu, gl, gr, gb, gu):
    """n=0, fracture on left side (m = M_1-1 or M_2-1)."""
    visc_mid_right = (visc[n][m] + visc[n + 1][m]) / 2
    vl[n][m] = 0
    vr[n][m] = -perm / visc_mid_right * (Pres[n + 1][m] - Pres[n][m]) / dr
    gl[n][m] = (fi[n][m] - func_list_0[m]) / dr
    gr[n][m] = (fi[n + 1][m] - fi[n][m]) / dr
    visc_mid_bot = (visc[n][m] + visc[n][m - 1]) / 2
    vu[n][m] = -q
    gu[n][m] = (fi[n][m + 1] - fi[n][m]) / dfi / r
    vb[n][m] = perm / visc_mid_bot * (Pres[n][m - 1] - Pres[n][m]) / dfi / r
    gb[n][m] = (fi[n][m] - fi[n][m - 1]) / dfi / r


def _handle_n0_frac_right(n, m, Pres, visc, fi, func_list_0, perm, dr, dfi, r, q, vl, vr, vb, vu, gl, gr, gb, gu):
    """n=0, fracture on right side (m = M_1 or M_2)."""
    visc_mid_right = (visc[n][m] + visc[n + 1][m]) / 2
    vl[n][m] = 0
    vr[n][m] = -perm / visc_mid_right * (Pres[n + 1][m] - Pres[n][m]) / dr
    gl[n][m] = (fi[n][m] - func_list_0[m]) / dr
    gr[n][m] = (fi[n + 1][m] - fi[n][m]) / dr
    visc_mid_up = (visc[n][m] + visc[n][m + 1]) / 2
    vb[n][m] = q
    gu[n][m] = (fi[n][m + 1] - fi[n][m]) / dfi / r
    vu[n][m] = -perm / visc_mid_up * (Pres[n][m + 1] - Pres[n][m]) / dfi / r
    gb[n][m] = (fi[n][m] - fi[n][m - 1]) / dfi / r


def _handle_frac_right(n, m, Pres, visc, fi, perm, dr, dfi, r, q, vl, vr, vb, vu, gl, gr, gb, gu):
    """Interior fracture cells on right side (m = M_1 or M_2, n <= N_1 or N_2)."""
    visc_mid_right = (visc[n][m] + visc[n + 1][m]) / 2
    visc_mid_left = (visc[n][m] + visc[n - 1][m]) / 2
    vl[n][m] = perm / visc_mid_left * (Pres[n - 1][m] - Pres[n][m]) / dr
    vr[n][m] = -perm / visc_mid_right * (Pres[n + 1][m] - Pres[n][m]) / dr
    gl[n][m] = (fi[n][m] - fi[n - 1][m]) / dr
    gr[n][m] = (fi[n + 1][m] - fi[n][m]) / dr
    visc_mid_up = (visc[n][m] + visc[n][m + 1]) / 2
    vb[n][m] = q
    gu[n][m] = (fi[n][m + 1] - fi[n][m]) / dfi / r
    vu[n][m] = -perm / visc_mid_up * (Pres[n][m + 1] - Pres[n][m]) / dfi / r
    gb[n][m] = (fi[n][m] - fi[n][m - 1]) / dfi / r


def _handle_frac_left(n, m, Pres, visc, fi, perm, dr, dfi, r, q, vl, vr, vb, vu, gl, gr, gb, gu):
    """Interior fracture cells on left side (m = M_1-1 or M_2-1, n <= N_1 or N_2)."""
    visc_mid_right = (visc[n][m] + visc[n + 1][m]) / 2
    visc_mid_left = (visc[n][m] + visc[n - 1][m]) / 2
    vl[n][m] = perm / visc_mid_left * (Pres[n - 1][m] - Pres[n][m]) / dr
    vr[n][m] = -perm / visc_mid_right * (Pres[n + 1][m] - Pres[n][m]) / dr
    gl[n][m] = (fi[n][m] - fi[n - 1][m]) / dr
    gr[n][m] = (fi[n + 1][m] - fi[n][m]) / dr
    visc_mid_bot = (visc[n][m] + visc[n][m - 1]) / 2
    vu[n][m] = -q
    gu[n][m] = (fi[n][m + 1] - fi[n][m]) / dfi / r
    vb[n][m] = perm / visc_mid_bot * (Pres[n][m - 1] - Pres[n][m]) / dfi / r
    gb[n][m] = (fi[n][m] - fi[n][m - 1]) / dfi / r
