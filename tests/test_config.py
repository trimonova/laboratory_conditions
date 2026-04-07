import numpy as np

from pore_pressure.config import load_config


def test_load_config_basic_values():
    config = load_config("configs/default.yaml")

    # Grid dimensions (from original input_parameters.py)
    assert config.N_r_full == 414
    assert config.M_fi_full == 360

    # Fracture indices
    assert config.M_1 == 45
    assert config.M_2 == 225

    # Derived physical values
    perm = 2e-15
    mu_water = 2e-3
    mu_oil = 0.2
    porosity = 0.4
    Cf = 1e-9
    Cr = 5e-10
    C_total = (Cf + Cr) * 25

    assert np.isclose(config.C_total, C_total)
    assert np.isclose(config.k_water, mu_water * porosity * C_total / perm)
    assert np.isclose(config.k_oil, mu_oil * porosity * C_total / perm)

    # Flow rate
    Q_center = 2e-6
    s = 1e-4
    assert np.isclose(config.q, Q_center / s / 4)

    # Grid lists
    assert len(config.delta_r_list) == 414
    assert len(config.delta_fi_list) == 360

    # Coordinate matrices shape
    assert len(config.coord_matrix_rad) == 414
    assert len(config.coord_matrix_rad[0]) == 360

    # XY matrices shape
    assert config.X_matrix.shape == (414, 360)
    assert config.Y_matrix.shape == (414, 360)

    # Initial pressure
    assert config.Pres_distrib.shape == (414, 360)
    assert np.all(config.Pres_distrib == 0.0)


def test_fracture_coords():
    config = load_config("configs/default.yaml")

    # N_1 and N_2 should be > 0
    assert config.N_1 > 0
    assert config.N_2 > 0

    # frac_coords should have N_1 + N_2 elements
    assert len(config.frac_coords) == config.N_1 + config.N_2

    # All frac_coords should be in CP_dict
    for coord in config.frac_coords:
        assert coord in config.CP_dict
        assert config.CP_dict[coord] == 1
