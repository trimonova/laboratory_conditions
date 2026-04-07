from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from pore_pressure.grid import (
    build_coord_matrices,
    build_xy_matrices,
    find_fracture_cell_indices,
)


@dataclass
class SimulationConfig:
    # --- Raw parameters (from YAML) ---
    # Rock
    permeability: float
    porosity: float
    rock_compressibility: float  # Cr

    # Fluid
    water_viscosity: float  # mu_water
    oil_viscosity: float  # mu_oil
    fluid_compressibility: float  # Cf
    total_compressibility_multiplier: float  # the "25" in (Cf+Cr)*25

    # Geometry
    sample_radius: float  # R
    well_radius: float  # r_well

    # Fracture
    frac_angle_1_deg: float
    frac_angle_2_deg: float
    frac_length_1: float
    frac_length_2: float
    oval_width: float

    # Grid
    delta_r: float
    delta_r_fine: float
    R_for_fine: float
    delta_fi_deg: float

    # Time
    delta_t: float
    num_steps: int  # T_exp
    T_exp_dir: float

    # Injection
    flow_rate: float  # Q_center
    seed_area: float  # s = 0.01*0.01

    # Initial pressure
    initial_pressure: float
    initial_pressure_file: str | None

    # Output
    output_directory: str
    show_plots: bool

    # Experiment data
    experiment_data_file: str

    # --- Derived values (computed) ---
    frac_angle: float = field(init=False)
    frac_angle_2: float = field(init=False)
    C_total: float = field(init=False)
    k_water: float = field(init=False)
    k_oil: float = field(init=False)
    c3_oil: float = field(init=False)
    c3_water: float = field(init=False)
    q: float = field(init=False)
    delta_fi: float = field(init=False)

    delta_r_list: list[float] = field(init=False, repr=False)
    delta_fi_list: list[float] = field(init=False, repr=False)
    N_r_full: int = field(init=False)
    M_fi_full: int = field(init=False)

    coord_matrix_rad: list = field(init=False, repr=False)
    coord_matrix_cart: list = field(init=False, repr=False)
    coord_matrix_rad_cell: list = field(init=False, repr=False)
    coord_matrix_cart_cell: list = field(init=False, repr=False)
    X_matrix: np.ndarray = field(init=False, repr=False)
    Y_matrix: np.ndarray = field(init=False, repr=False)
    X_matrix_cell: np.ndarray = field(init=False, repr=False)
    Y_matrix_cell: np.ndarray = field(init=False, repr=False)

    M_1: int = field(init=False)
    M_2: int = field(init=False)
    N_1: int = field(init=False)
    N_2: int = field(init=False)
    frac_coords: list[tuple[int, int]] = field(init=False, repr=False)
    CP_dict: dict = field(init=False, repr=False)
    wells_frac_coords: list = field(init=False, repr=False)
    N_sources: int = field(init=False)

    Pres_distrib: np.ndarray = field(init=False, repr=False)
    Courant_number_oil: float = field(init=False)
    Courant_number_water: float = field(init=False)

    def __post_init__(self) -> None:
        # Angles in radians
        self.frac_angle = np.radians(self.frac_angle_1_deg)
        self.frac_angle_2 = np.radians(self.frac_angle_2_deg)
        self.delta_fi = np.radians(self.delta_fi_deg)

        # Total compressibility and diffusion coefficients
        self.C_total = (self.fluid_compressibility + self.rock_compressibility) * self.total_compressibility_multiplier
        self.k_water = self.water_viscosity * self.porosity * self.C_total / self.permeability
        self.k_oil = self.oil_viscosity * self.porosity * self.C_total / self.permeability

        # Time-stepping coefficients
        self.c3_oil = self.k_oil / self.delta_t
        self.c3_water = self.k_water / self.delta_t

        # Specific flow rate
        self.q = self.flow_rate / self.seed_area / 4

        # Build grid lists
        N_r_fine = round(self.R_for_fine / self.delta_r_fine)
        self.delta_r_list = [self.delta_r_fine] * N_r_fine + [self.delta_r] * round(
            (self.sample_radius - self.well_radius - self.R_for_fine) / self.delta_r
        )
        self.N_r_full = len(self.delta_r_list)

        self.delta_fi_list = [self.delta_fi] * round(2 * np.pi / self.delta_fi)
        self.M_fi_full = len(self.delta_fi_list)

        # Coordinate matrices
        self.coord_matrix_rad, self.coord_matrix_cart = build_coord_matrices(
            self.N_r_full, self.M_fi_full, self.delta_r, self.delta_fi, self.well_radius, cell_center=False
        )
        self.coord_matrix_rad_cell, self.coord_matrix_cart_cell = build_coord_matrices(
            self.N_r_full, self.M_fi_full, self.delta_r, self.delta_fi, self.well_radius, cell_center=True
        )

        self.X_matrix, self.Y_matrix = build_xy_matrices(
            self.N_r_full, self.M_fi_full, self.delta_r, self.delta_fi, self.well_radius, cell_center=False
        )
        self.X_matrix_cell, self.Y_matrix_cell = build_xy_matrices(
            self.N_r_full, self.M_fi_full, self.delta_r, self.delta_fi, self.well_radius, cell_center=True
        )

        # Fracture cell indices
        self.M_1, self.M_2 = find_fracture_cell_indices(self.delta_fi_list, self.frac_angle, self.frac_angle_2)
        self.N_1, self.N_2 = _find_radial_fracture_indices(self.delta_r_list, self.frac_length_1, self.frac_length_2)
        self.N_sources = self.N_1 + self.N_2

        # Fracture coordinates and boundary conditions
        self.frac_coords = []
        for i in range(self.N_1):
            self.frac_coords.append((copy.copy(i), self.M_1))
        for i in range(self.N_2):
            self.frac_coords.append((copy.copy(i), self.M_2))

        self.CP_dict = {}
        for elem in self.frac_coords:
            self.CP_dict[elem] = 1

        self.wells_frac_coords = list(self.frac_coords)

        # Initial pressure distribution
        if self.initial_pressure_file:
            data = np.load(self.initial_pressure_file)
            self.Pres_distrib = data[-1] if data.ndim == 3 else data
        else:
            self.Pres_distrib = np.ones((self.N_r_full, self.M_fi_full)) * self.initial_pressure

        # Courant numbers (stability check)
        self.Courant_number_oil = (
            self.delta_t / self.k_oil / self.delta_fi**2 + self.delta_t / self.k_oil / self.delta_r_fine**2
        ) / 25
        self.Courant_number_water = (
            self.delta_t / self.k_water / self.delta_fi**2 + self.delta_t / self.k_water / self.delta_r_fine**2
        ) / 25


def _find_radial_fracture_indices(
    delta_r_list: list[float], frac_length_1: float, frac_length_2: float
) -> tuple[int, int]:
    """Find grid indices corresponding to fracture lengths along radial direction."""
    R_i = 0.0
    N_1 = 0
    N_2 = 0
    for i in range(len(delta_r_list) - 1):
        R_i += delta_r_list[i]
        if R_i <= frac_length_1 <= (R_i + delta_r_list[i + 1]):
            if abs(frac_length_1 - R_i) < abs(frac_length_1 - (R_i + delta_r_list[i + 1])):
                N_1 = copy.copy(i)
            else:
                N_1 = copy.copy(i + 1)
        if R_i <= frac_length_2 <= (R_i + delta_r_list[i + 1]):
            if abs(frac_length_2 - R_i) < abs(frac_length_2 - (R_i + delta_r_list[i + 1])):
                N_2 = copy.copy(i)
            else:
                N_2 = copy.copy(i + 1)

    if N_1 == 0 or N_2 == 0:
        raise ValueError("Could not find N_1 or N_2 fracture radial indices")

    return N_1, N_2


def load_config(path: str | Path) -> SimulationConfig:
    """Load simulation configuration from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    return SimulationConfig(
        # Rock
        permeability=raw["rock"]["permeability"],
        porosity=raw["rock"]["porosity"],
        rock_compressibility=raw["rock"]["compressibility"],
        # Fluid
        water_viscosity=raw["fluid"]["water_viscosity"],
        oil_viscosity=raw["fluid"]["oil_viscosity"],
        fluid_compressibility=raw["fluid"]["compressibility"],
        total_compressibility_multiplier=raw["fluid"]["total_compressibility_multiplier"],
        # Geometry
        sample_radius=raw["geometry"]["sample_radius"],
        well_radius=raw["geometry"]["well_radius"],
        # Fracture
        frac_angle_1_deg=raw["fracture"]["angle_1_deg"],
        frac_angle_2_deg=raw["fracture"]["angle_2_deg"],
        frac_length_1=raw["fracture"]["length_1"],
        frac_length_2=raw["fracture"]["length_2"],
        oval_width=raw["fracture"]["oval_width"],
        # Grid
        delta_r=raw["grid"]["delta_r"],
        delta_r_fine=raw["grid"]["delta_r_fine"],
        R_for_fine=raw["grid"]["R_for_fine"],
        delta_fi_deg=raw["grid"]["delta_fi_deg"],
        # Time
        delta_t=raw["time"]["delta_t"],
        num_steps=raw["time"]["num_steps"],
        T_exp_dir=raw["time"]["T_exp_dir"],
        # Injection
        flow_rate=raw["injection"]["flow_rate"],
        seed_area=raw["injection"]["seed_area"],
        # Initial pressure
        initial_pressure=raw.get("initial_pressure", 0.0),
        initial_pressure_file=raw.get("initial_pressure_file"),
        # Output
        output_directory=raw.get("output", {}).get("directory", "output"),
        show_plots=raw.get("output", {}).get("show_plots", True),
        # Experiment data
        experiment_data_file=raw.get("experiment_data_file", "data/experiment/data2.mat"),
    )
