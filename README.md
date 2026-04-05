# Laboratory Conditions: Pore Pressure Modeling

Numerical simulation of pore pressure distribution during fluid injection into laboratory rock samples. The project models two-phase flow (oil/water) in porous media with fracture propagation using finite difference methods in cylindrical coordinates.

## Injection Configurations

- **QinCenter** — center point injection through a single wellbore
- **QinPackers** — injection through packer-style arrangements (multi-point injection)

## Physical Model

- Two-phase fluid displacement (oil and water) in heterogeneous porous media
- Pressure diffusion solved implicitly (backward Euler) on a non-uniform radial grid
- Darcy flow with variable viscosity depending on local fluid phase
- Dynamic tracking of the displacement front boundary
- Fracture geometry specification and propagation

## Project Structure

```
laboratory_conditions/
├── pore_pressure_during_injection_QinCenter/
│   ├── QinCenter/                    # Core calculation modules
│   ├── QinCenter_attempt_2_Sav/      # Alternative implementation variant
│   ├── experiment_data/              # Lab measurements and comparison scripts
│   ├── result_folder*/               # Simulation output variants
│   ├── find_pore_pressure_during_injection_QinCenter.py  # Main entry point
│   └── input_parameters.py           # Configuration
│
├── pore_pressure_during_injection_QinPackers/
│   ├── QinPackers/                   # Core calculation modules
│   ├── QinPackers_corr_Sav/          # Corrected variant
│   ├── experiment_data/              # Lab measurements and comparison scripts
│   ├── result_folder/                # Simulation output
│   ├── find_pore_pressure_during_injection_QinPackers.py  # Main entry point
│   └── input_parameters.py           # Configuration
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `find_pore_pressure_*.py` | Sparse matrix pressure solver (scipy.sparse.linalg.spsolve) |
| `find_viscosity.py` | Effective viscosity calculation based on oil/water fraction |
| `find_bound_coords.py` | Displacement front boundary tracking |
| `find_func_matrix_remake.py` | Phase distribution matrix update after displacement |
| `newFuncMatrix_fix_5.py` | Velocity field from pressure gradients (Darcy's law) |
| `find_area.py` | Oil area fraction via polygon geometry (Shapely) |
| `start_to_do_replacement.py` | Fracture geometry and boundary condition initialization |
| `plot_results.py` | Contour plots of pressure, velocity, and phase distribution |

## Dependencies

- Python 3
- NumPy
- SciPy
- Matplotlib
- Shapely
- h5py

Install with:

```bash
pip install numpy scipy matplotlib shapely h5py
```

## Usage

### 1. Configure parameters

Edit `input_parameters.py` in the corresponding directory. Key parameters:

| Parameter | Default (QinCenter) | Description |
|-----------|---------------------|-------------|
| `perm` | 2×10⁻¹⁵ m² | Rock permeability |
| `porosity` | 0.4 | Porosity |
| `mu_oil` | 0.2 Pa·s | Oil viscosity |
| `mu_water` | 0.002 Pa·s | Water viscosity |
| `Q_center` | 0.2×10⁻⁶ m³/s | Injection flow rate |
| `R` | 0.215 m | Sample radius |
| `r_well` | 0.008 m | Well radius |

### 2. Run simulation

```bash
cd pore_pressure_during_injection_QinCenter
python find_pore_pressure_during_injection_QinCenter.py
```

The simulation iterates through time steps, saving results as `.npy` files and displaying plots.

### 3. Visualize results

```bash
cd result_folder
python plot_results.py
```

## Output

Each run produces the following arrays in `result_folder/`:

| File | Content |
|------|---------|
| `Pres_distrib_in_Time_2.npy` | Pressure distribution (N_r × M_fi × T) |
| `viscosity_in_Time.npy` | Effective viscosity field |
| `Func_matrix_in_Time.npy` | Phase distribution (1=oil, −1=water) |
| `velocity_in_Time.npy` | Velocity magnitude field |
| `bound_coords_cart_in_Time.npy` | Displacement front (cartesian) |
| `bound_coords_rad_in_Time.npy` | Displacement front (polar) |

## Experimental Validation

The `experiment_data/` directories contain laboratory measurements (`data2.mat`) and comparison scripts (`comparison_exp_simulat.py`) for validating simulation results against pressure sensor data from real injection experiments.

## Grid

- **Radial**: Non-uniform, ~425 nodes (finer near the well)
- **Angular**: Uniform, 360 nodes (1° spacing)
- **Time step**: 0.5 s
