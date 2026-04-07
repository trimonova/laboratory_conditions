# Pore Pressure Simulation

Numerical simulation of pore pressure distribution during fluid injection into laboratory rock samples. Two-phase flow (oil/water) in porous media with fracture propagation, using finite difference methods in cylindrical coordinates.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync              # install dependencies
uv sync --all-extras # install with dev tools (pytest, ruff)
```

## Usage

### Run simulation

```bash
uv run pore-pressure run --config configs/default.yaml --no-plots
```

With interactive plots:

```bash
uv run pore-pressure run --config configs/default.yaml
```

Custom output directory:

```bash
uv run pore-pressure run --config configs/default.yaml --output results/run_001
```

### Compare with experiment

```bash
uv run pore-pressure compare --config configs/default.yaml --results output/
```

### From Python / IDE

```python
from pore_pressure.config import load_config
from pore_pressure.simulation import run_simulation, save_results

config = load_config("configs/default.yaml")
result = run_simulation(config, show_plots=True)
save_results(result, "output/")
```

## Configuration

All simulation parameters are in `configs/default.yaml`:

| Section | Key parameters |
|---------|---------------|
| `rock` | permeability, porosity, compressibility |
| `fluid` | water/oil viscosity, compressibility |
| `geometry` | sample radius, well radius |
| `fracture` | angles, lengths, oval width |
| `grid` | radial/angular step sizes |
| `time` | time step, number of steps |
| `injection` | flow rate, seed area |

## Project Structure

```
├── pyproject.toml          # Project metadata, dependencies
├── configs/
│   └── default.yaml        # Simulation parameters
├── data/
│   └── experiment/
│       └── data2.mat       # Lab pressure measurements
├── src/pore_pressure/
│   ├── config.py           # YAML -> SimulationConfig dataclass
│   ├── grid.py             # Coordinate grid construction
│   ├── solver.py           # Sparse matrix pressure solver
│   ├── velocity.py         # Velocity field (Darcy's law)
│   ├── viscosity.py        # Effective viscosity from oil/water fractions
│   ├── level_set.py        # Level-set advection (upwind scheme)
│   ├── boundary.py         # Displacement front tracking
│   ├── reconstruction.py   # Level-set reinitialization
│   ├── fracture.py         # Initial fracture geometry
│   ├── geometry.py         # Cell area computation (Shapely)
│   ├── simulation.py       # Main time-stepping loop
│   ├── plotting.py         # Visualization
│   ├── experiment_data.py  # MAT file reader
│   ├── comparison.py       # Experiment vs simulation comparison
│   └── cli.py              # Command-line interface
├── tests/
│   └── test_config.py
└── output/                 # Simulation results (gitignored)
```

## Physical Model

- Two-phase displacement (oil/water) in porous media
- Implicit pressure solver (backward Euler, sparse matrix, `scipy.sparse.linalg.spsolve`)
- Darcy flow with variable viscosity
- Level-set method for displacement front tracking
- Upwind advection scheme for numerical stability
- Non-uniform radial grid (~414 nodes), uniform angular grid (360 nodes, 1 deg)

## Output

Results are saved as `.npy` files:

| File | Content |
|------|---------|
| `Pres_distrib_in_Time_2.npy` | Pressure field over time |
| `viscosity_in_Time.npy` | Viscosity field |
| `Func_matrix_in_Time.npy` | Level-set function |
| `velocity_in_Time.npy` | Velocity field |
| `bound_coords_*.npy` | Displacement front coordinates |

## Development

```bash
uv run pytest          # run tests
uv run ruff check src/ # lint
uv run ruff format src/ # format
```
