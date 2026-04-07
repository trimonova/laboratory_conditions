"""Command-line interface for pore pressure simulation."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pore pressure simulation during fluid injection")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run simulation")
    run_parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    run_parser.add_argument("--output", type=Path, default=None, help="Output directory (overrides config)")
    run_parser.add_argument("--no-plots", action="store_true", help="Disable interactive plots")

    # --- compare ---
    compare_parser = subparsers.add_parser("compare", help="Compare simulation with experiment")
    compare_parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    compare_parser.add_argument("--results", type=Path, required=True, help="Path to results directory")
    compare_parser.add_argument("--experiment", type=Path, default=None, help="Path to .mat file (overrides config)")
    compare_parser.add_argument("--sensors", type=int, nargs="+", default=None, help="Sensor IDs to plot")

    args = parser.parse_args()

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "compare":
        _cmd_compare(args)
    else:
        parser.print_help()


def _cmd_run(args: argparse.Namespace) -> None:
    from pore_pressure.config import load_config
    from pore_pressure.simulation import run_simulation, save_results

    config = load_config(args.config)

    show_plots = not args.no_plots and config.show_plots

    result = run_simulation(config, show_plots=show_plots)

    output_dir = args.output or config.output_directory
    save_results(result, output_dir)


def _cmd_compare(args: argparse.Namespace) -> None:
    import numpy as np

    from pore_pressure.config import load_config
    from pore_pressure.experiment_data import load_experiment_data
    from pore_pressure.comparison import compare_with_experiment

    config = load_config(args.config)

    results_dir = Path(args.results)
    Pres_distrib_in_Time = np.load(results_dir / "Pres_distrib_in_Time_2.npy")

    mat_path = args.experiment or config.experiment_data_file
    exp_data = load_experiment_data(mat_path)

    compare_with_experiment(Pres_distrib_in_Time, config, exp_data, sensor_ids=args.sensors)


if __name__ == "__main__":
    main()
