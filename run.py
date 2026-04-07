"""Запуск симуляции порового давления из IDE (VS Code / PyCharm)."""

from pore_pressure.config import load_config
from pore_pressure.simulation import run_simulation, save_results

config = load_config("configs/default.yaml")
config.num_steps=2
result = run_simulation(config, show_plots=True)
save_results(result, "output/")
