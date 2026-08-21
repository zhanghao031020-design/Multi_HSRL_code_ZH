from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml

from hsrl_sim.experiments import run_signal_mc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run six-channel signal-level Arm C Monte Carlo")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or Path("outputs") / datetime.now().strftime("signal-%Y%m%d-%H%M%S")
    result = run_signal_mc(
        output_dir=output_dir,
        seed=int(config["seed"]),
        n_truth=int(config["n_truth"]),
        replicates=int(config["replicates"]),
        signal_levels=tuple(float(value) for value in config["signal_levels"]),
        multistart=int(config["multistart"]),
        radius_points=int(config["radius_points"]),
        radius_min_um=float(config.get("radius_min_um", 0.01)),
        radius_max_um=float(config.get("radius_max_um", 15.0)),
        poisson=bool(config.get("poisson", True)),
        gain_relative_error=float(config.get("gain_relative_error", 0.02)),
        laser_energy_relative_error=float(config.get("laser_energy_relative_error", 0.01)),
        discriminator_relative_error=float(config.get("discriminator_relative_error", 0.02)),
        discriminator_cross_talk_relative_error=float(
            config.get("discriminator_cross_talk_relative_error", 0.02)
        ),
        min_molecular_expected_counts=float(config.get("min_molecular_expected_counts", 20.0)),
        model_mismatch=bool(config.get("model_mismatch", True)),
        coverage_target=float(config.get("coverage_target", 0.90)),
        max_interval_factor=float(config.get("max_interval_factor", 10.0)),
    )
    print(f"output_dir={output_dir}")
    for row in result.summary_records:
        if row["scenario_id"] == "all":
            print(row)


if __name__ == "__main__":
    main()
