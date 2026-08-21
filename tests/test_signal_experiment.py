from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from hsrl_sim.experiments import run_signal_mc


def test_signal_mc_writes_raw_and_paired_records(tmp_path: Path):
    result = run_signal_mc(
        output_dir=tmp_path,
        seed=20260821,
        n_truth=2,
        replicates=2,
        signal_levels=(0.5, 1.0),
        multistart=2,
        radius_points=96,
        poisson=True,
        gain_relative_error=0.02,
        laser_energy_relative_error=0.01,
        discriminator_relative_error=0.02,
    )

    assert len(result.paired_retrieval_records) == 8 * 5
    assert len(result.retrieval_records) == 8 * 5 * 2
    assert len(result.summary_records) == 3 * 5 * 2
    for filename in (
        "truth.csv",
        "observations.csv",
        "retrievals.csv",
        "paired_retrievals.csv",
        "summary.csv",
        "report.md",
        "config.yaml",
        "run_metadata.json",
    ):
        assert (tmp_path / filename).exists()

    with (tmp_path / "observations.csv").open(encoding="utf-8", newline="") as handle:
        observation_rows = list(csv.DictReader(handle))
    assert len(observation_rows) == 8
    assert len(json.loads(observation_rows[0]["raw_counts"])) == 6
    assert len(json.loads(observation_rows[0]["raw_signal"])) == 6
    assert np.asarray(json.loads(observation_rows[0]["covariance_counts"])).shape == (6, 6)
    assert "molecular_1064" in observation_rows[0]["channel_names"]
    assert len(json.loads(observation_rows[0]["gain_multiplier"])) == 6
    assert len(json.loads(observation_rows[0]["laser_energy_multiplier"])) == 3
    assert len(json.loads(observation_rows[0]["discriminator_multiplier"])) == 3
    assert np.asarray(json.loads(observation_rows[0]["cross_talk_multiplier"])).shape == (3, 2)

    assert {row["arm_id"] for row in result.retrieval_records} == {"base", "plus"}
    assert all("truth_id" in row and "seed" in row and "config_hash" in row for row in result.retrieval_records)
    assert all(row["arm_id"] == "paired_base_plus" for row in result.paired_retrieval_records)
    assert "Arm C" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_signal_mc_reproducibly_reuses_one_raw_observation_for_base_and_plus(tmp_path: Path):
    result = run_signal_mc(
        output_dir=tmp_path,
        seed=7,
        n_truth=1,
        replicates=1,
        signal_levels=(1.0,),
        multistart=2,
        radius_points=64,
        poisson=False,
    )

    base = [row for row in result.retrieval_records if row["arm_id"] == "base"]
    plus = [row for row in result.retrieval_records if row["arm_id"] == "plus"]
    assert len(base) == len(plus) == 5
    assert {row["observation_id"] for row in base} == {row["observation_id"] for row in plus}
    assert all(row["raw_observation_shared"] is True for row in result.paired_retrieval_records)
