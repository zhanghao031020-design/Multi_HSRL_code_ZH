from pathlib import Path
import csv
import json
import numpy as np

from hsrl_sim.experiments import run_optical_smoke


def test_optical_smoke_writes_recomputable_records(tmp_path: Path):
    result=run_optical_smoke(output_dir=tmp_path, seed=20260820, n_truth=2, replicates=2, alpha_errors=(0.1,0.3), multistart=2, radius_points=96)
    assert len(result.retrieval_records)>=4
    assert (tmp_path/"truth.csv").exists()
    assert (tmp_path/"observations.csv").exists()
    assert (tmp_path/"retrievals.csv").exists()
    assert (tmp_path/"summary.csv").exists()
    assert (tmp_path/"arm_a.csv").exists()
    assert (tmp_path/"report.md").exists()
    assert (tmp_path/"figures"/"gain_response.png").exists()
    assert (tmp_path/"config.yaml").exists()
    assert (tmp_path/"run_metadata.json").exists()
    assert "scenario-robust" in (tmp_path/"report.md").read_text(encoding="utf-8")
    metadata = json.loads((tmp_path/"run_metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["source_digest"]) == 64
    assert all("config_hash" in row for row in result.retrieval_records)
    assert all("selected_start_plus" in row for row in result.retrieval_records)
    assert all("objective_plus" in row for row in result.retrieval_records)
    assert all("multiple_solutions_base" in row for row in result.retrieval_records)
    assert all("multiple_solutions_plus" in row for row in result.retrieval_records)
    assert all("interval_lower_plus" in row for row in result.retrieval_records)
    assert all("covered_plus" in row for row in result.retrieval_records)
    assert all("interval_informative_plus" in row for row in result.retrieval_records)
    assert all("multiple_solution_rate_plus" in row for row in result.summary_records)
    assert all("coverage_plus" in row for row in result.summary_records)
    assert all("informative_interval_rate_plus" in row for row in result.summary_records)
    assert {row["product_id"] for row in result.retrieval_records} == {
        "Vf", "Vc", "Vf_over_Vc", "reff_total", "reff_coarse"
    }
    assert {row["scenario_id"] for row in result.summary_records} >= {
        "all", "urban_fine", "smoke_mixed"
    }
    with (tmp_path/"observations.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    covariance = np.asarray(json.loads(rows[0]["covariance_plus"]))
    assert covariance.shape == (6, 6)
    assert np.any(np.abs(covariance - np.diag(np.diag(covariance))) > 0)
    grouped = {}
    for row in rows:
        key = (row["truth_id"], row["replicate_id"])
        grouped.setdefault(key, []).append(json.loads(row["optical_base"]))
    assert all(all(vector == vectors[0] for vector in vectors) for vectors in grouped.values())
    base_estimates = {}
    for row in result.retrieval_records:
        if row["product_id"] != "Vc":
            continue
        key = (row["truth_id"], row["replicate_id"])
        base_estimates.setdefault(key, []).append(row["estimate_base"])
    assert all(
        all(value == values[0] for value in values)
        for values in base_estimates.values()
    )
