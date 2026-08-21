import numpy as np
import pytest

from hsrl_sim.metrics import paired_gain, paired_product_metrics


def test_paired_gain_is_rmse_reduction():
    base=np.array([2.0,4.0,6.0,8.0])
    plus=0.75*base
    truth=np.zeros(4)
    assert paired_gain(base,plus,truth)==pytest.approx(0.25)


def test_paired_metrics_preserve_failures_and_compute_bootstrap_interval():
    records=[
        {"truth_id":"a","estimate_base":2.0,"estimate_plus":1.0,"truth":0.0,"converged_base":True,"converged_plus":True,"multiple_solutions_base":False,"multiple_solutions_plus":False,"covered_base":True,"covered_plus":True,"interval_width_base":2.0,"interval_width_plus":1.0,"interval_informative_base":True,"interval_informative_plus":True},
        {"truth_id":"b","estimate_base":4.0,"estimate_plus":2.0,"truth":0.0,"converged_base":True,"converged_plus":True,"multiple_solutions_base":True,"multiple_solutions_plus":False,"covered_base":False,"covered_plus":True,"interval_width_base":4.0,"interval_width_plus":2.0,"interval_informative_base":True,"interval_informative_plus":False},
        {"truth_id":"c","estimate_base":0.0,"estimate_plus":0.0,"truth":0.0,"converged_base":False,"converged_plus":True,"multiple_solutions_base":False,"multiple_solutions_plus":True,"covered_base":False,"covered_plus":False,"interval_width_base":float("nan"),"interval_width_plus":3.0,"interval_informative_base":False,"interval_informative_plus":True},
    ]
    result=paired_product_metrics(records, bootstrap_samples=100, seed=7)
    assert result.n_total==3
    assert result.n_paired_success==2
    assert result.failure_rate_base==pytest.approx(1/3)
    assert result.multiple_solution_rate_base==pytest.approx(1/3)
    assert result.multiple_solution_rate_plus==pytest.approx(1/3)
    assert result.coverage_base==pytest.approx(0.5)
    assert result.coverage_plus==pytest.approx(2/3)
    assert result.mean_interval_width_base==pytest.approx(3.0)
    assert result.mean_interval_width_plus==pytest.approx(2.0)
    assert result.informative_interval_rate_base==pytest.approx(2/3)
    assert result.informative_interval_rate_plus==pytest.approx(2/3)
    assert result.gain>0
    assert result.bootstrap_ci_low<=result.gain<=result.bootstrap_ci_high
