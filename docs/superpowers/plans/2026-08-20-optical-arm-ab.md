# Tri-Wavelength HSRL Optical Arm A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, testable optical-level Arm A/B simulator that compares `3beta+2alpha` against `3beta+3alpha` for the conditional value of `alpha1064`.

**Architecture:** A typed aerosol state is converted from a bimodal lognormal volume distribution into three-wavelength aerosol optics using a mature Mie implementation. A shared six-element optical observation is sampled with a positive-definite covariance; the base branch removes only `alpha1064`. A bounded, multi-start four-parameter retrieval and paired bootstrap report product-level risk changes.

**Tech Stack:** Python 3.12, NumPy, SciPy, miepython, PyYAML, pytest.

---

### Task 1: Project Metadata And State/Distribution Kernel

**Files:**
- Create: `03 代码/pyproject.toml`
- Create: `03 代码/src/hsrl_sim/schemas.py`
- Create: `03 代码/src/hsrl_sim/distributions.py`
- Test: `03 代码/tests/test_distributions.py`

- [ ] **Step 1: Write failing tests for conserved modal volume and effective radius.**

```python
distribution = modal_volume_distribution(radius_m, total_volume=2e-12, median_radius_m=0.1e-6, sigma_g=1.6)
assert trapezoid(distribution, radius_m) == pytest.approx(2e-12, rel=2e-3)
assert effective_radius(radius_m, distribution) > 0
```

- [ ] **Step 2: Run the distribution test and verify import failure before implementation.**

Run: `python -m pytest tests/test_distributions.py -v`
Expected: collection failure because `hsrl_sim` does not yet exist.

- [ ] **Step 3: Implement frozen state dataclasses and lognormal distribution functions.**

```python
@dataclass(frozen=True)
class AerosolState:
    fine_volume: float
    coarse_volume: float
    fine_rv_m: float
    coarse_rv_m: float
```

- [ ] **Step 4: Re-run the distribution test and verify it passes.**

Run: `python -m pytest tests/test_distributions.py -v`
Expected: all distribution assertions pass.

### Task 2: Three-Wavelength Optical Forward Model

**Files:**
- Create: `03 代码/src/hsrl_sim/molecular.py`
- Create: `03 代码/src/hsrl_sim/mie_forward.py`
- Test: `03 代码/tests/test_mie_forward.py`

- [ ] **Step 1: Write failing tests for positive optics, linear volume scaling, and Rayleigh wavelength trend.**

```python
optics = compute_aerosol_optics(state, DEFAULT_WAVELENGTHS_M, radius_grid_m)
assert np.all(optics.alpha_aerosol_m_inv > 0)
assert np.all(optics.beta_aerosol_m_inv_sr_inv > 0)
assert molecular_backscatter_m_inv_sr_inv(355e-9) > molecular_backscatter_m_inv_sr_inv(1064e-9)
```

- [ ] **Step 2: Run the forward-model test and verify the missing-module failure.**

Run: `python -m pytest tests/test_mie_forward.py -v`
Expected: collection failure because forward-model modules do not yet exist.

- [ ] **Step 3: Implement vectorized Mie integration and an explicit Qback-to-differential-backscatter conversion.**

```python
beta = trapezoid(number_density * qback * radius_grid_m**2 / 4.0, radius_grid_m)
```

- [ ] **Step 4: Re-run the forward-model test and verify it passes.**

Run: `python -m pytest tests/test_mie_forward.py -v`
Expected: all physics trend and scaling assertions pass.

### Task 3: Paired Observation And Four-Parameter Retrieval

**Files:**
- Create: `03 代码/src/hsrl_sim/optical_observation.py`
- Create: `03 代码/src/hsrl_sim/inverse_lowdim.py`
- Test: `03 代码/tests/test_retrieval.py`

- [ ] **Step 1: Write failing tests that base is the plus vector without alpha1064 and that a noiseless plus retrieval recovers state.**

```python
assert np.array_equal(observation.optical_base, observation.optical_plus[:-1])
assert result.converged
assert result.state_estimate.fine_volume == pytest.approx(state.fine_volume, rel=0.08)
```

- [ ] **Step 2: Run retrieval tests and verify the expected missing-module failure.**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: collection failure because observation/retrieval modules do not yet exist.

- [ ] **Step 3: Implement PSD-safe covariance sampling, Cholesky whitening, transformed bounded least squares, and deterministic multi-starts.**

```python
residual = solve_triangular(cholesky(covariance, lower=True), observed - predicted, lower=True)
result = least_squares(residual_function, initial_log_parameters, bounds=(lower, upper))
```

- [ ] **Step 4: Re-run retrieval tests and verify they pass.**

Run: `python -m pytest tests/test_retrieval.py -v`
Expected: base/plus identity and deterministic noiseless recovery pass.

### Task 4: Arm A/B Experiment, Paired Metrics, And Smoke Run

**Files:**
- Create: `03 代码/src/hsrl_sim/metrics.py`
- Create: `03 代码/src/hsrl_sim/experiments.py`
- Create: `03 代码/src/hsrl_sim/io.py`
- Create: `03 代码/scripts/run_optical_mc.py`
- Create: `03 代码/configs/smoke.yaml`
- Test: `03 代码/tests/test_metrics.py`
- Test: `03 代码/tests/test_end_to_end.py`

- [ ] **Step 1: Write failing tests for paired gain and a five-truth smoke run that writes re-computable records.**

```python
summary = paired_product_metrics(records, product_id="Vc", bootstrap_samples=100, seed=7)
assert summary.gain > 0
assert (output_dir / "retrievals.csv").exists()
```

- [ ] **Step 2: Run metrics and end-to-end tests and verify missing-module failure.**

Run: `python -m pytest tests/test_metrics.py tests/test_end_to_end.py -v`
Expected: collection failure because experiments and metrics do not yet exist.

- [ ] **Step 3: Implement product evaluation, truth-level paired bootstrap, Arm A Fisher diagnostics, Arm B error scan, CSV outputs, and YAML-driven command.**

```python
gain = (rmse_base - rmse_plus) / rmse_base
bootstrap_units = records.groupby("truth_id").mean(numeric_only=True)
```

- [ ] **Step 4: Run the complete test suite and smoke command.**

Run: `python -m pytest -v; python scripts/run_optical_mc.py --config configs/smoke.yaml`
Expected: test suite passes and output directory contains config snapshot, truth, observations, retrievals, summary, and Arm A diagnostics.

### Task 5: Scope And Quality Review

**Files:**
- Review: `03 代码/README.md`
- Review: `03 代码/docs/superpowers/plans/2026-08-20-optical-arm-ab.md`
- Review: `03 代码/tests/`

- [ ] **Step 1: Check that no Arm C signal-level claim or full-particle-spectrum claim appears in output labels.**

Run: `rg -n "six.*signal|complete particle|full particle" README.md src tests`
Expected: no statement claiming completed Arm C or uniquely retrieved full particle spectrum.

- [ ] **Step 2: Run fresh full verification after review fixes.**

Run: `python -m pytest -v; python scripts/run_optical_mc.py --config configs/smoke.yaml`
Expected: all tests pass and the smoke output is reproducible with the stored seed and configuration.
