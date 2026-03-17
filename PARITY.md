# Numerical Parity with TRICERATOPS

This library is a modernized rewrite of the original
[TRICERATOPS](https://github.com/stevengiacalone/triceratops) codebase
(Giacalone & Dressing 2020, Giacalone et al. 2021). It implements the same
Bayesian framework, the same ~34 astrophysical scenarios, and the same
Monte Carlo evidence computation.

## Summary

The rewrite reproduces the original TRICERATOPS results **with four
documented numerical corrections** to bugs in the original code. Three of
the four are mathematically provable errors; the fourth (NC-04) corrects a
logic error whose impact is below Monte Carlo noise for tested targets.

All corrections are conservative: they fix cases where the original code
**underweighted** false-positive scenarios, meaning the original FPP values
were systematically biased high. After corrections, FPP values decrease
slightly for affected targets.

| ID | Bug | Impact on FPP |
|----|-----|---------------|
| [NC-01](#nc-01-log-sum-exp-fix-for-marginal-likelihood-underflow) | log-sum-exp underflow in marginal likelihood | Decreases (background scenarios gain probability mass) |
| [NC-02](#nc-02-analytic-psf-integral-replaces-adaptive-quadrature) | Numerical quadrature replaced by analytic PSF integral | Small, dominated by NC-01 |
| [NC-03](#nc-03-log10-to-natural-log-in-background-prior) | `np.log10` used where `np.log` is required in background prior | Decreases (background scenarios gain probability mass) |
| [NC-04](#nc-04-beb-collision-check-uses-wrong-orbit) | Wrong collision check in BEB q < 0.95 branch | Below Monte Carlo noise at n=10,000 |

## How we verify parity

### Golden regression tests

The `tests/golden/` directory contains deterministic regression tests for
two TESS targets (TOI-4051 and TOI-4155) run at fixed seeds. These tests
lock the per-scenario log-evidence (lnZ), FPP, and NFPP to known values
with explicit tolerances:

- **lnZ per scenario**: rtol <= 0.01 (1%) with identical random seed
- **FPP, NFPP**: atol <= 1e-4

Golden JSON fixtures (`tests/fixtures/golden/toi4051.json`,
`toi4155.json`) store the expected values. Any code change that shifts
these values beyond tolerance fails CI.

### Stub providers

Golden tests use stub catalog and population providers (no network calls),
making them fully deterministic and reproducible on any machine.

### Cross-reference parity tests

`tests/test_parity.py` contains optional tests that import the original
TRICERATOPS codebase (when available at a local path) and compare transit
model outputs directly. These are skipped in normal CI but can be run
locally to verify model-level equivalence.

## Detailed numerical changes

Each correction is documented below with: what changed, why the original
behavior was incorrect, the corrected formula, affected scenarios, and the
observed FPP shift.

---

### NC-01 — log-sum-exp fix for marginal likelihood underflow

**Status:** Applied
**Affects FPP:** Yes (decreases)
**Affects NFPP:** Yes

**The bug.** The original code computes marginal log-likelihood as:

```python
Z   = mean(exp(lnL + const))
lnZ = log(Z)
```

When most MC draws produce very poor fits (`lnL ~ -10,000`),
`exp(lnL - lnL_max)` underflows to 0.0 in float64 (threshold ~exp(-745)).
This sets `lnZ = -inf`, assigning **zero probability** to the scenario
regardless of actual evidence.

**The fix.** The log-sum-exp identity:

```
log(mean(exp(lnL))) = lnL_max + log(sum(exp(lnL - lnL_max)) / N)
```

Every term `exp(lnL[i] - lnL_max)` is in (0, 1] and never underflows.
Mathematically identical, numerically stable.

**Observed shift (TOI-4051, n=10,000, seed=42):**

| Metric | Original | Corrected |
|--------|----------|-----------|
| FPP    | 0.999173 | 0.996334  |
| NFPP   | 0.000146 | 0.000000  |

---

### NC-02 — Analytic PSF integral replaces adaptive quadrature

**Status:** Applied
**Affects FPP:** Yes (small, dominated by NC-01)
**Affects NFPP:** Yes

**The bug.** `calc_depths()` integrates a 2D Gaussian PSF over each
aperture pixel using `scipy.integrate.dblquad` with a Python loop. The
adaptive quadrature tolerance is non-uniform across the parameter space,
producing errors up to ~1e-6 for off-center stars.

**The fix.** The 2D Gaussian integral over a pixel box is separable with an
exact closed form using the standard normal CDF:

```
integral = A * [Phi((px+0.5-mu_x)/sigma) - Phi((px-0.5-mu_x)/sigma)]
             * [Phi((py+0.5-mu_y)/sigma) - Phi((py-0.5-mu_y)/sigma)]
```

`scipy.special.ndtr` evaluates this to machine precision in a single
vectorized call.

---

### NC-03 — log10 to natural log in background prior

**Status:** Applied
**Affects FPP:** Yes (decreases)
**Affects NFPP:** No

**The bug.** The occurrence-rate prior for background companions uses
`np.log10` at three call sites where the natural log `np.log` is required:

```python
# Original (incorrect)
lnprior = np.log10(n_comp / 0.1 * (1/3600)**2 * s**2)

# Corrected
lnprior = np.log(n_comp / 0.1 * (1/3600)**2 * s**2)
```

Since `log10(x) = log(x) / 2.303`, the original code underweights
background scenarios by a factor of ~2.3x in log space, inflating FPP.

**Affected scenarios:** BTP, BEB, BEBx2P, DTP, DEB, DEBx2P.

---

### NC-04 — BEB collision check uses wrong orbit

**Status:** Applied
**Affects FPP:** Below Monte Carlo noise at n=10,000
**Affects NFPP:** No

**The bug.** In `BEBScenario`, the q < 0.95 branch (standard-period EB)
uses the twin-orbit collision check (`coll_twin`) instead of the
standard-orbit check (`coll`). Since the twin orbit has a larger
semi-major axis, the collision criterion is less restrictive, admitting
physically impossible configurations.

**The fix.** Use `geometry["coll"]` for q < 0.95, matching the correct
pattern already used in `DEBScenario`.

**Observed shift:** Zero at 4 significant figures for both test targets.
The fix is logically correct but the affected draw count is negligible at
n=10,000.

---

## Acknowledgment

The original TRICERATOPS method and codebase were developed by
Steven Giacalone and Courtney Dressing. This rewrite builds on their
foundational work. The numerical corrections documented here are offered
in the spirit of improving the tool for the community, not as criticism
of the original implementation — these are subtle numerical issues that
are easy to miss in research code.

## References

- Giacalone, S. & Dressing, C. D. 2020, *"Vetting of 384 TESS Objects of Interest with TRICERATOPS and Statistical Validation of 12 Planet Candidates"*, AJ, 161, 24
- Giacalone, S. et al. 2021, *"Validation of 13 Hot and Potentially Terrestrial TESS Planets"*, AJ, 163, 99
