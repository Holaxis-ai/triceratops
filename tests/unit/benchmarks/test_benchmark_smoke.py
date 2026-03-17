"""Smoke tests for the benchmark harness (BenchmarkingEngine).

These tests verify that:
- BenchmarkingEngine runs the full TOI-4051 pipeline and returns per-scenario timings.
- Every scenario result has a finite, positive elapsed time.

Uses n=100 for speed.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

INTEGRATION_DIR = Path(__file__).parent.parent.parent / "fixtures" / "integration"
TIC_ID = 237101326

# Expected number of timing entries.
# The registry has 12 scenario classes; since no nearby-star transit depth is
# set in this fixture, NTP/NEB are excluded, leaving 10 scenario calls.
# Each EB call returns (EB, EBx2P) — 5 EB scenarios × 2 + 5 non-EB × 1 = 15
# total timing entries, one per unique ScenarioID that appears in the results.
EXPECTED_TIMING_COUNT = 15


@pytest.fixture(scope="module")
def benchmarking_engine_result():
    """Run BenchmarkingEngine on TOI-4051 fixture data with n=100."""
    from benchmarks.bench_full_pipeline import BenchmarkingEngine, load_toi4051_inputs  # type: ignore[import-not-found]

    np.random.seed(42)
    engine, light_curve, stellar_field, config, extra_kwargs = load_toi4051_inputs(n=100)

    result = engine._compute(
        light_curve=light_curve,
        stellar_field=stellar_field,
        period_days=extra_kwargs["period_days"],
        config=config,
        external_lcs=extra_kwargs["external_lcs"],
        contrast_curve=extra_kwargs["contrast_curve"],
    )
    return engine, result


@pytest.mark.unit
def test_benchmarking_engine_runs(benchmarking_engine_result):
    """BenchmarkingEngine should complete and return per-scenario timings."""
    engine, result = benchmarking_engine_result

    timings = engine.scenario_timings
    assert isinstance(timings, dict), "scenario_timings should be a dict"
    assert len(timings) == EXPECTED_TIMING_COUNT, (
        f"Expected {EXPECTED_TIMING_COUNT} timing entries, got {len(timings)}. "
        f"Keys: {sorted(timings.keys())}"
    )


@pytest.mark.unit
def test_all_scenarios_have_timing(benchmarking_engine_result):
    """Every scenario timing entry should be a finite, positive float."""
    engine, _ = benchmarking_engine_result

    for scenario_id, elapsed in engine.scenario_timings.items():
        assert isinstance(elapsed, float), (
            f"Timing for {scenario_id!r} is not a float: {elapsed!r}"
        )
        assert math.isfinite(elapsed), (
            f"Timing for {scenario_id!r} is not finite: {elapsed}"
        )
        assert elapsed > 0.0, (
            f"Timing for {scenario_id!r} is not positive: {elapsed}"
        )
