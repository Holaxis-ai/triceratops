"""Numerical regression guard for ValidationWorkspace after Step 5 rewrite.

Verifies that the workspace's compute_probs() produces bit-identical FPP/NFPP
before and after the orchestrator delegation rewrite. Any numerical drift
indicates a bug in the assembly ordering or RNG-state handling.
"""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.config.config import Config
from triceratops.domain.entities import LightCurve, Star, StellarField
from triceratops.domain.scenario_id import ScenarioID
from triceratops.domain.value_objects import StellarParameters
from triceratops.validation.workspace import ValidationWorkspace

# Known-good values recorded with seed=42, n_mc_samples=100, n_best_samples=10,
# scenario_ids=[TP, EB, PTP, PEB, STP, SEB] (non-TRILEGAL only).
KNOWN_FPP = 0.23634664252645243
KNOWN_NFPP = 0.0

NON_TRILEGAL_SCENARIOS = [
    ScenarioID.TP,
    ScenarioID.EB,
    ScenarioID.PTP,
    ScenarioID.PEB,
    ScenarioID.STP,
    ScenarioID.SEB,
]


def _default_star() -> Star:
    return Star(
        tic_id=12345678,
        ra_deg=83.82,
        dec_deg=-5.39,
        tmag=10.5,
        jmag=9.8,
        hmag=9.5,
        kmag=9.4,
        bmag=11.2,
        vmag=10.8,
        stellar_params=StellarParameters(
            mass_msun=1.0,
            radius_rsun=1.0,
            teff_k=5778.0,
            logg=4.44,
            metallicity_dex=0.0,
            parallax_mas=10.0,
        ),
    )


def _neighbor_star() -> Star:
    return Star(
        tic_id=99999999,
        ra_deg=83.83,
        dec_deg=-5.38,
        tmag=12.0,
        jmag=11.5,
        hmag=11.2,
        kmag=11.1,
        bmag=12.5,
        vmag=12.2,
    )


class _StubCatalogProvider:
    def query_nearby_stars(
        self, tic_id: int, search_radius_px: int, mission: str,
    ) -> StellarField:
        return StellarField(
            target_id=12345678,
            mission="TESS",
            search_radius_pixels=10,
            stars=[_default_star(), _neighbor_star()],
        )


def _make_lc() -> LightCurve:
    time = np.linspace(-0.1, 0.1, 50)
    flux = np.ones(50)
    flux[20:30] = 0.999
    return LightCurve(time_days=time, flux=flux, flux_err=0.001)


class TestWorkspaceNumericalRegression:
    def test_compute_probs_fpp_regression(self) -> None:
        """FPP from workspace is numerically identical after Step 5 rewrite."""
        np.random.seed(42)
        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=_StubCatalogProvider(),
            config=Config(n_mc_samples=100, n_best_samples=10),
        )
        np.random.seed(42)
        result = ws.compute_probs(
            _make_lc(), period_days=5.0,
            scenario_ids=NON_TRILEGAL_SCENARIOS,
        )
        assert result.fpp == pytest.approx(KNOWN_FPP, abs=1e-10)

    def test_compute_probs_nfpp_regression(self) -> None:
        """NFPP from workspace is numerically identical after Step 5 rewrite."""
        np.random.seed(42)
        ws = ValidationWorkspace(
            tic_id=12345678,
            sectors=np.array([1]),
            catalog_provider=_StubCatalogProvider(),
            config=Config(n_mc_samples=100, n_best_samples=10),
        )
        np.random.seed(42)
        result = ws.compute_probs(
            _make_lc(), period_days=5.0,
            scenario_ids=NON_TRILEGAL_SCENARIOS,
        )
        assert result.nfpp == pytest.approx(KNOWN_NFPP, abs=1e-10)

    def test_compute_probs_deterministic(self) -> None:
        """Two runs with the same seed produce identical results."""
        results = []
        for _ in range(2):
            np.random.seed(42)
            ws = ValidationWorkspace(
                tic_id=12345678,
                sectors=np.array([1]),
                catalog_provider=_StubCatalogProvider(),
                config=Config(n_mc_samples=100, n_best_samples=10),
            )
            np.random.seed(42)
            r = ws.compute_probs(
                _make_lc(), period_days=5.0,
                scenario_ids=NON_TRILEGAL_SCENARIOS,
            )
            results.append(r)
        assert results[0].fpp == results[1].fpp
        assert results[0].nfpp == results[1].nfpp
