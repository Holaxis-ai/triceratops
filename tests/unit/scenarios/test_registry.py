"""Tests for triceratops.scenarios.registry -- ScenarioRegistry."""
from __future__ import annotations

import numpy as np
import pytest

from triceratops.domain.result import ScenarioResult
from triceratops.domain.scenario_id import ScenarioID
from triceratops.limb_darkening.catalog import FixedLDCCatalog
from triceratops.scenarios.base import BaseScenario
from triceratops.scenarios.registry import DEFAULT_REGISTRY, ScenarioRegistry

# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class _StubScenario(BaseScenario):
    """Configurable stub for registry testing."""

    def __init__(self, sid: ScenarioID, eb: bool = False) -> None:
        super().__init__(FixedLDCCatalog())
        self._sid = sid
        self._eb = eb

    @property
    def scenario_id(self) -> ScenarioID:
        return self._sid

    @property
    def is_eb(self) -> bool:
        return self._eb

    def _get_host_ldc(self, sp, mission, P_orb, kwargs):
        return FixedLDCCatalog().get_coefficients("TESS", 0, 5000, 4.5)

    def _sample_priors(self, n, sp, P_orb, config, **kw):
        return {"incs": np.full(n, 89.0), "eccs": np.zeros(n),
                "argps": np.zeros(n), "rps": np.full(n, 2.0)}

    def _compute_orbital_geometry(self, samples, P_orb, sp, config, **kw):
        n = len(P_orb)
        return {"a": np.full(n, 1e12), "Ptra": np.full(n, 0.05),
                "b": np.zeros(n), "coll": np.zeros(n, dtype=bool)}

    def _evaluate_lnL(self, lc, lnsigma, samples, geom, ldc, ext, config):
        return np.full(config.n_mc_samples, -1.0), None

    def _pack_result(self, samples, geom, ldc, lnZ, idx, sp, ext, twin=False):
        n = len(idx)
        return ScenarioResult(
            scenario_id=self._sid, host_star_tic_id=0, ln_evidence=lnZ,
            host_mass_msun=np.ones(n), host_radius_rsun=np.ones(n),
            host_u1=np.full(n, 0.4), host_u2=np.full(n, 0.2),
            period_days=np.full(n, 5.0), inclination_deg=np.full(n, 89.0),
            impact_parameter=np.zeros(n), eccentricity=np.zeros(n),
            arg_periastron_deg=np.zeros(n), planet_radius_rearth=np.full(n, 2.0),
            eb_mass_msun=np.zeros(n), eb_radius_rsun=np.zeros(n),
            flux_ratio_eb_tess=np.zeros(n), companion_mass_msun=np.zeros(n),
            companion_radius_rsun=np.zeros(n), flux_ratio_companion_tess=np.zeros(n),
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScenarioRegistry:
    def test_register_and_get(self) -> None:
        reg = ScenarioRegistry()
        s = _StubScenario(ScenarioID.TP)
        reg.register(s)
        assert reg.get(ScenarioID.TP) is s

    def test_register_duplicate_raises(self) -> None:
        reg = ScenarioRegistry()
        reg.register(_StubScenario(ScenarioID.TP))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_StubScenario(ScenarioID.TP))

    def test_replace_no_error(self) -> None:
        reg = ScenarioRegistry()
        s1 = _StubScenario(ScenarioID.TP)
        s2 = _StubScenario(ScenarioID.TP)
        reg.register(s1)
        reg.replace(s2)
        assert reg.get(ScenarioID.TP) is s2

    def test_get_unregistered_raises(self) -> None:
        reg = ScenarioRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get(ScenarioID.TP)

    def test_get_or_none_registered(self) -> None:
        reg = ScenarioRegistry()
        s = _StubScenario(ScenarioID.TP)
        reg.register(s)
        assert reg.get_or_none(ScenarioID.TP) is s

    def test_get_or_none_unregistered(self) -> None:
        reg = ScenarioRegistry()
        assert reg.get_or_none(ScenarioID.TP) is None

    def test_contains_registered(self) -> None:
        reg = ScenarioRegistry()
        reg.register(_StubScenario(ScenarioID.TP))
        assert ScenarioID.TP in reg

    def test_contains_unregistered(self) -> None:
        reg = ScenarioRegistry()
        assert ScenarioID.TP not in reg

    def test_len_correct(self) -> None:
        reg = ScenarioRegistry()
        assert len(reg) == 0
        reg.register(_StubScenario(ScenarioID.TP))
        assert len(reg) == 1
        reg.register(_StubScenario(ScenarioID.EB, eb=True))
        assert len(reg) == 2

    def test_all_scenarios_returns_list(self) -> None:
        reg = ScenarioRegistry()
        reg.register(_StubScenario(ScenarioID.TP))
        reg.register(_StubScenario(ScenarioID.EB, eb=True))
        result = reg.all_scenarios()
        assert len(result) == 2

    def test_planet_scenarios_subset(self) -> None:
        reg = ScenarioRegistry()
        reg.register(_StubScenario(ScenarioID.TP))
        reg.register(_StubScenario(ScenarioID.PTP))
        reg.register(_StubScenario(ScenarioID.EB, eb=True))
        planets = reg.planet_scenarios()
        planet_ids = {s.scenario_id for s in planets}
        assert planet_ids == {ScenarioID.TP, ScenarioID.PTP}

    def test_eb_scenarios_all_is_eb_true(self) -> None:
        reg = ScenarioRegistry()
        reg.register(_StubScenario(ScenarioID.TP))
        reg.register(_StubScenario(ScenarioID.EB, eb=True))
        reg.register(_StubScenario(ScenarioID.PEB, eb=True))
        ebs = reg.eb_scenarios()
        assert len(ebs) == 2
        assert all(s.is_eb for s in ebs)

    def test_nearby_scenarios_empty_when_none_registered(self) -> None:
        reg = ScenarioRegistry()
        reg.register(_StubScenario(ScenarioID.TP))
        assert reg.nearby_scenarios() == []

    def test_nearby_scenarios_returns_registered(self) -> None:
        reg = ScenarioRegistry()
        reg.register(_StubScenario(ScenarioID.NTP))
        reg.register(_StubScenario(ScenarioID.NEB, eb=True))
        nearby = reg.nearby_scenarios()
        nearby_ids = {s.scenario_id for s in nearby}
        assert nearby_ids == {ScenarioID.NTP, ScenarioID.NEB}

    def test_iter(self) -> None:
        reg = ScenarioRegistry()
        reg.register(_StubScenario(ScenarioID.TP))
        reg.register(_StubScenario(ScenarioID.EB, eb=True))
        ids = list(reg)
        assert ScenarioID.TP in ids
        assert ScenarioID.EB in ids

    def test_empty_registry(self) -> None:
        reg = ScenarioRegistry()
        assert len(reg) == 0
        assert reg.all_scenarios() == []
        assert reg.planet_scenarios() == []
        assert reg.eb_scenarios() == []

    def test_default_registry_importable(self) -> None:
        assert DEFAULT_REGISTRY is not None

    def test_default_registry_is_scenario_registry_instance(self) -> None:
        assert isinstance(DEFAULT_REGISTRY, ScenarioRegistry)
