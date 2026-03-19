"""ScenarioRegistry: typed mapping from ScenarioID to Scenario instances.

Replaces the hardcoded if/elif dispatch in triceratops.py:calc_probs().
"""
from __future__ import annotations

from collections.abc import Iterator

from triceratops.domain.scenario_id import ScenarioID
from triceratops.scenarios.base import Scenario


class ScenarioRegistry:
    """Dict-backed registry mapping ScenarioIDs to Scenario instances.

    Usage:
        registry = ScenarioRegistry()
        registry.register(TTPScenario(ldc_catalog))
        result = registry.get(ScenarioID.TP)
    """

    def __init__(self) -> None:
        self._scenarios: dict[ScenarioID, Scenario] = {}

    def register(self, scenario: Scenario) -> None:
        """Add a scenario to the registry.

        Raises:
            ValueError: If a scenario with the same ScenarioID is already registered.
        """
        sid = scenario.scenario_id
        if sid in self._scenarios:
            raise ValueError(
                f"Scenario {sid!r} is already registered. "
                f"Use replace() to replace an existing scenario."
            )
        self._scenarios[sid] = scenario

    def replace(self, scenario: Scenario) -> None:
        """Register or replace a scenario (no error if already exists)."""
        self._scenarios[scenario.scenario_id] = scenario

    def get(self, scenario_id: ScenarioID) -> Scenario:
        """Return the registered scenario for the given ID.

        Raises:
            KeyError: If scenario_id is not registered.
        """
        if scenario_id not in self._scenarios:
            raise KeyError(
                f"Scenario {scenario_id!r} is not registered. "
                f"Registered: {sorted(self._scenarios)}"
            )
        return self._scenarios[scenario_id]

    def get_or_none(self, scenario_id: ScenarioID) -> Scenario | None:
        """Return the scenario or None if not registered."""
        return self._scenarios.get(scenario_id)

    def all_scenarios(self) -> list[Scenario]:
        """Return all registered scenarios in insertion order."""
        return list(self._scenarios.values())

    def planet_scenarios(self) -> list[Scenario]:
        """Return scenarios in ScenarioID.planet_scenarios() that are registered."""
        return [
            self._scenarios[sid]
            for sid in ScenarioID.planet_scenarios()
            if sid in self._scenarios
        ]

    def eb_scenarios(self) -> list[Scenario]:
        """Return all registered EB scenarios (including twin variants)."""
        return [s for s in self._scenarios.values() if s.is_eb]

    def trilegal_scenarios(self) -> list[Scenario]:
        """Return registered scenarios that require TRILEGAL data."""
        return [
            self._scenarios[sid]
            for sid in ScenarioID.trilegal_scenarios()
            if sid in self._scenarios
        ]

    def nearby_scenarios(self) -> list[Scenario]:
        """Return registered N-scenarios (used for NFPP computation)."""
        return [
            self._scenarios[sid]
            for sid in ScenarioID.nearby_scenarios()
            if sid in self._scenarios
        ]

    def __contains__(self, scenario_id: ScenarioID) -> bool:
        return scenario_id in self._scenarios

    def __len__(self) -> int:
        return len(self._scenarios)

    def __iter__(self) -> Iterator[ScenarioID]:
        return iter(self._scenarios)


def build_default_registry(ldc_catalog: object | None = None) -> ScenarioRegistry:
    """Create a fully-populated registry with all scenario types.

    Args:
        ldc_catalog: LDC catalog instance. If None, uses the default
            LimbDarkeningCatalog. Pass FixedLDCCatalog for testing.
    """
    from triceratops.scenarios.background_scenarios import (
        BEBScenario,
        BTPScenario,
        DEBScenario,
        DTPScenario,
    )
    from triceratops.scenarios.companion_scenarios import (
        PEBScenario,
        PTPScenario,
        SEBScenario,
        STPScenario,
    )
    from triceratops.scenarios.nearby_scenarios import (
        NEBUnknownScenario,
        NTPUnknownScenario,
    )
    from triceratops.scenarios.target_scenarios import TEBScenario, TTPScenario

    if ldc_catalog is None:
        from triceratops.limb_darkening.catalog import LimbDarkeningCatalog
        ldc_catalog = LimbDarkeningCatalog()
    reg = ScenarioRegistry()

    # Register one scenario per ScenarioID.
    # NTP/NEB remain available for direct use and experimentation via the
    # registry, but ValidationEngine's actual-nearby parity path dispatches
    # real field neighbors through TP/EB kernels instead of these pooled
    # "unknown nearby host" implementations.
    for cls in [
        TTPScenario, TEBScenario,
        PTPScenario, PEBScenario,
        STPScenario, SEBScenario,
        DTPScenario, DEBScenario,
        BTPScenario, BEBScenario,
        NTPUnknownScenario, NEBUnknownScenario,
    ]:
        reg.register(cls(ldc_catalog))

    return reg


DEFAULT_REGISTRY: ScenarioRegistry = build_default_registry()
