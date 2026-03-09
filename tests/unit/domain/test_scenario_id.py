"""Tests for triceratops.domain.scenario_id."""
from __future__ import annotations

from triceratops.domain.scenario_id import ScenarioID


class TestScenarioID:
    def test_planet_scenarios_does_not_include_stp(self) -> None:
        assert ScenarioID.STP not in ScenarioID.planet_scenarios()

    def test_planet_scenarios_includes_dtp(self) -> None:
        assert ScenarioID.DTP in ScenarioID.planet_scenarios()

    def test_planet_scenarios_exact(self) -> None:
        assert ScenarioID.planet_scenarios() == frozenset(
            {ScenarioID.TP, ScenarioID.PTP, ScenarioID.DTP}
        )

    def test_nearby_scenarios(self) -> None:
        assert ScenarioID.nearby_scenarios() == frozenset(
            {ScenarioID.NTP, ScenarioID.NEB, ScenarioID.NEBX2P}
        )

    def test_string_values(self) -> None:
        assert ScenarioID.TP == "TP"
        assert ScenarioID.EB == "EB"
        assert ScenarioID.EBX2P == "EBx2P"
        assert ScenarioID.PTP == "PTP"
        assert ScenarioID.DTP == "DTP"
        assert ScenarioID.NEB == "NEB"

    def test_twin_eb_scenarios_subset_of_eb_scenarios(self) -> None:
        twin = ScenarioID.twin_eb_scenarios()
        eb = ScenarioID.eb_scenarios()
        assert twin.issubset(eb), f"Twin scenarios not subset of EB: {twin - eb}"
