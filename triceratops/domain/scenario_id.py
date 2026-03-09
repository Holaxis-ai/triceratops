"""Scenario identifiers for all supported astrophysical scenarios."""
from __future__ import annotations

from enum import StrEnum


class ScenarioID(StrEnum):
    """All supported astrophysical scenarios. StrEnum so values are strings."""

    # Target star
    TP = "TP"
    EB = "EB"
    EBX2P = "EBx2P"
    # Bound companion
    PTP = "PTP"
    PEB = "PEB"
    PEBX2P = "PEBx2P"
    # Sibling (companion-hosted)
    STP = "STP"
    SEB = "SEB"
    SEBX2P = "SEBx2P"
    # Diluted background
    DTP = "DTP"
    DEB = "DEB"
    DEBX2P = "DEBx2P"
    # Background companion EB
    BTP = "BTP"
    BEB = "BEB"
    BEBX2P = "BEBx2P"
    # Nearby unknown / evolved
    NTP = "NTP"
    NEB = "NEB"
    NEBX2P = "NEBx2P"

    @classmethod
    def planet_scenarios(cls) -> frozenset[ScenarioID]:
        """Scenarios that count as true planet detections for FPP.

        FPP = 1 - sum(P(s) for s in planet_scenarios()).

        Matches triceratops.py:1635:
            self.FPP = 1-(prob_df.prob[0]+prob_df.prob[3]+prob_df.prob[9])
        where index 0=TP, 3=PTP, 9=DTP. STP (index 6) is NOT included — this
        matches the published TRICERATOPS methodology. Do not change.
        """
        return frozenset({cls.TP, cls.PTP, cls.DTP})

    @classmethod
    def nearby_scenarios(cls) -> frozenset[ScenarioID]:
        """Scenarios counted toward NFPP."""
        return frozenset({cls.NTP, cls.NEB, cls.NEBX2P})

    @classmethod
    def trilegal_scenarios(cls) -> frozenset[ScenarioID]:
        """Scenarios requiring TRILEGAL background population data."""
        return frozenset({cls.DTP, cls.DEB, cls.DEBX2P,
                          cls.BTP, cls.BEB, cls.BEBX2P})

    @classmethod
    def contrast_scenarios(cls) -> frozenset[ScenarioID]:
        """Scenarios that use a contrast curve when available."""
        return frozenset({cls.PTP, cls.PEB, cls.PEBX2P,
                          cls.STP, cls.SEB, cls.SEBX2P,
                          cls.DTP, cls.DEB, cls.DEBX2P,
                          cls.BTP, cls.BEB, cls.BEBX2P})

    @classmethod
    def twin_eb_scenarios(cls) -> frozenset[ScenarioID]:
        """Base EB scenarios that also return a twin (q>=0.95) result."""
        return frozenset({cls.EB, cls.PEB, cls.SEB, cls.DEB, cls.BEB, cls.NEB})

    @classmethod
    def eb_scenarios(cls) -> frozenset[ScenarioID]:
        """All EB scenarios including twin variants."""
        return frozenset({cls.EB, cls.EBX2P,
                          cls.PEB, cls.PEBX2P,
                          cls.SEB, cls.SEBX2P,
                          cls.DEB, cls.DEBX2P,
                          cls.BEB, cls.BEBX2P,
                          cls.NEB, cls.NEBX2P})
