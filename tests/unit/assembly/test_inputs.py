"""Tests for assembly types: AssembledInputs, AssemblyMetadata, AssemblyConfig."""
from __future__ import annotations

import dataclasses

import pytest

from triceratops.assembly import AssembledInputs, AssemblyConfig, AssemblyMetadata
from triceratops.assembly.errors import AssemblyConfigError
from triceratops.domain.entities import StellarField
from triceratops.domain.value_objects import StellarParameters
from triceratops.lightcurve.ephemeris import Ephemeris, ResolvedTarget


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stellar_field(tic_id: int = 11111) -> StellarField:
    from triceratops.domain.entities import Star

    star = Star(
        tic_id=tic_id,
        ra_deg=50.0,
        dec_deg=15.0,
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
        flux_ratio=1.0,
        transit_depth_required=0.01,
    )
    return StellarField(
        target_id=tic_id,
        mission="TESS",
        search_radius_pixels=10,
        stars=[star],
    )


def _make_resolved_target(tic_id: int = 11111) -> ResolvedTarget:
    return ResolvedTarget(target_ref=f"TIC {tic_id}", tic_id=tic_id)


# ---------------------------------------------------------------------------
# AssembledInputs tests
# ---------------------------------------------------------------------------


class TestAssembledInputsConstruction:
    def test_constructs_with_all_none_optionals(self) -> None:
        """Construct with only required fields; all optionals default to None."""
        rt = _make_resolved_target()
        sf = _make_stellar_field()
        ai = AssembledInputs(resolved_target=rt, stellar_field=sf)

        assert ai.resolved_target is rt
        assert ai.stellar_field is sf
        assert ai.light_curve is None
        assert ai.contrast_curve is None
        assert ai.molusc_data is None
        assert ai.trilegal_population is None
        assert ai.external_lcs is None
        assert isinstance(ai.metadata, AssemblyMetadata)

    def test_rejects_non_resolved_target(self) -> None:
        """Passing a non-ResolvedTarget for resolved_target raises TypeError."""
        sf = _make_stellar_field()
        with pytest.raises(TypeError, match="ResolvedTarget"):
            AssembledInputs(resolved_target={"target_ref": "x", "tic_id": 1}, stellar_field=sf)  # type: ignore[arg-type]

    def test_rejects_string_as_resolved_target(self) -> None:
        """Passing a string for resolved_target raises TypeError."""
        sf = _make_stellar_field()
        with pytest.raises(TypeError, match="ResolvedTarget"):
            AssembledInputs(resolved_target="TIC 11111", stellar_field=sf)  # type: ignore[arg-type]

    def test_rejects_non_stellar_field(self) -> None:
        """Passing a non-StellarField for stellar_field raises TypeError."""
        rt = _make_resolved_target()
        with pytest.raises(TypeError, match="StellarField"):
            AssembledInputs(resolved_target=rt, stellar_field={"target_id": 1})  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        """AssembledInputs is frozen — attribute assignment raises."""
        rt = _make_resolved_target()
        sf = _make_stellar_field()
        ai = AssembledInputs(resolved_target=rt, stellar_field=sf)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ai.light_curve = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AssemblyMetadata tests
# ---------------------------------------------------------------------------


class TestAssemblyMetadata:
    def test_warnings_independent_per_instance(self) -> None:
        """Two instances have independent warning tuples (default factory)."""
        m1 = AssemblyMetadata()
        m2 = AssemblyMetadata()
        # Tuples are immutable so identity is not the concern;
        # verify both start empty and are equal but are separate defaults.
        assert m1.warnings == ()
        assert m2.warnings == ()

    def test_defaults(self) -> None:
        """Default construction produces all-empty metadata."""
        m = AssemblyMetadata()
        assert m.source_labels == ()
        assert m.warnings == ()
        assert m.artifact_ids == ()
        assert m.created_at_utc is None
        assert m.assembler_version is None
        assert m.per_input_source == ()

    def test_is_frozen(self) -> None:
        """AssemblyMetadata is frozen — attribute assignment raises."""
        m = AssemblyMetadata()
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.warnings = ("oops",)  # type: ignore[misc]

    def test_constructs_with_values(self) -> None:
        """Can construct with explicit values."""
        m = AssemblyMetadata(
            source_labels=("catalog", "trilegal"),
            warnings=("low SNR",),
            artifact_ids=("abc123",),
            created_at_utc="2026-03-10T00:00:00Z",
            assembler_version="0.1.0",
            per_input_source=(("stellar_field", "MAST"),),
        )
        assert m.source_labels == ("catalog", "trilegal")
        assert m.warnings == ("low SNR",)
        assert m.artifact_ids == ("abc123",)
        assert m.created_at_utc == "2026-03-10T00:00:00Z"
        assert m.assembler_version == "0.1.0"
        assert m.per_input_source == (("stellar_field", "MAST"),)


# ---------------------------------------------------------------------------
# AssemblyConfig tests
# ---------------------------------------------------------------------------


class TestAssemblyConfig:
    def test_rejects_search_radius_below_1(self) -> None:
        """catalog_search_radius_px=0 raises AssemblyConfigError."""
        with pytest.raises(AssemblyConfigError):
            AssemblyConfig(catalog_search_radius_px=0)

    def test_rejects_negative_search_radius(self) -> None:
        """catalog_search_radius_px=-5 raises AssemblyConfigError."""
        with pytest.raises(AssemblyConfigError):
            AssemblyConfig(catalog_search_radius_px=-5)

    def test_rejects_unknown_mission(self) -> None:
        """Unknown mission string raises AssemblyConfigError."""
        with pytest.raises(AssemblyConfigError, match="Hubble"):
            AssemblyConfig(mission="Hubble")

    def test_accepts_valid_missions(self) -> None:
        """TESS, Kepler, K2 all construct without error."""
        for mission in ("TESS", "Kepler", "K2"):
            cfg = AssemblyConfig(mission=mission)
            assert cfg.mission == mission

    def test_default_values(self) -> None:
        """Default construction produces expected defaults."""
        cfg = AssemblyConfig()
        assert cfg.include_light_curve is True
        assert cfg.include_trilegal is True
        assert cfg.include_contrast_curve is True
        assert cfg.include_molusc is True
        assert cfg.include_external_lcs is True
        assert cfg.lc_config is None
        assert cfg.catalog_search_radius_px == 10
        assert cfg.mission == "TESS"
        assert cfg.contrast_curve_band == "TESS"
        assert cfg.trilegal_cache_path is None
        assert cfg.require_light_curve is True
        assert cfg.require_stellar_params is True

    def test_is_frozen(self) -> None:
        """AssemblyConfig is frozen — attribute assignment raises."""
        cfg = AssemblyConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.mission = "Kepler"  # type: ignore[misc]

    def test_config_error_inherits_assembly_error(self) -> None:
        """AssemblyConfigError is a subclass of AssemblyError."""
        from triceratops.assembly.errors import AssemblyError

        with pytest.raises(AssemblyError):
            AssemblyConfig(catalog_search_radius_px=0)


# ---------------------------------------------------------------------------
# Error hierarchy tests
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_acquisition_error_inherits_assembly_error(self) -> None:
        from triceratops.assembly.errors import AcquisitionError, AssemblyError

        assert issubclass(AcquisitionError, AssemblyError)

    def test_trilegal_acquisition_inherits_acquisition(self) -> None:
        from triceratops.assembly.errors import AcquisitionError, TRILEGALAcquisitionError

        assert issubclass(TRILEGALAcquisitionError, AcquisitionError)

    def test_catalog_acquisition_inherits_acquisition(self) -> None:
        from triceratops.assembly.errors import AcquisitionError, CatalogAcquisitionError

        assert issubclass(CatalogAcquisitionError, AcquisitionError)

    def test_artifact_load_inherits_acquisition(self) -> None:
        from triceratops.assembly.errors import AcquisitionError, ArtifactLoadError

        assert issubclass(ArtifactLoadError, AcquisitionError)

    def test_data_error_inherits_assembly_error(self) -> None:
        from triceratops.assembly.errors import AssemblyDataError, AssemblyError

        assert issubclass(AssemblyDataError, AssemblyError)

    def test_lightcurve_error_inherits_data_error(self) -> None:
        from triceratops.assembly.errors import AssemblyDataError, AssemblyLightCurveError

        assert issubclass(AssemblyLightCurveError, AssemblyDataError)

    def test_config_error_inherits_assembly_error(self) -> None:
        from triceratops.assembly.errors import AssemblyConfigError, AssemblyError

        assert issubclass(AssemblyConfigError, AssemblyError)
