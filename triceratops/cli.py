"""Command-line interface for high-level TRICERATOPS workflows."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from triceratops.config.config import Config
from triceratops.domain.scenario_id import ScenarioID
from triceratops.lightcurve.config import LightCurveConfig
from triceratops.lightcurve.ephemeris import Ephemeris
from triceratops.validation.runner import (
    ApertureConfig,
    FppRunConfig,
    run_tess_fpp,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    aperture_pixels = tuple(_parse_pixel(text) for text in args.aperture_pixel)
    aperture_mode = "custom" if aperture_pixels else args.aperture_mode
    sectors: tuple[int, ...] | str
    if args.all_sectors:
        sectors = "all"
    elif args.sector:
        sectors = tuple(args.sector)
    else:
        sectors = "auto"

    lc_config = LightCurveConfig(
        cadence=args.cadence,
        sectors=sectors,
        quality_mask=args.quality_mask,
        detrend_method=args.detrend_method,
        sigma_clip=args.sigma_clip,
        flatten_window_length=args.flatten_window_length,
        flatten_polyorder=args.flatten_polyorder,
        phase_window_factor=args.phase_window_factor,
        flux_type=args.flux_type,
        cadence_days_override=args.cadence_days_override,
        supersampling_rate=args.supersampling_rate,
    )
    compute_config = Config(
        n_mc_samples=args.n_mc_samples,
        lnz_const=args.lnz_const,
        n_best_samples=args.n_best_samples,
        parallel=not args.no_parallel,
        flat_priors=args.flat_priors,
        mission="TESS",
        n_workers=args.n_workers,
    )
    scenario_ids = (
        tuple(ScenarioID[name] for name in args.scenario)
        if args.scenario
        else None
    )
    run_config = FppRunConfig(
        aperture=ApertureConfig(
            mode=aperture_mode,
            threshold_sigma=args.aperture_threshold,
            custom_pixels=aperture_pixels,
        ),
        lightcurve=lc_config,
        compute=compute_config,
        bin_count=args.bin_count,
        transit_depth=args.transit_depth,
        search_radius_px=args.search_radius_px,
        sigma_psf_px=args.sigma_psf_px,
        trilegal_cache_path=args.trilegal_cache_path,
        exofop_cache_ttl_seconds=args.exofop_cache_ttl_seconds,
        exofop_disk_cache_dir=args.exofop_disk_cache_dir,
        scenario_ids=scenario_ids,
    )
    ephemeris = _build_ephemeris(args)
    result = run_tess_fpp(args.target, config=run_config, ephemeris=ephemeris)

    if args.json:
        print(json.dumps(_result_to_dict(result), indent=2, sort_keys=True))
        return

    print(f"target: TIC {result.resolved_target.tic_id}")
    print(f"fpp: {result.validation_result.fpp:.8f}")
    print(f"nfpp: {result.validation_result.nfpp:.8f}")
    print(f"sectors: {','.join(str(s) for s in result.light_curve_result.sectors_used)}")
    print(f"cadence: {result.light_curve_result.cadence_used}")
    print(f"transit_depth: {result.transit_depth:.8g}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a tutorial-style TESS FPP validation in one command.",
    )
    parser.add_argument("target", help="TOI target string or TIC ID")
    parser.add_argument("--transit-depth", type=float, default=None)
    parser.add_argument("--period-days", type=float, default=None)
    parser.add_argument("--t0-btjd", type=float, default=None)
    parser.add_argument("--duration-hours", type=float, default=None)
    parser.add_argument("--bin-count", type=int, default=None)
    parser.add_argument(
        "--aperture-mode",
        choices=("default", "pipeline", "threshold", "all"),
        default="default",
    )
    parser.add_argument(
        "--aperture-threshold",
        type=float,
        default=3.0,
        help="Sigma threshold for threshold apertures.",
    )
    parser.add_argument(
        "--aperture-pixel",
        action="append",
        default=[],
        metavar="COL,ROW",
        help="Custom aperture pixel. Repeat to add multiple pixels.",
    )
    parser.add_argument("--search-radius-px", type=int, default=10)
    parser.add_argument("--sigma-psf-px", type=float, default=0.75)
    parser.add_argument(
        "--cadence",
        choices=("auto", "20sec", "2min", "10min", "30min"),
        default="auto",
    )
    parser.add_argument(
        "--quality-mask",
        choices=("default", "hard", "none"),
        default="default",
    )
    parser.add_argument(
        "--detrend-method",
        choices=("flatten", "none"),
        default="flatten",
    )
    parser.add_argument("--sigma-clip", type=float, default=5.0)
    parser.add_argument("--flatten-window-length", type=int, default=401)
    parser.add_argument("--flatten-polyorder", type=int, default=3)
    parser.add_argument("--phase-window-factor", type=float, default=5.0)
    parser.add_argument(
        "--flux-type",
        choices=("pdcsap_flux", "sap_flux"),
        default="pdcsap_flux",
    )
    parser.add_argument("--cadence-days-override", type=float, default=None)
    parser.add_argument("--supersampling-rate", type=int, default=20)
    parser.add_argument("--sector", type=int, action="append", default=[])
    parser.add_argument("--all-sectors", action="store_true")
    parser.add_argument("--n-mc-samples", type=int, default=20_000)
    parser.add_argument("--lnz-const", type=float, default=650.0)
    parser.add_argument("--n-best-samples", type=int, default=1000)
    parser.add_argument("--n-workers", type=int, default=0)
    parser.add_argument("--no-parallel", action="store_true")
    parser.add_argument("--flat-priors", action="store_true")
    parser.add_argument("--scenario", action="append", default=[], choices=[sid.name for sid in ScenarioID])
    parser.add_argument("--trilegal-cache-path", default=None)
    parser.add_argument("--exofop-cache-ttl-seconds", type=int, default=6 * 3600)
    parser.add_argument("--exofop-disk-cache-dir", default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def _build_ephemeris(args: argparse.Namespace) -> Ephemeris | None:
    if args.period_days is None and args.t0_btjd is None and args.duration_hours is None:
        return None
    if args.period_days is None or args.t0_btjd is None:
        raise ValueError(
            "--period-days and --t0-btjd must both be provided for manual ephemeris input"
        )
    return Ephemeris(
        period_days=args.period_days,
        t0_btjd=args.t0_btjd,
        duration_hours=args.duration_hours,
    )


def _parse_pixel(text: str) -> tuple[int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected COL,ROW for aperture pixel, got {text!r}")
    return int(parts[0]), int(parts[1])


def _result_to_dict(result) -> dict[str, object]:
    return {
        "target_ref": result.resolved_target.target_ref,
        "tic_id": result.resolved_target.tic_id,
        "fpp": result.validation_result.fpp,
        "nfpp": result.validation_result.nfpp,
        "transit_depth": result.transit_depth,
        "sectors_used": list(result.light_curve_result.sectors_used),
        "cadence_used": result.light_curve_result.cadence_used,
        "scenario_probabilities": {
            scenario.scenario_id.name: scenario.relative_probability
            for scenario in result.validation_result.scenario_results
        },
    }


__all__ = ["main"]
