"""Best-fit light-curve plots for each non-negligible scenario.

Implements plot_fits() — a grid of subplots (3 columns: TP-type, EB-type,
EBx2P-type) showing the data and the best-fit model for every scenario that
has a non-negligible relative probability.
"""
from __future__ import annotations

import numpy as np

from triceratops.domain.entities import LightCurve
from triceratops.domain.result import ScenarioResult, ValidationResult
from triceratops.domain.scenario_id import ScenarioID

# Minimum relative probability to display a scenario subplot.
_MIN_PROB = 1e-10

# Grouping of scenario IDs into the three columns (TP-type, EB-type, EBx2P-type).
_TP_SCENARIOS: frozenset[ScenarioID] = frozenset({
    ScenarioID.TP, ScenarioID.PTP, ScenarioID.STP,
    ScenarioID.DTP, ScenarioID.BTP, ScenarioID.NTP,
})
_EB_SCENARIOS: frozenset[ScenarioID] = frozenset({
    ScenarioID.EB, ScenarioID.PEB, ScenarioID.SEB,
    ScenarioID.DEB, ScenarioID.BEB, ScenarioID.NEB,
})
_EBX2P_SCENARIOS: frozenset[ScenarioID] = frozenset({
    ScenarioID.EBX2P, ScenarioID.PEBX2P, ScenarioID.SEBX2P,
    ScenarioID.DEBX2P, ScenarioID.BEBX2P, ScenarioID.NEBX2P,
})


def _column_for_scenario(sid: ScenarioID) -> int:
    """Return the subplot column index (0, 1, or 2) for a scenario ID."""
    if sid in _TP_SCENARIOS:
        return 0
    if sid in _EB_SCENARIOS:
        return 1
    return 2


def _is_companion_scenario(sid: ScenarioID) -> bool:
    """Return True for scenarios where the transit host is not the target star."""
    return sid in frozenset({
        ScenarioID.STP, ScenarioID.SEB, ScenarioID.SEBX2P,
        ScenarioID.BTP, ScenarioID.BEB, ScenarioID.BEBX2P,
    })


def _best_fit_model(
    model_time: np.ndarray,
    scenario_result: ScenarioResult,
    light_curve: LightCurve,
) -> np.ndarray:
    """Compute the best-fit model light curve for a single scenario.

    Uses the median of the best-fit parameter arrays stored in
    ``scenario_result`` to produce a single representative model curve.

    Args:
        model_time: Dense time array for plotting the smooth model.
        scenario_result: Scenario best-fit arrays (shape: n_best_samples).
        light_curve: Original light curve (used for cadence / supersampling).

    Returns:
        Normalised flux array, shape (len(model_time),).
    """
    from triceratops.config.config import CONST
    from triceratops.likelihoods.geometry import semi_major_axis
    from triceratops.likelihoods.transit_model import (
        simulate_eb_transit,
        simulate_planet_transit,
    )

    sr = scenario_result
    sid = sr.scenario_id

    # Guard: if the scenario was effectively skipped (all mass == 0), return flat.
    if float(np.median(sr.host_mass_msun)) == 0.0:
        return np.ones(len(model_time))

    # Median best-fit scalars
    M_s = float(np.median(sr.host_mass_msun))
    R_s = float(np.median(sr.host_radius_rsun))
    u1 = float(np.median(sr.host_u1))
    u2 = float(np.median(sr.host_u2))
    P_orb = float(np.median(sr.period_days))
    inc = float(np.median(sr.inclination_deg))
    ecc = float(np.median(sr.eccentricity))
    argp = float(np.median(sr.arg_periastron_deg))

    fr_comp = float(np.median(sr.flux_ratio_companion_tess))
    companion_is_host = _is_companion_scenario(sid)

    is_eb = sid in (ScenarioID.eb_scenarios())

    if is_eb:
        R_eb = float(np.median(sr.eb_radius_rsun))
        fr_eb = float(np.median(sr.flux_ratio_eb_tess))

        # Semi-major axis uses total system mass for EBs
        M_eb = float(np.median(sr.eb_mass_msun))
        a = float(semi_major_axis(np.array([P_orb]), M_s + M_eb)[0])

        flux, _ = simulate_eb_transit(
            time=model_time,
            rs=R_s,
            rcomp=R_eb,
            eb_flux_ratio=fr_eb,
            period=P_orb,
            inc=inc,
            a=a,
            u1=u1,
            u2=u2,
            ecc=ecc,
            argp=argp,
            companion_flux_ratio=fr_comp,
            companion_is_host=companion_is_host,
            exptime=light_curve.cadence_days,
            nsamples=light_curve.supersampling_rate,
        )
    else:
        R_p = float(np.median(sr.planet_radius_rearth))
        a = float(semi_major_axis(np.array([P_orb]), M_s)[0])

        flux = simulate_planet_transit(
            time=model_time,
            rp=R_p,
            period=P_orb,
            inc=inc,
            a=a,
            rs=R_s,
            u1=u1,
            u2=u2,
            ecc=ecc,
            argp=argp,
            companion_flux_ratio=fr_comp,
            companion_is_host=companion_is_host,
            exptime=light_curve.cadence_days,
            nsamples=light_curve.supersampling_rate,
        )

    return np.asarray(flux)


def plot_fits(
    light_curve: LightCurve,
    validation_result: ValidationResult,
    save: bool = False,
    fname: str | None = None,
) -> None:
    """Plot best-fit model light curves for all non-negligible scenarios.

    Produces a grid of subplots with three columns:
      - Column 0: TP-type scenarios (TP, PTP, STP, DTP, BTP, NTP)
      - Column 1: EB-type scenarios (EB, PEB, SEB, DEB, BEB, NEB)
      - Column 2: EBx2P-type scenarios (EBx2P, PEBX2P, …)

    Each subplot shows the phase-folded data as blue dots and the best-fit
    model as a solid black line.  Scenarios with
    ``relative_probability < 1e-10`` are skipped.

    Args:
        light_curve: Phase-folded, normalised LightCurve used in the run.
        validation_result: ValidationResult from a completed compute_probs()
            call.  Each ScenarioResult must have non-empty best-fit arrays.
        save: If True, save the figure to a file; otherwise call plt.show().
        fname: Base filename (without extension).  If ``save`` is True and
            ``fname`` is None, defaults to ``TIC<id>_fits.pdf``.
    """
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    # Filter to scenarios worth plotting
    visible = [
        r for r in validation_result.scenario_results
        if r.relative_probability >= _MIN_PROB
        and len(r.host_mass_msun) > 0
    ]

    if not visible:
        # Nothing to plot -- emit a figure with a single "no scenarios" message
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5, 0.5,
            "No scenarios with relative_probability ≥ 1e-10",
            ha="center", va="center", transform=ax.transAxes, fontsize=12,
        )
        ax.axis("off")
        if save:
            if fname is None:
                fname = f"TIC{validation_result.target_id}_fits"
            plt.savefig(f"{fname}.pdf")
        else:
            plt.tight_layout()
            plt.show()
        plt.close(fig)
        return

    nrows = len(visible)
    model_time = np.linspace(float(np.min(light_curve.time_days)),
                              float(np.max(light_curve.time_days)), 200)

    fig, axes = plt.subplots(nrows, 3, figsize=(12, nrows * 4), sharex=False)

    # Ensure axes is always 2-D, shape (nrows, 3)
    if nrows == 1:
        axes = np.array([axes])

    # Arrange visible scenarios into (row, col) grid.
    # Each row corresponds to one scenario; the column is determined by scenario type.
    # Rows are filled top-to-bottom in the order scenarios appear.
    for row_idx, sr in enumerate(visible):
        col_idx = _column_for_scenario(sr.scenario_id)

        # All three columns show the same scenario's data; the other two
        # columns in this row are left blank so the layout stays clean.
        # (This matches the original: one scenario per grid row.)
        for c in range(3):
            ax = axes[row_idx, c]
            if c != col_idx:
                ax.axis("off")
                continue

            # Renormalise data to the host star's flux contribution
            fr = float(np.median(sr.flux_ratio_companion_tess))
            if _is_companion_scenario(sr.scenario_id) and 0.0 < fr < 1.0:
                lc_plot = light_curve.with_renorm(fr)
            else:
                lc_plot = light_curve

            y_formatter = ticker.ScalarFormatter(useOffset=False)
            ax.yaxis.set_major_formatter(y_formatter)

            ax.errorbar(
                lc_plot.time_days,
                lc_plot.flux,
                lc_plot.flux_err,
                fmt="o",
                color="dodgerblue",
                elinewidth=1.0,
                capsize=0,
                markeredgecolor="black",
                alpha=0.25,
                zorder=0,
                rasterized=True,
            )

            # Best-fit model
            try:
                best_model = _best_fit_model(model_time, sr, light_curve)
            except Exception:  # noqa: BLE001
                best_model = np.ones(len(model_time))

            ax.plot(model_time, best_model, "k-", lw=2.5, zorder=2)

            ax.set_ylabel("normalised flux", fontsize=11)

            # Annotations: host star TIC ID and scenario name
            host_label = (
                str(sr.host_star_tic_id) if sr.host_star_tic_id != 0
                else str(validation_result.target_id)
            )
            ax.annotate(
                host_label,
                xy=(0.05, 0.92),
                xycoords="axes fraction",
                fontsize=11,
            )
            ax.annotate(
                str(sr.scenario_id),
                xy=(0.05, 0.05),
                xycoords="axes fraction",
                fontsize=11,
            )

        # x-label on the last row only
        if row_idx == nrows - 1:
            for c in range(3):
                if not axes[row_idx, c].get_visible():
                    continue
                axes[row_idx, c].set_xlabel(
                    "days from transit centre", fontsize=11,
                )

    if save:
        plt.tight_layout()
        if fname is None:
            fname = f"TIC{validation_result.target_id}_fits"
        plt.savefig(f"{fname}.pdf")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)
