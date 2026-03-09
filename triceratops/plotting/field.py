"""Stellar field plot: star positions relative to the target with search radius.

Implements a simplified version of the original plot_field() that uses
angular coordinates (separation_arcsec, position_angle_deg) instead of
pixel-grid coordinates, since pixel-grid data is not available in the
new domain objects.
"""
from __future__ import annotations

from math import ceil, floor

import numpy as np

from triceratops.domain.entities import StellarField


def plot_field(
    stellar_field: StellarField,
    search_radius_px: int | float,
    save: bool = False,
    fname: str | None = None,
) -> None:
    """Plot star positions in the photometric field around the target.

    Displays neighbor star positions relative to the target in angular
    coordinates (arcsec), colour-coded by TESS magnitude, with a dashed
    circle marking the search radius.  The target star is shown as a
    larger star marker.

    Args:
        stellar_field: StellarField containing the target and all neighbour
            stars.  stars[0] is always the target.
        search_radius_px: Search radius in pixels.  Converted to arcsec using
            the TESS pixel scale (20.25 arcsec/pixel) for the radius circle.
        save: If True, save the figure to a file instead of showing it.
        fname: Output filename (without extension).  Ignored when ``save``
            is False.  If ``save`` is True and ``fname`` is None, a default
            name ``TIC<id>_field.pdf`` is used.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    TESS_ARCSEC_PER_PIX = 20.25
    search_radius_arcsec = search_radius_px * TESS_ARCSEC_PER_PIX

    target = stellar_field.target
    stars = stellar_field.stars

    # Build arrays for plotting.
    # Position angles are East of North; convert to (x_arcsec, y_arcsec):
    #   x = sep * sin(PA)  (East is positive x)
    #   y = sep * cos(PA)  (North is positive y)
    pa_rad = np.array([np.deg2rad(s.position_angle_deg) for s in stars])
    sep = np.array([s.separation_arcsec for s in stars])
    x = sep * np.sin(pa_rad)
    y = sep * np.cos(pa_rad)
    tmags = np.array([s.tmag for s in stars])

    vmin = floor(float(np.nanmin(tmags)))
    vmax = ceil(float(np.nanmax(tmags)))

    fig, ax = plt.subplots(figsize=(7, 6.5))
    plt.subplots_adjust(right=0.88)

    # Search radius circle (dashed)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        search_radius_arcsec * np.sin(theta),
        search_radius_arcsec * np.cos(theta),
        "k--",
        alpha=0.5,
        zorder=0,
        label=f"Search radius ({search_radius_px} px)",
    )

    # Neighbour stars (circles, scaled by marker area)
    if len(stars) > 1:
        sc = ax.scatter(
            x[1:],
            y[1:],
            c=tmags[1:],
            s=75,
            edgecolors="k",
            cmap=cm.viridis_r,
            vmin=vmin,
            vmax=vmax,
            zorder=2,
            rasterized=True,
            label="Neighbour stars",
        )
    else:
        # No neighbours -- create a dummy scatter for the colourbar
        sc = ax.scatter(
            [], [],
            c=[],
            cmap=cm.viridis_r,
            vmin=vmin,
            vmax=vmax,
        )

    # Target star (larger star marker)
    ax.scatter(
        [x[0]],
        [y[0]],
        c=[tmags[0]],
        s=250,
        marker="*",
        edgecolors="k",
        cmap=cm.viridis_r,
        vmin=vmin,
        vmax=vmax,
        zorder=3,
        label=f"Target (TIC {target.tic_id})",
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.ax.set_ylabel("TESS mag", rotation=270, fontsize=12, labelpad=18)

    ax.set_xlabel("East offset (arcsec)", fontsize=12)
    ax.set_ylabel("North offset (arcsec)", fontsize=12)
    ax.set_title(f"Stellar field — TIC {target.tic_id}", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_aspect("equal")

    # Compass arrows: N up, E right (already in our coord system)
    arrow_len = search_radius_arcsec * 0.15
    ax.annotate(
        "", xy=(arrow_len, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="k"),
    )
    ax.text(arrow_len * 1.15, 0, "E", ha="left", va="center", fontsize=10)
    ax.annotate(
        "", xy=(0, arrow_len), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="k"),
    )
    ax.text(0, arrow_len * 1.15, "N", ha="center", va="bottom", fontsize=10)

    if save:
        plt.tight_layout()
        if fname is None:
            fname = f"TIC{target.tic_id}_field"
        plt.savefig(f"{fname}.pdf")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)
