"""EB q<0.95 / q>=0.95 branching helpers.

The eclipsing-binary mass-ratio split appears in every EB scenario's
_evaluate_lnL method:

  - q < 0.95  → standard EB at the observed period P_orb using geometry["coll"]
  - q >= 0.95 → equal-mass twin EB at 2×P_orb using geometry["coll_twin"]

This module provides a single helper, ``build_eb_branch_masks``, that
constructs both transit masks from the common inputs so the duplicated
four-line block is written once and tested in isolation.

NC-04 note
----------
For BEBScenario the q<0.95 block must pass ``geometry["coll"]`` (not
``geometry["coll_twin"]``).  That invariant is respected here because
callers always supply the standard ``coll`` array for the q<0.95 branch;
the helper does not make any assumption about which collision array is
"correct" — it just threads through whatever the caller supplies.
"""
from __future__ import annotations

import numpy as np

from triceratops.scenarios.constants import EB_Q_TWIN_THRESHOLD
from triceratops.scenarios.kernels import build_transit_mask


def build_eb_branch_masks(
    qs: np.ndarray,
    incs: np.ndarray,
    ptra: np.ndarray,
    coll: np.ndarray,
    ptra_twin: np.ndarray,
    coll_twin: np.ndarray,
    extra_mask: np.ndarray | None = None,
    q_threshold: float = EB_Q_TWIN_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Build transit masks for the standard (q<threshold) and twin (q>=threshold) EB branches.

    Args:
        qs: Mass-ratio samples, shape (N,).
        incs: Inclination samples in degrees, shape (N,).
        ptra: Geometric transit probabilities for the standard branch, shape (N,).
        coll: Collision flags for the standard branch, shape (N,). True = reject.
        ptra_twin: Geometric transit probabilities for the twin branch, shape (N,).
        coll_twin: Collision flags for the twin branch, shape (N,). True = reject.
        extra_mask: Optional additional boolean mask applied to *both* branches
            (e.g. main-sequence filter, qs_comp != 0 check).  Shape (N,).
        q_threshold: Mass-ratio boundary (default EB_Q_TWIN_THRESHOLD = 0.95).
            Draws with q < threshold use the standard branch; draws with
            q >= threshold use the twin branch.

    Returns:
        (mask, mask_twin): Boolean arrays of shape (N,) selecting which draws
        should have their likelihood evaluated in each branch.  Draws outside
        a branch are set to False (and the caller's lnL initialised to -inf
        for those slots will remain unchanged).
    """
    q_lt = qs < q_threshold
    q_ge = qs >= q_threshold

    if extra_mask is not None:
        q_lt = q_lt & extra_mask
        q_ge = q_ge & extra_mask

    mask = build_transit_mask(incs, ptra, coll, extra_mask=q_lt)
    mask_twin = build_transit_mask(incs, ptra_twin, coll_twin, extra_mask=q_ge)

    return mask, mask_twin
