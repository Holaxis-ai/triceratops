"""Load MOLUSC CSV files into MoluscData objects."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pandas import read_csv

from triceratops.domain.molusc import MoluscData


def load_molusc_file(path: Path) -> MoluscData:
    """Load a MOLUSC CSV file into a MoluscData object.

    Reads the CSV once at prep time. The returned MoluscData is
    picklable and carries no filesystem references.
    """
    df = read_csv(path)
    if len(df) == 0:
        raise ValueError(f"MOLUSC file {path} has zero data rows")
    required = {"semi-major axis(AU)", "eccentricity", "mass ratio"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MOLUSC file {path} is missing columns: {missing}")
    arrays = {}
    for col, key in [
        ("semi-major axis(AU)", "semi_major_axis_au"),
        ("eccentricity", "eccentricity"),
        ("mass ratio", "mass_ratio"),
    ]:
        try:
            arr = np.asarray(df[col], dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"MOLUSC file {path}: column '{col}' contains non-numeric values. "
                f"Original error: {exc}"
            ) from exc
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"MOLUSC file {path}: column '{col}' contains non-finite values "
                f"(NaN or inf)."
            )
        arrays[key] = arr
    return MoluscData(**arrays)
