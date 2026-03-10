from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import ClassVar

from triceratops.lightcurve.ephemeris import Ephemeris, ResolvedTarget
from triceratops.lightcurve.exofop.time_conventions import normalize_epoch_to_btjd
from triceratops.lightcurve.exofop.toi_table import (
    fetch_exofop_toi_table,
    fetch_exofop_toi_table_for_toi,
)


# ---------------------------------------------------------------------------
# Lookup types (frozen dataclasses replacing Pydantic models)
# ---------------------------------------------------------------------------


class LookupStatus(str, Enum):
    OK = "ok"
    DATA_UNAVAILABLE = "data_unavailable"
    TIMEOUT = "timeout"
    RUNTIME_ERROR = "runtime_error"


@dataclass(frozen=True)
class SourceRecord:
    name: str
    version: str
    retrieved_at: datetime
    query: str


@dataclass(frozen=True)
class ToiResolutionResult:
    status: LookupStatus
    toi_query: str
    tic_id: int | None = None
    matched_toi: str | None = None
    period_days: float | None = None
    t0_btjd: float | None = None
    duration_hours: float | None = None
    depth_ppm: float | None = None
    missing_fields: tuple[str, ...] = ()
    source_record: SourceRecord | None = None
    raw_row: dict[str, str] | None = None
    message: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return "timeout" in name or "timed out" in msg or "timeout" in msg


def _status_from_exception(exc: Exception) -> LookupStatus:
    return LookupStatus.TIMEOUT if _is_timeout_error(exc) else LookupStatus.RUNTIME_ERROR


def _to_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            return float(text)
        except Exception:
            continue
    return None


def _to_int(row: dict[str, str], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            return int(float(text))
        except Exception:
            continue
    return None


def _normalize_toi_text(toi: str | float) -> str:
    if isinstance(toi, float):
        return f"{toi:.2f}".rstrip("0").rstrip(".")
    text = str(toi).strip().upper().replace("TOI-", "").replace("TOI", "")
    return text.strip()


def _parse_toi_number(toi: str | float) -> float | None:
    try:
        return float(_normalize_toi_text(toi))
    except Exception:
        return None


def _to_btjd(epoch: float | None) -> float | None:
    return normalize_epoch_to_btjd(epoch)


def _first_matching_toi_row(rows: list[dict[str, str]], toi_query: str | float) -> dict[str, str] | None:
    toi_num = _parse_toi_number(toi_query)
    toi_text = _normalize_toi_text(toi_query)
    for row in rows:
        row_toi = str(row.get("toi") or "").strip()
        if not row_toi:
            continue
        if toi_num is not None:
            try:
                if abs(float(row_toi) - toi_num) < 1e-6:
                    return row
            except Exception:
                pass
        if row_toi.upper().replace("TOI-", "").replace("TOI", "").strip() == toi_text:
            return row
    return None


# ---------------------------------------------------------------------------
# Core resolution function
# ---------------------------------------------------------------------------


def resolve_toi_to_tic_ephemeris_depth(
    toi: str | float,
    *,
    cache_ttl_seconds: int = 24 * 3600,
    disk_cache_dir: str | Path | None = None,
) -> ToiResolutionResult:
    toi_query = str(toi)
    single_source = SourceRecord(
        name="exofop_toi_table_single",
        version="download_toi.php?toi=<TOI>",
        retrieved_at=_now(),
        query=f"toi={toi_query}",
    )
    full_source = SourceRecord(
        name="exofop_toi_table",
        version="download_toi.php",
        retrieved_at=_now(),
        query=f"toi={toi_query}",
    )

    matched: dict[str, str] | None = None
    resolved_source = single_source
    errors: list[str] = []

    try:
        scoped = fetch_exofop_toi_table_for_toi(
            toi_query,
            cache_ttl_seconds=int(cache_ttl_seconds),
            disk_cache_dir=disk_cache_dir,
        )
        matched = _first_matching_toi_row([dict(r) for r in scoped.rows], toi)
    except Exception as exc:
        errors.append(f"TOI-scoped fetch failed: {type(exc).__name__}: {exc}")

    if matched is None:
        resolved_source = full_source
        try:
            table = fetch_exofop_toi_table(
                cache_ttl_seconds=int(cache_ttl_seconds),
                disk_cache_dir=disk_cache_dir,
            )
        except Exception as exc:
            if errors:
                message = "; ".join(errors + [f"full-table fetch failed: {type(exc).__name__}: {exc}"])
            else:
                message = f"Failed to fetch ExoFOP TOI table: {type(exc).__name__}: {exc}"
            return ToiResolutionResult(
                status=_status_from_exception(exc),
                toi_query=toi_query,
                source_record=full_source,
                message=message,
            )
        matched = _first_matching_toi_row([dict(r) for r in table.rows], toi)
        if matched is None:
            return ToiResolutionResult(
                status=LookupStatus.DATA_UNAVAILABLE,
                toi_query=toi_query,
                source_record=SourceRecord(
                    name=full_source.name,
                    version=full_source.version,
                    retrieved_at=full_source.retrieved_at,
                    query=f"toi={toi_query} -> no row match in ExoFOP table",
                ),
                message=f"TOI '{toi_query}' was not found in ExoFOP TOI table",
            )

    tic_id = _to_int(matched, ("tic_id", "tic", "ticid"))
    period_days = _to_float(matched, ("period_days", "period", "per"))
    t0_raw = _to_float(matched, ("epoch_btjd", "epoch_bjd", "epoch", "t0_btjd", "t0"))
    duration_hours = _to_float(
        matched,
        ("duration_hours", "duration_hr", "duration_hrs", "duration", "dur"),
    )
    depth_ppm = _to_float(matched, ("depth_ppm", "depth", "dep_ppt", "dep_ppm"))

    missing_fields: list[str] = []
    if tic_id is None:
        missing_fields.append("tic_id")
    if period_days is None:
        missing_fields.append("period_days")
    if t0_raw is None:
        missing_fields.append("t0_btjd")
    if duration_hours is None:
        missing_fields.append("duration_hours")
    if depth_ppm is None:
        missing_fields.append("depth_ppm")

    required_core_missing = any(
        f in missing_fields for f in ("tic_id", "period_days", "t0_btjd", "duration_hours")
    )
    status = LookupStatus.DATA_UNAVAILABLE if required_core_missing else LookupStatus.OK
    msg = None
    if missing_fields:
        msg = f"Matched TOI row but missing fields: {', '.join(missing_fields)}"
    if errors:
        prefix = "; ".join(errors)
        msg = f"{prefix}; {msg}" if msg else prefix

    return ToiResolutionResult(
        status=status,
        toi_query=toi_query,
        tic_id=tic_id,
        matched_toi=str(matched.get("toi") or toi_query),
        period_days=period_days,
        t0_btjd=_to_btjd(t0_raw),
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
        missing_fields=tuple(missing_fields),
        source_record=SourceRecord(
            name=resolved_source.name,
            version=resolved_source.version,
            retrieved_at=resolved_source.retrieved_at,
            query=f"toi={toi_query} -> matched toi={matched.get('toi')}",
        ),
        raw_row={str(k): str(v) for k, v in matched.items()},
        message=msg,
    )


# ---------------------------------------------------------------------------
# ExoFopEphemerisResolver — implements EphemerisResolver protocol
# ---------------------------------------------------------------------------


class ExoFopEphemerisResolver:
    """Resolves a TOI string to a ResolvedTarget via ExoFOP.

    Implements the EphemerisResolver protocol:
        def resolve(self, target: str) -> ResolvedTarget
    """

    def __init__(
        self,
        *,
        cache_ttl_seconds: int = 6 * 3600,
        disk_cache_dir: str | Path | None = None,
    ) -> None:
        self._cache_ttl_seconds = cache_ttl_seconds
        self._disk_cache_dir = disk_cache_dir

    def resolve(self, target: str) -> ResolvedTarget:
        """Resolve a TOI string (e.g. "395.01") to a ResolvedTarget.

        Raises LightCurveError (or subclass) if the TOI cannot be resolved
        or is missing required ephemeris fields.
        """
        from triceratops.lightcurve.errors import LightCurveError

        result = resolve_toi_to_tic_ephemeris_depth(
            target,
            cache_ttl_seconds=self._cache_ttl_seconds,
            disk_cache_dir=self._disk_cache_dir,
        )

        if result.status != LookupStatus.OK:
            raise LightCurveError(
                f"ExoFOP resolution failed for '{target}': "
                f"{result.message or result.status.value}"
            )

        if result.tic_id is None:
            raise LightCurveError(
                f"ExoFOP resolution for '{target}' returned no TIC ID"
            )
        if result.period_days is None or result.t0_btjd is None:
            raise LightCurveError(
                f"ExoFOP resolution for '{target}' missing period or epoch: "
                f"{result.message}"
            )

        ephemeris = Ephemeris(
            period_days=result.period_days,
            t0_btjd=result.t0_btjd,
            duration_hours=result.duration_hours,
        )

        warnings: list[str] = []
        if result.message:
            warnings.append(result.message)

        return ResolvedTarget(
            target_ref=target,
            tic_id=result.tic_id,
            ephemeris=ephemeris,
            source="exofop",
        )


__all__ = [
    "LookupStatus",
    "SourceRecord",
    "ToiResolutionResult",
    "ExoFopEphemerisResolver",
    "resolve_toi_to_tic_ephemeris_depth",
]
