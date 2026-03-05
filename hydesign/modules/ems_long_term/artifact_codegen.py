from __future__ import annotations

from pathlib import Path

from hydesign.project_builder.source_extract import extract_symbols_module

SUPPORTED_WSB_EMS_LONG_TERM_TECHNOLOGIES = ("wind", "solar", "battery")


def _normalize_names(names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for name in names:
        item = name.strip().lower().replace("-", "_")
        if not item or item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    return tuple(normalized)


def _validate_technologies(technologies: tuple[str, ...] | list[str]) -> None:
    normalized = _normalize_names(technologies)
    if normalized != SUPPORTED_WSB_EMS_LONG_TERM_TECHNOLOGIES:
        supported = ", ".join(SUPPORTED_WSB_EMS_LONG_TERM_TECHNOLOGIES)
        raise NotImplementedError(
            "Phase 1 EMS long-term generation only supports the exact technology set "
            f"({supported}). Requested: {normalized!r}."
        )


def source_file_hash_targets(
    technologies: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    _validate_technologies(technologies)
    return ("hydesign/ems/ems.py",)


def render_wind_solar_battery_ems_long_term(
    technologies: tuple[str, ...] | list[str],
) -> str:
    _validate_technologies(technologies)
    return extract_symbols_module(
        source_path=Path("hydesign") / "ems" / "ems.py",
        symbol_names=(
            "ems_long_term_operation",
            "ems_long_term_operation_comp",
            "operation_solar_batt_deg",
        ),
        header_lines=(
            "# Auto-generated combined EMS long-term operation module",
            "# Combined from original EMS source of truth: hydesign/ems/ems.py:313-596, hydesign/ems/ems.py:1606-1752",
        ),
    )
