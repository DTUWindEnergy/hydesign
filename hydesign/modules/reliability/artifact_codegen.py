from __future__ import annotations

from pathlib import Path

from hydesign.project_builder.source_extract import render_full_module

SUPPORTED_WSB_RELIABILITY_TECHNOLOGIES = ("wind", "solar", "battery")


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
    if normalized != SUPPORTED_WSB_RELIABILITY_TECHNOLOGIES:
        supported = ", ".join(SUPPORTED_WSB_RELIABILITY_TECHNOLOGIES)
        raise NotImplementedError(
            "Phase 1 reliability generation only supports the exact technology set "
            f"({supported}). Requested: {normalized!r}."
        )


def source_file_hash_targets(
    technologies: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    _validate_technologies(technologies)
    return ("hydesign/reliability.py",)


def render_wind_solar_battery_reliability(
    technologies: tuple[str, ...] | list[str],
) -> str:
    _validate_technologies(technologies)
    return render_full_module(
        source_path=Path("hydesign") / "reliability.py",
        header_lines=(
            "# Auto-generated combined reliability module",
            "# Combined from original reliability source of truth: hydesign/reliability.py",
        ),
    )
