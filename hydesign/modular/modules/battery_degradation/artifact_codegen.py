from __future__ import annotations

from pathlib import Path

from hydesign.modular.project_builder.source_extract import render_full_module

SUPPORTED_WSB_BATTERY_DEGRADATION_TECHNOLOGIES = ("wind", "solar", "battery")


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
    if normalized != SUPPORTED_WSB_BATTERY_DEGRADATION_TECHNOLOGIES:
        supported = ", ".join(SUPPORTED_WSB_BATTERY_DEGRADATION_TECHNOLOGIES)
        raise NotImplementedError(
            "Phase 1 battery degradation generation only supports the exact technology set "
            f"({supported}). Requested: {normalized!r}."
        )


def source_file_hash_targets(
    technologies: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    _validate_technologies(technologies)
    return ("hydesign/battery_degradation.py",)


def render_wind_solar_battery_battery_degradation(
    technologies: tuple[str, ...] | list[str],
) -> str:
    _validate_technologies(technologies)
    return render_full_module(
        source_path=Path("hydesign") / "battery_degradation.py",
        header_lines=(
            "# Auto-generated combined battery degradation module",
            "# Combined from original battery degradation source of truth: hydesign/battery_degradation.py",
        ),
        replacements=(
            (
                "from hydesign.ems.ems import expand_to_lifetime\n",
                "from .ems import expand_to_lifetime\n",
            ),
        ),
    )
