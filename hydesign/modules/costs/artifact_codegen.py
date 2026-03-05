from __future__ import annotations

from pathlib import Path

from hydesign.modules.costs.registry import compose_costs_modules
from hydesign.project_builder.source_extract import extract_symbols_module

SUPPORTED_WSB_COSTS_TECHNOLOGIES = ("wind", "solar", "battery")


def compose_supported_costs_modules(
    technologies: tuple[str, ...] | list[str],
):
    normalized = tuple(name.strip().lower().replace("-", "_") for name in technologies)
    if normalized != SUPPORTED_WSB_COSTS_TECHNOLOGIES:
        supported = ", ".join(SUPPORTED_WSB_COSTS_TECHNOLOGIES)
        raise NotImplementedError(
            "Phase 1 costs generation only supports the exact technology set "
            f"({supported}). Requested: {normalized!r}."
        )
    return compose_costs_modules(normalized)


def source_file_hash_targets(modules) -> tuple[str, ...]:
    paths: list[str] = []
    for module in modules:
        for ref in module.source_of_truth:
            path, _, _ = ref.partition(":")
            if path not in paths:
                paths.append(path)
    return tuple(paths)


def render_wind_solar_battery_costs(
    technologies: tuple[str, ...] | list[str],
) -> str:
    modules = compose_supported_costs_modules(technologies)
    source_path = Path("hydesign") / "costs" / "costs.py"
    symbols: list[str] = []
    refs: list[str] = []
    for module in modules:
        refs.extend(module.source_of_truth)
        for symbol in module.symbols:
            if symbol not in symbols:
                symbols.append(symbol)
    return extract_symbols_module(
        source_path=source_path,
        symbol_names=tuple(symbols),
        header_lines=(
            "# Auto-generated combined costs module",
            "# Combined from original costs source of truth: " + ", ".join(refs),
        ),
    )
