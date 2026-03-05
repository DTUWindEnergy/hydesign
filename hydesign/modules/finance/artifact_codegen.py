from __future__ import annotations

from pathlib import Path

from hydesign.modules.finance.registry import compose_finance_modules
from hydesign.project_builder.source_extract import extract_symbols_module

SUPPORTED_WSB_FINANCE_TECHNOLOGIES = ("wind", "solar", "battery")


def compose_supported_finance_modules(
    technologies: tuple[str, ...] | list[str],
):
    normalized = tuple(name.strip().lower().replace("-", "_") for name in technologies)
    if normalized != SUPPORTED_WSB_FINANCE_TECHNOLOGIES:
        supported = ", ".join(SUPPORTED_WSB_FINANCE_TECHNOLOGIES)
        raise NotImplementedError(
            "Phase 1 finance generation only supports the exact technology set "
            f"({supported}). Requested: {normalized!r}."
        )
    return compose_finance_modules(normalized)


def source_file_hash_targets(modules) -> tuple[str, ...]:
    paths: list[str] = []
    for module in modules:
        for ref in module.source_of_truth:
            path, _, _ = ref.partition(":")
            if path not in paths:
                paths.append(path)
    return tuple(paths)


def render_wind_solar_battery_finance(
    technologies: tuple[str, ...] | list[str],
) -> str:
    modules = compose_supported_finance_modules(technologies)
    source_path = Path("hydesign") / "finance" / "finance.py"
    refs: list[str] = []
    symbols: list[str] = []
    for module in modules:
        refs.extend(module.source_of_truth)
    for module in modules:
        for symbol in module.symbols:
            if symbol not in symbols:
                symbols.append(symbol)
    return extract_symbols_module(
        source_path=source_path,
        symbol_names=tuple(symbols),
        header_lines=(
            "# Auto-generated combined finance module",
            "# Combined from original finance source of truth: " + ", ".join(refs),
        ),
    )
