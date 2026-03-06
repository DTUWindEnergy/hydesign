from __future__ import annotations

from hydesign.modular.modules.assembly.base import BASE_ASSEMBLY_MODULE
from hydesign.modular.modules.assembly.battery import BATTERY_ASSEMBLY_MODULE
from hydesign.modular.modules.assembly.contracts import AssemblyModuleSpec
from hydesign.modular.modules.assembly.finance import FINANCE_ASSEMBLY_MODULE
from hydesign.modular.modules.assembly.operations import OPERATIONS_ASSEMBLY_MODULE
from hydesign.modular.modules.assembly.solar import SOLAR_ASSEMBLY_MODULE
from hydesign.modular.modules.assembly.wind import WIND_ASSEMBLY_MODULE

AVAILABLE_ASSEMBLY_MODULES = {
    "base": BASE_ASSEMBLY_MODULE,
    "wind": WIND_ASSEMBLY_MODULE,
    "solar": SOLAR_ASSEMBLY_MODULE,
    "battery": BATTERY_ASSEMBLY_MODULE,
    "operations": OPERATIONS_ASSEMBLY_MODULE,
    "finance": FINANCE_ASSEMBLY_MODULE,
}


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


def compose_assembly_modules(
    technologies: tuple[str, ...] | list[str],
) -> tuple[AssemblyModuleSpec, ...]:
    names = _normalize_names(technologies)
    ordered: list[AssemblyModuleSpec] = [AVAILABLE_ASSEMBLY_MODULES["base"]]

    if "wind" in names:
        ordered.append(AVAILABLE_ASSEMBLY_MODULES["wind"])
    if "solar" in names:
        ordered.append(AVAILABLE_ASSEMBLY_MODULES["solar"])
    if "battery" in names:
        ordered.append(AVAILABLE_ASSEMBLY_MODULES["battery"])

    ordered.append(AVAILABLE_ASSEMBLY_MODULES["operations"])
    ordered.append(AVAILABLE_ASSEMBLY_MODULES["finance"])

    included = {spec.name for spec in ordered}
    for spec in ordered:
        missing = [name for name in spec.depends_on if name not in included]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                f"Assembly module '{spec.name}' is missing dependencies: {missing_str}."
            )
    return tuple(ordered)
