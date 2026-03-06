from __future__ import annotations

from hydesign.modular.modules.ems.base import BASE_EMS_MODULE
from hydesign.modular.modules.ems.battery import BATTERY_EMS_MODULE
from hydesign.modular.modules.ems.contracts import EMSModuleSpec
from hydesign.modular.modules.ems.p2x import P2X_EMS_MODULE
from hydesign.modular.modules.ems.solar import SOLAR_EMS_MODULE
from hydesign.modular.modules.ems.wind import WIND_EMS_MODULE

AVAILABLE_EMS_MODULES = {
    "base": BASE_EMS_MODULE,
    "wind": WIND_EMS_MODULE,
    "solar": SOLAR_EMS_MODULE,
    "battery": BATTERY_EMS_MODULE,
    "p2x": P2X_EMS_MODULE,
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


def compose_ems_modules(
    technologies: tuple[str, ...] | list[str],
) -> tuple[EMSModuleSpec, ...]:
    """Return ordered EMS modules for a technology set."""

    names = _normalize_names(technologies)
    ordered: list[EMSModuleSpec] = [AVAILABLE_EMS_MODULES["base"]]

    for name in names:
        if name == "base":
            continue
        try:
            spec = AVAILABLE_EMS_MODULES[name]
        except KeyError as exc:
            raise KeyError(f"Unknown EMS module '{name}'.") from exc
        ordered.append(spec)

    included = {spec.name for spec in ordered}
    for spec in ordered:
        missing = [
            dependency for dependency in spec.depends_on if dependency not in included
        ]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(
                f"EMS module '{spec.name}' is missing dependencies: {missing_str}."
            )

    return tuple(ordered)
