from __future__ import annotations

from hydesign.modules.costs.battery import BATTERY_COSTS_MODULE
from hydesign.modules.costs.contracts import CostsModuleSpec
from hydesign.modules.costs.shared import SHARED_COSTS_MODULE
from hydesign.modules.costs.solar import SOLAR_COSTS_MODULE
from hydesign.modules.costs.wind import WIND_COSTS_MODULE

AVAILABLE_COSTS_MODULES = {
    "wind": WIND_COSTS_MODULE,
    "solar": SOLAR_COSTS_MODULE,
    "battery": BATTERY_COSTS_MODULE,
    "shared": SHARED_COSTS_MODULE,
}


def compose_costs_modules(
    technologies: tuple[str, ...] | list[str],
) -> tuple[CostsModuleSpec, ...]:
    names = tuple(name.strip().lower().replace("-", "_") for name in technologies)
    ordered: list[CostsModuleSpec] = [SHARED_COSTS_MODULE]
    for name in names:
        if name == "shared":
            continue
        try:
            ordered.append(AVAILABLE_COSTS_MODULES[name])
        except KeyError as exc:
            raise KeyError(f"Unknown costs module '{name}'.") from exc
    return tuple(ordered)
