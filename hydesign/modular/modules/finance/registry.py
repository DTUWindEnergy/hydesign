from __future__ import annotations

from hydesign.modular.modules.finance.base import BASE_FINANCE_MODULE
from hydesign.modular.modules.finance.battery import BATTERY_FINANCE_MODULE
from hydesign.modular.modules.finance.contracts import FinanceModuleSpec
from hydesign.modular.modules.finance.shared import SHARED_FINANCE_MODULE
from hydesign.modular.modules.finance.solar import SOLAR_FINANCE_MODULE
from hydesign.modular.modules.finance.wind import WIND_FINANCE_MODULE

AVAILABLE_FINANCE_MODULES = {
    "base": BASE_FINANCE_MODULE,
    "wind": WIND_FINANCE_MODULE,
    "solar": SOLAR_FINANCE_MODULE,
    "battery": BATTERY_FINANCE_MODULE,
    "shared": SHARED_FINANCE_MODULE,
}


def compose_finance_modules(
    technologies: tuple[str, ...] | list[str],
) -> tuple[FinanceModuleSpec, ...]:
    names = tuple(name.strip().lower().replace("-", "_") for name in technologies)
    ordered: list[FinanceModuleSpec] = [BASE_FINANCE_MODULE, SHARED_FINANCE_MODULE]
    for name in names:
        if name in {"base", "shared"}:
            continue
        try:
            ordered.append(AVAILABLE_FINANCE_MODULES[name])
        except KeyError as exc:
            raise KeyError(f"Unknown finance module '{name}'.") from exc
    return tuple(ordered)
