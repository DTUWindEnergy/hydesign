"""Finance module contracts and WSB artifact generation."""

from hydesign.modular.modules.finance.artifact_codegen import (
    SUPPORTED_WSB_FINANCE_TECHNOLOGIES,
    compose_supported_finance_modules,
    render_wind_solar_battery_finance,
    source_file_hash_targets,
)
from hydesign.modular.modules.finance.base import BASE_FINANCE_MODULE
from hydesign.modular.modules.finance.battery import BATTERY_FINANCE_MODULE
from hydesign.modular.modules.finance.contracts import FinanceModuleSpec
from hydesign.modular.modules.finance.registry import (
    AVAILABLE_FINANCE_MODULES,
    compose_finance_modules,
)
from hydesign.modular.modules.finance.shared import SHARED_FINANCE_MODULE
from hydesign.modular.modules.finance.solar import SOLAR_FINANCE_MODULE
from hydesign.modular.modules.finance.wind import WIND_FINANCE_MODULE

__all__ = [
    "AVAILABLE_FINANCE_MODULES",
    "BASE_FINANCE_MODULE",
    "BATTERY_FINANCE_MODULE",
    "FinanceModuleSpec",
    "SHARED_FINANCE_MODULE",
    "SOLAR_FINANCE_MODULE",
    "SUPPORTED_WSB_FINANCE_TECHNOLOGIES",
    "WIND_FINANCE_MODULE",
    "compose_finance_modules",
    "compose_supported_finance_modules",
    "render_wind_solar_battery_finance",
    "source_file_hash_targets",
]
