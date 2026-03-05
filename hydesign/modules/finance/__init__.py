"""Finance module contracts and WSB artifact generation."""

from hydesign.modules.finance.artifact_codegen import (
    SUPPORTED_WSB_FINANCE_TECHNOLOGIES,
    compose_supported_finance_modules,
    render_wind_solar_battery_finance,
    source_file_hash_targets,
)
from hydesign.modules.finance.base import BASE_FINANCE_MODULE
from hydesign.modules.finance.battery import BATTERY_FINANCE_MODULE
from hydesign.modules.finance.contracts import FinanceModuleSpec
from hydesign.modules.finance.registry import (
    AVAILABLE_FINANCE_MODULES,
    compose_finance_modules,
)
from hydesign.modules.finance.shared import SHARED_FINANCE_MODULE
from hydesign.modules.finance.solar import SOLAR_FINANCE_MODULE
from hydesign.modules.finance.wind import WIND_FINANCE_MODULE

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
