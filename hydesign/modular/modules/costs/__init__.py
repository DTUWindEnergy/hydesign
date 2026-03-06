"""Costs module contracts and WSB artifact generation."""

from hydesign.modular.modules.costs.artifact_codegen import (
    SUPPORTED_WSB_COSTS_TECHNOLOGIES,
    compose_supported_costs_modules,
    render_wind_solar_battery_costs,
    source_file_hash_targets,
)
from hydesign.modular.modules.costs.battery import BATTERY_COSTS_MODULE
from hydesign.modular.modules.costs.contracts import CostsModuleSpec
from hydesign.modular.modules.costs.registry import (
    AVAILABLE_COSTS_MODULES,
    compose_costs_modules,
)
from hydesign.modular.modules.costs.shared import SHARED_COSTS_MODULE
from hydesign.modular.modules.costs.solar import SOLAR_COSTS_MODULE
from hydesign.modular.modules.costs.wind import WIND_COSTS_MODULE

__all__ = [
    "AVAILABLE_COSTS_MODULES",
    "BATTERY_COSTS_MODULE",
    "CostsModuleSpec",
    "SHARED_COSTS_MODULE",
    "SOLAR_COSTS_MODULE",
    "SUPPORTED_WSB_COSTS_TECHNOLOGIES",
    "WIND_COSTS_MODULE",
    "compose_costs_modules",
    "compose_supported_costs_modules",
    "render_wind_solar_battery_costs",
    "source_file_hash_targets",
]
