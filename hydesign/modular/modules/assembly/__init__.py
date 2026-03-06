"""Assembly module generation for the baseline WSB combined project."""

from hydesign.modular.modules.assembly.artifact_codegen import (
    SUPPORTED_WSB_ASSEMBLY_TECHNOLOGIES,
    compose_supported_assembly_modules,
    render_wind_solar_battery_assembly,
    source_file_hash_targets,
)
from hydesign.modular.modules.assembly.base import BASE_ASSEMBLY_MODULE
from hydesign.modular.modules.assembly.battery import BATTERY_ASSEMBLY_MODULE
from hydesign.modular.modules.assembly.contracts import (
    AssemblyComponentSpec,
    AssemblyDesignVariable,
    AssemblyModuleSpec,
)
from hydesign.modular.modules.assembly.finance import FINANCE_ASSEMBLY_MODULE
from hydesign.modular.modules.assembly.operations import OPERATIONS_ASSEMBLY_MODULE
from hydesign.modular.modules.assembly.registry import (
    AVAILABLE_ASSEMBLY_MODULES,
    compose_assembly_modules,
)
from hydesign.modular.modules.assembly.solar import SOLAR_ASSEMBLY_MODULE
from hydesign.modular.modules.assembly.wind import WIND_ASSEMBLY_MODULE

__all__ = [
    "ASSEMBLY_MODULES",
    "AssemblyComponentSpec",
    "AssemblyDesignVariable",
    "AssemblyModuleSpec",
    "AVAILABLE_ASSEMBLY_MODULES",
    "BASE_ASSEMBLY_MODULE",
    "BATTERY_ASSEMBLY_MODULE",
    "FINANCE_ASSEMBLY_MODULE",
    "OPERATIONS_ASSEMBLY_MODULE",
    "SOLAR_ASSEMBLY_MODULE",
    "SUPPORTED_WSB_ASSEMBLY_TECHNOLOGIES",
    "WIND_ASSEMBLY_MODULE",
    "compose_assembly_modules",
    "compose_supported_assembly_modules",
    "render_wind_solar_battery_assembly",
    "source_file_hash_targets",
]

ASSEMBLY_MODULES = AVAILABLE_ASSEMBLY_MODULES
