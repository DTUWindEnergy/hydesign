"""EMS module contracts and built-in technology metadata."""

from hydesign.modular.modules.ems.artifact_codegen import (
    SUPPORTED_WSB_CPLEX_TECHNOLOGIES,
    compose_supported_cplex_modules,
    default_project_name,
    generated_package_path,
    project_package_init_text,
    render_wind_solar_battery_cplex_ems,
    source_file_hash_targets,
)
from hydesign.modular.modules.ems.base import BASE_EMS_MODULE
from hydesign.modular.modules.ems.battery import BATTERY_EMS_MODULE
from hydesign.modular.modules.ems.contracts import EMSExecutionStage, EMSModuleSpec
from hydesign.modular.modules.ems.p2x import P2X_EMS_MODULE
from hydesign.modular.modules.ems.registry import (
    AVAILABLE_EMS_MODULES,
    compose_ems_modules,
)
from hydesign.modular.modules.ems.solar import SOLAR_EMS_MODULE
from hydesign.modular.modules.ems.wind import WIND_EMS_MODULE

__all__ = [
    "AVAILABLE_EMS_MODULES",
    "BASE_EMS_MODULE",
    "BATTERY_EMS_MODULE",
    "EMSExecutionStage",
    "EMSModuleSpec",
    "P2X_EMS_MODULE",
    "SOLAR_EMS_MODULE",
    "SUPPORTED_WSB_CPLEX_TECHNOLOGIES",
    "WIND_EMS_MODULE",
    "compose_ems_modules",
    "compose_supported_cplex_modules",
    "default_project_name",
    "generated_package_path",
    "project_package_init_text",
    "render_wind_solar_battery_cplex_ems",
    "source_file_hash_targets",
]
