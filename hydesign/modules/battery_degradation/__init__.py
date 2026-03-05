"""Battery degradation artifact generation for the baseline WSB combined project."""

from hydesign.modules.battery_degradation.artifact_codegen import (
    SUPPORTED_WSB_BATTERY_DEGRADATION_TECHNOLOGIES,
    render_wind_solar_battery_battery_degradation,
    source_file_hash_targets,
)

__all__ = [
    "SUPPORTED_WSB_BATTERY_DEGRADATION_TECHNOLOGIES",
    "render_wind_solar_battery_battery_degradation",
    "source_file_hash_targets",
]
