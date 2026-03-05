"""Wind artifact generation for the baseline WSB combined project."""

from hydesign.modules.wind.artifact_codegen import (
    SUPPORTED_WSB_WIND_TECHNOLOGIES,
    render_wind_solar_battery_wind,
    source_file_hash_targets,
)

__all__ = [
    "SUPPORTED_WSB_WIND_TECHNOLOGIES",
    "render_wind_solar_battery_wind",
    "source_file_hash_targets",
]
