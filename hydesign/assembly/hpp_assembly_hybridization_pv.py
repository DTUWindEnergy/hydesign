"""Compatibility wrapper for the legacy PV-hybridization assembly.

This module now delegates to the unified base assembly implementation.
"""

import warnings

from hydesign.assembly.hpp_assembly import hpp_model as _base_hpp_model


class hpp_model(_base_hpp_model):
    """Backward-compatible alias to the unified base `hpp_model`.

    Use technology-specific operation year inputs (`wind_operation_year`,
    `solar_operation_year`, `battery_operation_year`) instead of legacy
    hybridization-specific parameters.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "`hpp_assembly_hybridization_pv.hpp_model` is deprecated and now "
            "uses `hpp_assembly.hpp_model`. Pass operation-year inputs per "
            "technology (e.g. `wind_operation_year`, `solar_operation_year`).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
