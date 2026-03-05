from __future__ import annotations

from pathlib import Path

from hydesign.project_builder.contracts import ArtifactContract

DEFAULT_OUTPUT_ROOT = Path("hydesign") / "combined_projects"

DEFAULT_ARTIFACT_REGISTRY = {
    "weather": ArtifactContract(
        name="weather",
        output_filename="weather.py",
        source_package="hydesign.modules.weather",
        entry_symbol="ABL_comp",
        description="Weather extraction, interpolation, and representative-year helpers.",
    ),
    "wind": ArtifactContract(
        name="wind",
        output_filename="wind.py",
        source_package="hydesign.modules.wind",
        entry_symbol="wpp_comp",
        description="Wind surrogate, production, and degradation helpers.",
    ),
    "pv": ArtifactContract(
        name="pv",
        output_filename="pv.py",
        source_package="hydesign.modules.pv",
        entry_symbol="pvp_comp",
        description="PV production and degradation helpers.",
    ),
    "ems": ArtifactContract(
        name="ems",
        output_filename="ems.py",
        source_package="hydesign.modules.ems",
        entry_symbol="ems_comp",
        description="Dispatch and operational optimization layer.",
    ),
    "ems_long_term": ArtifactContract(
        name="ems_long_term",
        output_filename="ems_long_term.py",
        source_package="hydesign.modules.ems_long_term",
        entry_symbol="ems_long_term_operation_comp",
        description="Long-term degraded operation and penalty propagation layer.",
    ),
    "battery_degradation": ArtifactContract(
        name="battery_degradation",
        output_filename="battery_degradation.py",
        source_package="hydesign.modules.battery_degradation",
        entry_symbol="battery_degradation_comp",
        description="Battery degradation and temperature-loss helpers.",
    ),
    "reliability": ArtifactContract(
        name="reliability",
        output_filename="reliability.py",
        source_package="hydesign.modules.reliability",
        entry_symbol="reliability_comp",
        description="Reliability and degradation extensions for generated projects.",
    ),
    "costs": ArtifactContract(
        name="costs",
        output_filename="costs.py",
        source_package="hydesign.modules.costs",
        entry_symbol="costs_comp",
        description="CAPEX and OPEX calculations for the selected technologies.",
    ),
    "finance": ArtifactContract(
        name="finance",
        output_filename="finance.py",
        source_package="hydesign.modules.finance",
        entry_symbol="finance_comp",
        description="Project economics derived from production and cost outputs.",
    ),
    "assembly": ArtifactContract(
        name="assembly",
        output_filename="assembly.py",
        source_package="hydesign.modules.assembly",
        entry_symbol="hpp_model",
        description="Top-level OpenMDAO assembly and public evaluate API.",
    ),
}

DEFAULT_ARTIFACTS = tuple(DEFAULT_ARTIFACT_REGISTRY.keys())
