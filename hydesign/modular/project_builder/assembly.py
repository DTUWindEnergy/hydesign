from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from hydesign.modular.modules.assembly import (
    render_wind_solar_battery_assembly,
)
from hydesign.modular.modules.ems import default_project_name
from hydesign.modular.project_builder.battery_degradation import (
    build_baseline_wsb_battery_degradation_project,
)
from hydesign.modular.project_builder.builder import should_build
from hydesign.modular.project_builder.common import (
    build_generated_artifact,
    request_with_merged_artifacts,
    write_manifest,
)
from hydesign.modular.project_builder.costs import build_baseline_wsb_costs_project
from hydesign.modular.project_builder.ems import build_baseline_cplex_ems_project
from hydesign.modular.project_builder.ems_long_term import (
    build_baseline_wsb_ems_long_term_project,
)
from hydesign.modular.project_builder.finance import build_baseline_wsb_finance_project
from hydesign.modular.project_builder.pv import build_baseline_wsb_pv_project
from hydesign.modular.project_builder.registry import DEFAULT_OUTPUT_ROOT
from hydesign.modular.project_builder.reliability import (
    build_baseline_wsb_reliability_project,
)
from hydesign.modular.project_builder.weather import build_baseline_wsb_weather_project
from hydesign.modular.project_builder.wind import build_baseline_wsb_wind_project


def build_baseline_wsb_assembly_project(
    *,
    project_name: str | None = None,
    technologies: tuple[str, ...] | list[str] = ("wind", "solar", "battery"),
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    enable_build: bool = True,
    regenerate: bool = False,
    builder_version: str = "0.2",
    build_dependencies: bool = True,
) -> Path:
    project_name = project_name or default_project_name()

    if build_dependencies:
        # Assembly depends on the already-generated local WSB support modules.
        build_baseline_cplex_ems_project(
            project_name=project_name,
            technologies=technologies,
            output_root=output_root,
            enable_build=enable_build,
            regenerate=regenerate,
            builder_version=builder_version,
        )
        build_baseline_wsb_costs_project(
            project_name=project_name,
            technologies=technologies,
            output_root=output_root,
            enable_build=enable_build,
            regenerate=regenerate,
            builder_version=builder_version,
        )
        build_baseline_wsb_finance_project(
            project_name=project_name,
            technologies=technologies,
            output_root=output_root,
            enable_build=enable_build,
            regenerate=regenerate,
            builder_version=builder_version,
        )
        build_baseline_wsb_weather_project(
            project_name=project_name,
            technologies=technologies,
            output_root=output_root,
            enable_build=enable_build,
            regenerate=regenerate,
            builder_version=builder_version,
        )
        build_baseline_wsb_wind_project(
            project_name=project_name,
            technologies=technologies,
            output_root=output_root,
            enable_build=enable_build,
            regenerate=regenerate,
            builder_version=builder_version,
        )
        build_baseline_wsb_pv_project(
            project_name=project_name,
            technologies=technologies,
            output_root=output_root,
            enable_build=enable_build,
            regenerate=regenerate,
            builder_version=builder_version,
        )
        build_baseline_wsb_battery_degradation_project(
            project_name=project_name,
            technologies=technologies,
            output_root=output_root,
            enable_build=enable_build,
            regenerate=regenerate,
            builder_version=builder_version,
        )
        build_baseline_wsb_ems_long_term_project(
            project_name=project_name,
            technologies=technologies,
            output_root=output_root,
            enable_build=enable_build,
            regenerate=regenerate,
            builder_version=builder_version,
        )
        build_baseline_wsb_reliability_project(
            project_name=project_name,
            technologies=technologies,
            output_root=output_root,
            enable_build=enable_build,
            regenerate=regenerate,
            builder_version=builder_version,
        )

    request, manifest_data = request_with_merged_artifacts(
        project_name=project_name,
        technologies=technologies,
        requested_artifacts=("assembly",),
        output_root=output_root,
        enable_build=enable_build,
        regenerate=regenerate,
        builder_version=builder_version,
    )
    if not should_build(request, manifest_data):
        return request.output_dir

    manifest = build_generated_artifact(
        request=request,
        output_filename="assembly.py",
        rendered_text=render_wind_solar_battery_assembly(request.technologies),
        built_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_manifest(request.manifest_path, manifest)
    return request.output_dir
