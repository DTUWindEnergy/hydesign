from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from hydesign.modular.modules.ems import default_project_name
from hydesign.modular.modules.weather import render_wind_solar_battery_weather
from hydesign.modular.project_builder.builder import should_build
from hydesign.modular.project_builder.common import (
    build_generated_artifact,
    request_with_merged_artifacts,
    write_manifest,
)
from hydesign.modular.project_builder.registry import DEFAULT_OUTPUT_ROOT


def build_baseline_wsb_weather_project(
    *,
    project_name: str | None = None,
    technologies: tuple[str, ...] | list[str] = ("wind", "solar", "battery"),
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    enable_build: bool = True,
    regenerate: bool = False,
    builder_version: str = "0.2",
) -> Path:
    project_name = project_name or default_project_name()
    request, manifest_data = request_with_merged_artifacts(
        project_name=project_name,
        technologies=technologies,
        requested_artifacts=("weather",),
        output_root=output_root,
        enable_build=enable_build,
        regenerate=regenerate,
        builder_version=builder_version,
    )
    if not should_build(request, manifest_data):
        return request.output_dir

    manifest = build_generated_artifact(
        request=request,
        output_filename="weather.py",
        rendered_text=render_wind_solar_battery_weather(request.technologies),
        built_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_manifest(request.manifest_path, manifest)
    return request.output_dir
