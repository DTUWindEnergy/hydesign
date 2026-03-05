from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from hydesign.modules.ems import (
    default_project_name,
    render_wind_solar_battery_cplex_ems,
)
from hydesign.project_builder.builder import should_build
from hydesign.project_builder.common import (
    collect_generated_files,
    hash_sources,
    request_with_merged_artifacts,
    write_manifest,
    write_package_init,
)
from hydesign.project_builder.contracts import BuildManifest
from hydesign.project_builder.registry import DEFAULT_OUTPUT_ROOT
from hydesign.project_builder.source_hashes import source_hash_targets_for_request


def build_baseline_cplex_ems_project(
    *,
    project_name: str | None = None,
    technologies: tuple[str, ...] | list[str] = ("wind", "solar", "battery"),
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    enable_build: bool = True,
    regenerate: bool = False,
    builder_version: str = "0.2",
) -> Path:
    """Generate the first combined EMS artifact from the original WSB CPLEX source."""

    project_name = project_name or default_project_name()
    request, manifest_data = request_with_merged_artifacts(
        project_name=project_name,
        technologies=technologies,
        requested_artifacts=("ems",),
        output_root=output_root,
        enable_build=enable_build,
        regenerate=regenerate,
        builder_version=builder_version,
    )
    if not should_build(request, manifest_data):
        return request.output_dir

    request.output_dir.mkdir(parents=True, exist_ok=True)

    ems_path = request.output_dir / "ems.py"
    ems_path.write_text(
        render_wind_solar_battery_cplex_ems(request.technologies),
        encoding="utf-8",
    )
    write_package_init(request.output_dir)

    manifest = BuildManifest.from_request(
        request,
        generated_files=collect_generated_files(request.output_dir),
        source_hashes=hash_sources(
            source_hash_targets_for_request(request.artifacts, request.technologies)
        ),
        built_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    write_manifest(request.manifest_path, manifest)
    return request.output_dir
