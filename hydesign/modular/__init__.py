"""Public entrypoint for the modular HyDesign implementation."""

from hydesign.modular.project_builder import (
    DEFAULT_ARTIFACT_REGISTRY,
    DEFAULT_ARTIFACTS,
    DEFAULT_OUTPUT_ROOT,
    ArtifactContract,
    BuildManifest,
    BuildRequest,
    ProjectBuildSpec,
    build_baseline_cplex_ems_project,
    build_baseline_wsb_assembly_project,
    build_baseline_wsb_costs_project,
    build_baseline_wsb_finance_project,
    build_project,
    load_project_build_spec,
    manifest_matches_request,
    should_build,
)

__all__ = [
    "ArtifactContract",
    "BuildManifest",
    "BuildRequest",
    "DEFAULT_ARTIFACTS",
    "DEFAULT_ARTIFACT_REGISTRY",
    "DEFAULT_OUTPUT_ROOT",
    "ProjectBuildSpec",
    "build_baseline_cplex_ems_project",
    "build_baseline_wsb_assembly_project",
    "build_baseline_wsb_costs_project",
    "build_baseline_wsb_finance_project",
    "build_project",
    "load_project_build_spec",
    "manifest_matches_request",
    "should_build",
]
