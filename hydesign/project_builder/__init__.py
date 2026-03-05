"""Contracts and helpers for explicit combined-project builds."""

from hydesign.project_builder.assembly import build_baseline_wsb_assembly_project
from hydesign.project_builder.builder import manifest_matches_request, should_build
from hydesign.project_builder.contracts import (
    ArtifactContract,
    BuildManifest,
    BuildRequest,
)
from hydesign.project_builder.costs import build_baseline_wsb_costs_project
from hydesign.project_builder.ems import build_baseline_cplex_ems_project
from hydesign.project_builder.finance import build_baseline_wsb_finance_project
from hydesign.project_builder.project import (
    ProjectBuildSpec,
    build_project,
    load_project_build_spec,
)
from hydesign.project_builder.registry import (
    DEFAULT_ARTIFACT_REGISTRY,
    DEFAULT_ARTIFACTS,
    DEFAULT_OUTPUT_ROOT,
)

__all__ = [
    "ArtifactContract",
    "BuildManifest",
    "BuildRequest",
    "DEFAULT_ARTIFACTS",
    "DEFAULT_ARTIFACT_REGISTRY",
    "DEFAULT_OUTPUT_ROOT",
    "ProjectBuildSpec",
    "build_baseline_wsb_assembly_project",
    "build_baseline_cplex_ems_project",
    "build_baseline_wsb_costs_project",
    "build_baseline_wsb_finance_project",
    "build_project",
    "load_project_build_spec",
    "manifest_matches_request",
    "should_build",
]
