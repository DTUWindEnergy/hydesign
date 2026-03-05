from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from hydesign.project_builder.assembly import build_baseline_wsb_assembly_project
from hydesign.project_builder.battery_degradation import (
    build_baseline_wsb_battery_degradation_project,
)
from hydesign.project_builder.contracts import BuildRequest
from hydesign.project_builder.costs import build_baseline_wsb_costs_project
from hydesign.project_builder.ems import build_baseline_cplex_ems_project
from hydesign.project_builder.ems_long_term import (
    build_baseline_wsb_ems_long_term_project,
)
from hydesign.project_builder.finance import build_baseline_wsb_finance_project
from hydesign.project_builder.pv import build_baseline_wsb_pv_project
from hydesign.project_builder.registry import DEFAULT_OUTPUT_ROOT
from hydesign.project_builder.reliability import build_baseline_wsb_reliability_project
from hydesign.project_builder.weather import build_baseline_wsb_weather_project
from hydesign.project_builder.wind import build_baseline_wsb_wind_project

WSB_TECHNOLOGIES = ("wind", "solar", "battery")
WSB_FULL_ARTIFACTS = (
    "weather",
    "wind",
    "pv",
    "ems",
    "ems_long_term",
    "battery_degradation",
    "reliability",
    "costs",
    "finance",
    "assembly",
)

WSB_ARTIFACT_DEPENDENCIES = {
    "weather": (),
    "wind": ("ems",),
    "pv": (),
    "ems": (),
    "ems_long_term": (),
    "battery_degradation": ("ems",),
    "reliability": (),
    "costs": (),
    "finance": (),
    "assembly": (
        "weather",
        "wind",
        "pv",
        "ems",
        "ems_long_term",
        "battery_degradation",
        "reliability",
        "costs",
        "finance",
    ),
}

WSB_ARTIFACT_BUILDERS = {
    "weather": build_baseline_wsb_weather_project,
    "wind": build_baseline_wsb_wind_project,
    "pv": build_baseline_wsb_pv_project,
    "ems": build_baseline_cplex_ems_project,
    "ems_long_term": build_baseline_wsb_ems_long_term_project,
    "battery_degradation": build_baseline_wsb_battery_degradation_project,
    "reliability": build_baseline_wsb_reliability_project,
    "costs": build_baseline_wsb_costs_project,
    "finance": build_baseline_wsb_finance_project,
    "assembly": build_baseline_wsb_assembly_project,
}


def _unique_ordered(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip().lower().replace("-", "_")
        if not item or item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return tuple(ordered)


def _expand_wsb_artifacts(
    requested_artifacts: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    required: set[str] = set()

    def visit(name: str) -> None:
        try:
            dependencies = WSB_ARTIFACT_DEPENDENCIES[name]
        except KeyError as exc:
            raise KeyError(f"Unknown WSB artifact {name!r}.") from exc
        for dependency in dependencies:
            visit(dependency)
        required.add(name)

    for artifact_name in _unique_ordered(requested_artifacts):
        visit(artifact_name)
    return tuple(name for name in WSB_FULL_ARTIFACTS if name in required)


@dataclass(frozen=True)
class ProjectBuildSpec:
    technologies: tuple[str, ...] | list[str]
    project_name: str = ""
    artifacts: tuple[str, ...] | list[str] | None = None
    enable_build: bool = True
    regenerate: bool = False
    output_root: Path | str = DEFAULT_OUTPUT_ROOT
    builder_version: str = "0.2"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectBuildSpec":
        allowed_keys = {
            "technologies",
            "project_name",
            "artifacts",
            "enable_build",
            "regenerate",
            "output_root",
            "builder_version",
        }
        unknown = sorted(set(data) - allowed_keys)
        if unknown:
            unknown_str = ", ".join(unknown)
            raise ValueError(f"Unknown project build spec keys: {unknown_str}")
        if "technologies" not in data:
            raise ValueError("Project build spec requires 'technologies'.")
        return cls(
            technologies=data["technologies"],
            project_name=data.get("project_name", ""),
            artifacts=data.get("artifacts"),
            enable_build=data.get("enable_build", True),
            regenerate=data.get("regenerate", False),
            output_root=data.get("output_root", DEFAULT_OUTPUT_ROOT),
            builder_version=str(data.get("builder_version", "0.2")),
        )

    def resolved_artifacts(self) -> tuple[str, ...]:
        technologies = self.normalized_technologies()
        requested = _unique_ordered(self.artifacts or WSB_FULL_ARTIFACTS)
        if technologies == WSB_TECHNOLOGIES:
            return _expand_wsb_artifacts(requested)
        tech_str = ", ".join(technologies)
        raise NotImplementedError(
            "Public project building is currently implemented only for the "
            f"wind+solar+battery baseline. Requested: ({tech_str})."
        )

    def normalized_technologies(self) -> tuple[str, ...]:
        return BuildRequest(
            project_name=self.project_name or "_spec_probe",
            technologies=self.technologies,
            artifacts=("assembly",),
            output_root=self.output_root,
            enable_build=self.enable_build,
            regenerate=self.regenerate,
            builder_version=self.builder_version,
        ).technologies

    def to_build_request(self) -> BuildRequest:
        return BuildRequest(
            project_name=self.project_name,
            technologies=self.technologies,
            artifacts=self.resolved_artifacts(),
            output_root=self.output_root,
            enable_build=self.enable_build,
            regenerate=self.regenerate,
            builder_version=self.builder_version,
        )


def load_project_build_spec(
    spec: ProjectBuildSpec | Mapping[str, Any] | Path | str,
) -> ProjectBuildSpec:
    if isinstance(spec, ProjectBuildSpec):
        return spec
    if isinstance(spec, Mapping):
        return ProjectBuildSpec.from_dict(spec)

    path = Path(spec)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text) or {}
    if not isinstance(data, Mapping):
        raise ValueError("Project build spec file must contain a mapping.")
    return ProjectBuildSpec.from_dict(data)


def build_project(
    spec: ProjectBuildSpec | Mapping[str, Any] | Path | str,
) -> Path:
    project_spec = load_project_build_spec(spec)
    request = project_spec.to_build_request()

    if request.technologies != WSB_TECHNOLOGIES:
        tech_str = ", ".join(request.technologies)
        raise NotImplementedError(
            "Public project building is currently implemented only for the "
            f"wind+solar+battery baseline. Requested: ({tech_str})."
        )

    common_kwargs = {
        "project_name": request.project_name,
        "technologies": request.technologies,
        "output_root": request.output_root,
        "enable_build": request.enable_build,
        "regenerate": request.regenerate,
        "builder_version": request.builder_version,
    }
    for artifact_name in request.artifacts:
        builder = WSB_ARTIFACT_BUILDERS[artifact_name]
        if artifact_name == "assembly":
            builder(build_dependencies=False, **common_kwargs)
        else:
            builder(**common_kwargs)
    return request.output_dir
