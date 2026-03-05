from __future__ import annotations

from hydesign.modules.assembly import (
    source_file_hash_targets as assembly_source_file_hash_targets,
)
from hydesign.modules.battery_degradation import (
    source_file_hash_targets as battery_degradation_source_file_hash_targets,
)
from hydesign.modules.costs import (
    compose_supported_costs_modules,
)
from hydesign.modules.costs import (
    source_file_hash_targets as costs_source_file_hash_targets,
)
from hydesign.modules.ems import (
    compose_supported_cplex_modules,
)
from hydesign.modules.ems import (
    source_file_hash_targets as ems_source_file_hash_targets,
)
from hydesign.modules.ems_long_term import (
    source_file_hash_targets as ems_long_term_source_file_hash_targets,
)
from hydesign.modules.finance import (
    compose_supported_finance_modules,
)
from hydesign.modules.finance import (
    source_file_hash_targets as finance_source_file_hash_targets,
)
from hydesign.modules.pv import source_file_hash_targets as pv_source_file_hash_targets
from hydesign.modules.reliability import (
    source_file_hash_targets as reliability_source_file_hash_targets,
)
from hydesign.modules.weather import (
    source_file_hash_targets as weather_source_file_hash_targets,
)
from hydesign.modules.wind import (
    source_file_hash_targets as wind_source_file_hash_targets,
)


def _unique_ordered(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return tuple(ordered)


def source_hash_targets_for_artifact(
    artifact_name: str,
    technologies: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    if artifact_name == "assembly":
        return assembly_source_file_hash_targets(technologies)
    if artifact_name == "weather":
        return weather_source_file_hash_targets(technologies)
    if artifact_name == "wind":
        return wind_source_file_hash_targets(technologies)
    if artifact_name == "pv":
        return pv_source_file_hash_targets(technologies)
    if artifact_name == "ems":
        return ems_source_file_hash_targets(
            compose_supported_cplex_modules(technologies)
        )
    if artifact_name == "ems_long_term":
        return ems_long_term_source_file_hash_targets(technologies)
    if artifact_name == "battery_degradation":
        return battery_degradation_source_file_hash_targets(technologies)
    if artifact_name == "reliability":
        return reliability_source_file_hash_targets(technologies)
    if artifact_name == "costs":
        return costs_source_file_hash_targets(
            compose_supported_costs_modules(technologies)
        )
    if artifact_name == "finance":
        return finance_source_file_hash_targets(
            compose_supported_finance_modules(technologies)
        )
    raise NotImplementedError(
        f"Source hash target resolution is not implemented for artifact {artifact_name!r}."
    )


def source_hash_targets_for_request(
    artifacts: tuple[str, ...] | list[str],
    technologies: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    targets: list[str] = []
    for artifact_name in artifacts:
        targets.extend(source_hash_targets_for_artifact(artifact_name, technologies))
    return _unique_ordered(targets)
