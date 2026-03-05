from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssemblyDesignVariable:
    name: str
    label: str


@dataclass(frozen=True)
class AssemblyComponentSpec:
    subsystem_name: str
    constructor_code: str
    key_map_code: str | None = None


@dataclass(frozen=True)
class AssemblyModuleSpec:
    name: str
    description: str
    source_of_truth: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    import_lines: tuple[str, ...] = ()
    init_setup_lines: tuple[str, ...] = ()
    component_specs: tuple[AssemblyComponentSpec, ...] = ()
    init_set_val_lines: tuple[str, ...] = ()
    evaluate_set_val_lines: tuple[str, ...] = ()
    design_variables: tuple[AssemblyDesignVariable, ...] = ()
    evaluate_setup_lines: tuple[str, ...] = ()
    evaluate_post_run_lines: tuple[str, ...] = ()
    output_labels: tuple[str, ...] = ()
    output_terms: tuple[str, ...] = ()
