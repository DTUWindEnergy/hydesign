from __future__ import annotations

from pathlib import Path
from typing import Iterable

from hydesign.modular.modules.assembly.contracts import (
    AssemblyComponentSpec,
    AssemblyModuleSpec,
)
from hydesign.modular.modules.assembly.registry import compose_assembly_modules

SUPPORTED_WSB_ASSEMBLY_TECHNOLOGIES = ("wind", "solar", "battery")
ASSEMBLY_GENERATOR_FILES = (
    "hydesign/modular/modules/assembly/contracts.py",
    "hydesign/modular/modules/assembly/base.py",
    "hydesign/modular/modules/assembly/wind.py",
    "hydesign/modular/modules/assembly/solar.py",
    "hydesign/modular/modules/assembly/battery.py",
    "hydesign/modular/modules/assembly/operations.py",
    "hydesign/modular/modules/assembly/finance.py",
    "hydesign/modular/modules/assembly/registry.py",
    "hydesign/modular/modules/assembly/artifact_codegen.py",
)
OUTPUT_LABELS = (
    "NPV_over_CAPEX",
    "NPV [MEuro]",
    "IRR",
    "LCOE [Euro/MWh]",
    "Revenues [MEuro]",
    "CAPEX [MEuro]",
    "OPEX [MEuro]",
    "Wind CAPEX [MEuro]",
    "Wind OPEX [MEuro]",
    "PV CAPEX [MEuro]",
    "PV OPEX [MEuro]",
    "Batt CAPEX [MEuro]",
    "Batt OPEX [MEuro]",
    "Shared CAPEX [MEuro]",
    "Shared OPEX [MEuro]",
    "penalty lifetime [MEuro]",
    "Mean Annual Electricity Sold [GWh]",
    "GUF",
    "grid [MW]",
    "wind [MW]",
    "solar [MW]",
    "Battery Energy [MWh]",
    "Battery Power [MW]",
    "Total curtailment [GWh]",
    "Total curtailment with deg [GWh]",
    "Awpp [km2]",
    "Apvp [km2]",
    "Plant area [km2]",
    "Rotor diam [m]",
    "Hub height [m]",
    "Number of batteries used in lifetime",
    "Break-even PPA price [Euro/MWh]",
    "Capacity factor wind [-]",
    "AEP [GWh]",
    "AEP with degradation [GWh]",
)
COMPONENT_ORDER = (
    "abl",
    "genericWT",
    "genericWake",
    "wpp",
    "pvp",
    "ems",
    "battery_degradation",
    "battery_loss_in_capacity_due_to_temp",
    "wpp_with_degradation",
    "pvp_with_degradation",
    "battery_with_reliability",
    "wpp_with_reliability",
    "pvp_with_reliability",
    "ems_long_term_operation",
    "wpp_cost",
    "pvp_cost",
    "battery_cost",
    "shared_cost",
    "finance",
)
OUTPUT_TERMS = (
    "prob['NPV_over_CAPEX']",
    "prob['NPV'] / 1e6",
    "prob['IRR']",
    "prob['LCOE']",
    "prob['revenues'] / 1e6",
    "prob['CAPEX'] / 1e6",
    "prob['OPEX'] / 1e6",
    "prob.get_val('finance.CAPEX_w') / 1e6",
    "prob.get_val('finance.OPEX_w') / 1e6",
    "prob.get_val('finance.CAPEX_s') / 1e6",
    "prob.get_val('finance.OPEX_s') / 1e6",
    "prob.get_val('finance.CAPEX_b') / 1e6",
    "prob.get_val('finance.OPEX_b') / 1e6",
    "prob.get_val('finance.CAPEX_el') / 1e6",
    "prob.get_val('finance.OPEX_el') / 1e6",
    "prob['penalty_lifetime'] / 1e6",
    "prob['mean_AEP'] / 1e3",
    "prob['mean_AEP'] / (self.sim_pars['G_MW'] * 365 * 24)",
    "self.sim_pars['G_MW']",
    "wind_MW",
    "solar_MW",
    "b_E",
    "b_P",
    "prob['total_curtailment'] / 1e3",
    "prob['total_curtailment_with_deg'] / 1e3",
    "Awpp",
    "prob.get_val('shared_cost.Apvp')",
    "max(Awpp, prob.get_val('shared_cost.Apvp'))",
    "d",
    "hh",
    "prob.get_val('battery_degradation.n_batteries') * (b_P > 0)",
    "prob['break_even_PPA_price']",
    "cf_wind",
    "AEP",
    "AEP_deg",
)


def _normalize_names(names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for name in names:
        item = name.strip().lower().replace("-", "_")
        if not item or item in seen:
            continue
        normalized.append(item)
        seen.add(item)
    return tuple(normalized)


def _extract_symbol_definitions(
    *,
    source_path: Path,
    symbol_names: tuple[str, ...] | list[str],
) -> dict[str, str]:
    import ast

    source = source_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    definitions: dict[str, str] = {}
    for node in module.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in symbol_names:
                segment = ast.get_source_segment(source, node)
                if segment is None:
                    lines = source.splitlines(keepends=True)
                    segment = "".join(lines[node.lineno - 1 : node.end_lineno])
                definitions[node.name] = (
                    segment if segment.endswith("\n") else segment + "\n"
                )
    missing = [name for name in symbol_names if name not in definitions]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(f"Missing symbols in {source_path}: {missing_str}")
    return definitions


def compose_supported_assembly_modules(
    technologies: tuple[str, ...] | list[str],
) -> tuple[AssemblyModuleSpec, ...]:
    normalized = _normalize_names(technologies)
    if normalized != SUPPORTED_WSB_ASSEMBLY_TECHNOLOGIES:
        supported = ", ".join(SUPPORTED_WSB_ASSEMBLY_TECHNOLOGIES)
        raise NotImplementedError(
            "Phase 1 assembly generation only supports the exact technology set "
            f"({supported}). Requested: {normalized!r}."
        )
    return compose_assembly_modules(normalized)


def _unique_ordered(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def _render_line_block(lines: tuple[str, ...] | list[str], indent: int) -> str:
    prefix = " " * indent
    if not lines:
        return f"{prefix}pass"
    return "\n".join(f"{prefix}{line}" for line in lines)


def _render_expression_lines(
    expr: str, indent: int, *, trailing_comma: bool
) -> list[str]:
    prefix = " " * indent
    lines = expr.splitlines() or [expr]
    rendered = [f"{prefix}{line}" for line in lines]
    if trailing_comma:
        rendered[-1] = rendered[-1] + ","
    return rendered


def _render_component(spec: AssemblyComponentSpec, indent: int = 12) -> str:
    prefix = " " * indent
    lines = [f"{prefix}(", f'{prefix}    "{spec.subsystem_name}",']
    lines.extend(
        _render_expression_lines(spec.constructor_code, indent + 4, trailing_comma=True)
    )
    if spec.key_map_code is not None:
        lines.extend(
            _render_expression_lines(spec.key_map_code, indent + 4, trailing_comma=True)
        )
    lines.append(f"{prefix}),")
    return "\n".join(lines)


def _collect_import_lines(modules: tuple[AssemblyModuleSpec, ...]) -> tuple[str, ...]:
    base_imports = (
        "import datetime",
        "import os",
        "import numpy as np",
        "import openmdao.api as om",
        "import pandas as pd",
        "import xarray as xr",
        "import yaml",
        "from hydesign.look_up_tables import lut_filepath",
    )
    module_imports: list[str] = []
    for module in modules:
        module_imports.extend(module.import_lines)
    return tuple(_unique_ordered(base_imports + tuple(module_imports)))


def _collect_design_variable_names(
    modules: tuple[AssemblyModuleSpec, ...],
) -> tuple[str, ...]:
    names: list[str] = []
    for module in modules:
        names.extend(variable.name for variable in module.design_variables)
    return tuple(_unique_ordered(names))


def _collect_design_variable_labels(
    modules: tuple[AssemblyModuleSpec, ...],
) -> tuple[str, ...]:
    labels: list[str] = []
    for module in modules:
        labels.extend(variable.label for variable in module.design_variables)
    return tuple(labels)


def _collect_component_specs(
    modules: tuple[AssemblyModuleSpec, ...],
) -> tuple[AssemblyComponentSpec, ...]:
    specs_by_name: dict[str, AssemblyComponentSpec] = {}
    for module in modules:
        for spec in module.component_specs:
            specs_by_name[spec.subsystem_name] = spec
    missing = [name for name in COMPONENT_ORDER if name not in specs_by_name]
    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(
            "Assembly component ordering is missing registered subsystem specs: "
            f"{missing_str}"
        )
    return tuple(specs_by_name[name] for name in COMPONENT_ORDER)


def _collect_lines(
    modules: tuple[AssemblyModuleSpec, ...], attr_name: str
) -> tuple[str, ...]:
    lines: list[str] = []
    for module in modules:
        lines.extend(getattr(module, attr_name))
    return tuple(lines)


def _render_signature(names: tuple[str, ...], indent: int) -> str:
    prefix = " " * indent
    return "\n".join(f"{prefix}{name}," for name in names)


def _render_string_list(values: tuple[str, ...], indent: int) -> str:
    prefix = " " * indent
    return "\n".join(f"{prefix}{value!r}," for value in values)


def _render_output_terms(values: tuple[str, ...], indent: int) -> str:
    lines: list[str] = []
    for value in values:
        lines.extend(_render_expression_lines(value, indent, trailing_comma=True))
    return "\n".join(lines)


def _source_summary(modules: tuple[AssemblyModuleSpec, ...]) -> str:
    refs: list[str] = []
    for module in modules:
        refs.extend(module.source_of_truth)
    return ", ".join(_unique_ordered(refs))


def source_file_hash_targets(
    technologies: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    modules = compose_supported_assembly_modules(technologies)
    paths: list[str] = []
    for module in modules:
        for ref in module.source_of_truth:
            path, _, _ = ref.partition(":")
            paths.append(path)
    return tuple(_unique_ordered(tuple(paths) + ASSEMBLY_GENERATOR_FILES))


def render_wind_solar_battery_assembly(
    technologies: tuple[str, ...] | list[str],
) -> str:
    modules = compose_supported_assembly_modules(technologies)
    definitions = _extract_symbol_definitions(
        source_path=Path("hydesign") / "assembly" / "hpp_assembly.py",
        symbol_names=("hpp_base", "mkdir"),
    )

    import_lines = _collect_import_lines(modules)
    design_var_names = _collect_design_variable_names(modules)
    design_var_labels = _collect_design_variable_labels(modules)
    component_specs = _collect_component_specs(modules)
    init_setup_lines = _collect_lines(modules, "init_setup_lines")
    init_set_val_lines = _collect_lines(modules, "init_set_val_lines")
    evaluate_set_val_lines = _collect_lines(modules, "evaluate_set_val_lines")
    evaluate_setup_lines = _collect_lines(modules, "evaluate_setup_lines")
    evaluate_post_run_lines = _collect_lines(modules, "evaluate_post_run_lines")
    rendered_components = "\n".join(_render_component(spec) for spec in component_specs)
    rendered_output_labels = _render_string_list(OUTPUT_LABELS, indent=12)
    rendered_output_terms = _render_output_terms(OUTPUT_TERMS, indent=16)
    source_summary = _source_summary(modules)

    return f'''# Auto-generated combined assembly module
# Combined from original assembly source of truth: {source_summary}

{chr(10).join(import_lines)}


{definitions["hpp_base"].rstrip()}


class hpp_model(hpp_base):
    """HPP design evaluator built from modular assembly contributions."""

    def __init__(self, sim_pars_fn, **kwargs):
        hpp_base.__init__(self, sim_pars_fn=sim_pars_fn, **kwargs)

        N_time = self.N_time
        N_ws = self.N_ws
        wpp_efficiency = self.wpp_efficiency
        sim_pars = self.sim_pars
        life_y = self.life_y
        wind_deg_yr = self.wind_deg_yr
        wind_deg = self.wind_deg
        share_WT_deg_types = self.share_WT_deg_types
        price = self.price

        input_ts_fn = sim_pars["input_ts_fn"]
        genWT_fn = sim_pars["genWT_fn"]
        genWake_fn = sim_pars["genWake_fn"]
        latitude = sim_pars["latitude"]
        longitude = sim_pars["longitude"]
        altitude = sim_pars["altitude"]
        weeks_per_season_per_year = sim_pars["weeks_per_season_per_year"]
        ems_type = sim_pars["ems_type"]
        max_num_batteries_allowed = sim_pars["max_num_batteries_allowed"]
        reliability_ts_battery = sim_pars["reliability_ts_battery"]
        reliability_ts_trans = sim_pars["reliability_ts_trans"]
        reliability_ts_wind = sim_pars["reliability_ts_wind"]
        reliability_ts_pv = sim_pars["reliability_ts_pv"]
        battery_price_reduction_per_year = sim_pars["battery_price_reduction_per_year"]
        work_dir = sim_pars["work_dir"]

{_render_line_block(init_setup_lines, indent=8)}

        comps = [
{rendered_components}
        ]

        prob = self.get_prob(comps)
        prob.setup()

{_render_line_block(init_set_val_lines, indent=8)}

        self.prob = prob
        self.list_out_vars = [
{rendered_output_labels}
        ]
        self.list_vars = [
{_render_string_list(design_var_labels, indent=12)}
        ]

    def evaluate(
        self,
{_render_signature(design_var_names, indent=8)}
    ):
        self.inputs = [
{_render_signature(design_var_names, indent=12)}
        ]

        prob = self.prob

{_render_line_block(evaluate_setup_lines, indent=8)}

{_render_line_block(evaluate_set_val_lines, indent=8)}

        prob.run_model()
        self.prob = prob

{_render_line_block(evaluate_post_run_lines, indent=8)}

        outputs = np.hstack(
            [
{rendered_output_terms}
            ]
        )
        self.outputs = outputs
        return outputs


{definitions["mkdir"].rstrip()}


__all__ = ["hpp_base", "hpp_model", "mkdir"]
'''
