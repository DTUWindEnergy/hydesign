from __future__ import annotations

from pathlib import Path
from typing import Iterable

from hydesign.modules.ems.artifact_contracts import EMSArtifactModule, EMSPortSpec
from hydesign.modules.ems.base import BASE_EMS_ARTIFACT
from hydesign.modules.ems.battery import BATTERY_EMS_ARTIFACT
from hydesign.modules.ems.solar import SOLAR_EMS_ARTIFACT
from hydesign.modules.ems.wind import WIND_EMS_ARTIFACT

AVAILABLE_EMS_ARTIFACT_MODULES = {
    "base": BASE_EMS_ARTIFACT,
    "wind": WIND_EMS_ARTIFACT,
    "solar": SOLAR_EMS_ARTIFACT,
    "battery": BATTERY_EMS_ARTIFACT,
}

SUPPORTED_WSB_CPLEX_TECHNOLOGIES = ("wind", "solar", "battery")

BASELINE_INPUT_ORDER = (
    "wind_t",
    "solar_t",
    "price_t",
    "b_P",
    "b_E",
    "G_MW",
    "battery_depth_of_discharge",
    "battery_charge_efficiency",
    "peak_hr_quantile",
    "cost_of_battery_P_fluct_in_peak_price_ratio",
    "n_full_power_hours_expected_per_day_at_peak_price",
)

BASELINE_OUTPUT_ORDER = (
    "wind_t_ext",
    "solar_t_ext",
    "price_t_ext",
    "hpp_t",
    "hpp_curt_t",
    "b_t",
    "b_E_SOC_t",
    "penalty_t",
)

BASELINE_SOLVER_PARAM_ORDER = (
    "wind_ts",
    "solar_ts",
    "price_ts",
    "P_batt_MW",
    "E_batt_MWh_t",
    "hpp_grid_connection",
    "battery_depth_of_discharge",
    "charge_efficiency",
    "peak_hr_quantile",
    "cost_of_battery_P_fluct_in_peak_price_ratio",
    "n_full_power_hours_expected_per_day_at_peak_price",
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


def compose_supported_cplex_modules(
    technologies: tuple[str, ...] | list[str],
) -> tuple[EMSArtifactModule, ...]:
    names = _normalize_names(technologies)
    if names != SUPPORTED_WSB_CPLEX_TECHNOLOGIES:
        supported = ", ".join(SUPPORTED_WSB_CPLEX_TECHNOLOGIES)
        raise NotImplementedError(
            "Phase 1 EMS generation only supports the exact technology set "
            f"({supported}). Requested: {names!r}."
        )
    return (
        AVAILABLE_EMS_ARTIFACT_MODULES["base"],
        AVAILABLE_EMS_ARTIFACT_MODULES["wind"],
        AVAILABLE_EMS_ARTIFACT_MODULES["solar"],
        AVAILABLE_EMS_ARTIFACT_MODULES["battery"],
    )


def _unique_ordered(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def _ordered_ports(
    modules: tuple[EMSArtifactModule, ...],
    *,
    port_type: str,
    order: tuple[str, ...],
) -> list[EMSPortSpec]:
    ports_by_name: dict[str, EMSPortSpec] = {}
    for module in modules:
        ports = (
            module.component_inputs
            if port_type == "input"
            else module.component_outputs
        )
        for port in ports:
            ports_by_name[port.name] = port
    return [ports_by_name[name] for name in order]


def _ordered_call_args(
    modules: tuple[EMSArtifactModule, ...],
    *,
    attr_name: str,
    param_order: tuple[str, ...],
) -> list[str]:
    args_by_name: dict[str, str] = {}
    for module in modules:
        for item in getattr(module, attr_name):
            name, _, expression = item.partition("=")
            args_by_name[name.strip()] = expression.strip()
    return [f"{name}={args_by_name[name]}" for name in param_order]


def _ordered_lines(modules: tuple[EMSArtifactModule, ...], attr_name: str) -> list[str]:
    lines: list[str] = []
    for module in modules:
        lines.extend(getattr(module, attr_name))
    return _unique_ordered(lines)


def _render_ports(ports: list[EMSPortSpec], indent: int) -> str:
    prefix = " " * indent
    return "\n".join(f"{prefix}({port.name!r}, {port.kwargs_code})," for port in ports)


def _render_lines(lines: list[str], indent: int) -> str:
    prefix = " " * indent
    if not lines:
        return f"{prefix}pass"
    return "\n".join(f"{prefix}{line}" for line in lines)


def _render_items(lines: list[str], indent: int) -> str:
    prefix = " " * indent
    return "\n".join(f"{prefix}{line}," for line in lines)


def _render_signature(names: tuple[str, ...], indent: int) -> str:
    prefix = " " * indent
    return "\n".join(f"{prefix}{name}," for name in names)


def _render_function_signature(
    params: tuple[str, ...],
    *,
    defaults: dict[str, str] | None = None,
    indent: int = 4,
) -> str:
    defaults = defaults or {}
    prefix = " " * indent
    rendered: list[str] = []
    for name in params:
        if name in defaults:
            rendered.append(f"{prefix}{name}={defaults[name]},")
        else:
            rendered.append(f"{prefix}{name},")
    return "\n".join(rendered)


def _render_source_summary(modules: tuple[EMSArtifactModule, ...]) -> str:
    refs: list[str] = []
    for module in modules:
        refs.extend(module.spec.source_of_truth)
    return ", ".join(_unique_ordered(refs))


def source_file_hash_targets(
    modules: tuple[EMSArtifactModule, ...],
) -> tuple[str, ...]:
    paths: list[str] = []
    for module in modules:
        for ref in module.spec.source_of_truth:
            path, _, _ = ref.partition(":")
            paths.append(path)
    return tuple(_unique_ordered(paths))


def render_wind_solar_battery_cplex_ems(
    technologies: tuple[str, ...] | list[str],
) -> str:
    modules = compose_supported_cplex_modules(technologies)
    input_ports = _ordered_ports(modules, port_type="input", order=BASELINE_INPUT_ORDER)
    output_ports = _ordered_ports(
        modules, port_type="output", order=BASELINE_OUTPUT_ORDER
    )

    compute_prepare_lines = _ordered_lines(modules, "compute_prepare_lines")
    dataframe_assignment_lines = _ordered_lines(modules, "dataframe_assignment_lines")
    batch_array_lines = _ordered_lines(modules, "batch_array_lines")
    batch_slice_lines = _ordered_lines(modules, "batch_slice_lines")
    batch_assign_lines = _ordered_lines(modules, "batch_assign_lines")
    finalize_batch_lines = _ordered_lines(modules, "finalize_batch_lines")
    cplex_setup_lines = _ordered_lines(modules, "cplex_setup_lines")
    pre_loop_constraint_lines = _ordered_lines(modules, "pre_loop_constraint_lines")
    loop_constraint_lines = _ordered_lines(modules, "loop_constraint_lines")
    extract_df_lines = _ordered_lines(modules, "extract_df_lines")
    reindex_lines = _ordered_lines(modules, "reindex_lines")
    post_process_lines = _ordered_lines(modules, "post_process_lines")
    lifetime_output_lines = _ordered_lines(modules, "lifetime_output_lines")

    compute_call_args = _ordered_call_args(
        modules, attr_name="compute_call_args", param_order=BASELINE_SOLVER_PARAM_ORDER
    )
    batch_call_args = _ordered_call_args(
        modules, attr_name="batch_call_args", param_order=BASELINE_SOLVER_PARAM_ORDER
    )

    power_balance_terms = _ordered_lines(modules, "power_balance_terms")
    objective_sum_terms = _ordered_lines(modules, "objective_sum_terms")
    objective_tail_terms = _ordered_lines(modules, "objective_tail_terms")
    penalty_expression_terms = _ordered_lines(modules, "penalty_expression_terms")
    recompute_power_terms = _ordered_lines(modules, "recompute_power_terms")

    source_summary = _render_source_summary(modules)
    objective_sum_expr = " + ".join(objective_sum_terms)
    objective_tail_expr = "".join(
        f"\n        - {line}" for line in objective_tail_terms
    )
    power_balance_expr = " + ".join(["-P_curtailment_t[t]"] + power_balance_terms)
    recompute_expr = " + ".join(recompute_power_terms)
    penalty_expr = penalty_expression_terms[0]

    return f'''# Auto-generated combined EMS module
# Combined from original EMS source of truth: {source_summary}

import numpy as np
import openmdao.api as om
import pandas as pd
from docplex.mp.model import Model

from hydesign.openmdao_wrapper import ComponentWrapper


class ems:
    """Generated baseline EMS for wind + solar + battery using the original CPLEX logic."""

    def __init__(
        self,
        N_time,
        life_y=25,
        intervals_per_hour=1,
        weeks_per_season_per_year=None,
        ems_type="cplex",
    ):
        self.weeks_per_season_per_year = weeks_per_season_per_year
        self.N_time = int(N_time)
        self.ems_type = ems_type
        self.life_y = life_y
        self.life_h = 365 * 24 * life_y
        self.life_intervals = self.life_h * intervals_per_hour
        self.intervals_per_hour = intervals_per_hour

    def compute(
        self,
{_render_signature(BASELINE_INPUT_ORDER, indent=8)}
        **kwargs,
    ):
        if self.ems_type != "cplex":
            raise Warning("This generated EMS only supports cplex")

{_render_lines(compute_prepare_lines, indent=8)}

        WSPr_df = pd.DataFrame(
            index=pd.date_range(
                start="01-01-1991 00:00", periods=len(wind_t), freq="1h"
            )
        )
{_render_lines(dataframe_assignment_lines, indent=8)}

        P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts = ems_cplex(
{_render_items(compute_call_args, indent=12)}
        )

        outputs = {{}}
{_render_lines(lifetime_output_lines, indent=8)}
        return (
            outputs["wind_t_ext"],
            outputs["solar_t_ext"],
            outputs["price_t_ext"],
            outputs["hpp_t"],
            outputs["hpp_curt_t"],
            outputs["b_t"],
            outputs["b_E_SOC_t"],
            outputs["penalty_t"],
        )


class ems_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = ems(**insta_inp)
        super().__init__(
            inputs=[
{_render_ports(input_ports, indent=16)}
            ],
            outputs=[
{_render_ports(output_ports, indent=16)}
            ],
            function=model.compute,
            partial_options=[{{"dependent": False, "val": 0}}],
        )


def expand_to_lifetime(
    x,
    life_y=25,
    intervals_per_hour=1,
    weeks_per_season_per_year=None,
    axis=0,
    additional_intervals=0,
    life=None,
):
    if life is None:
        life = life_y * 365 * 24 * intervals_per_hour
    life = life + additional_intervals
    if weeks_per_season_per_year == None:
        len_x = np.size(x, axis=axis)
        N_repeats = int(np.ceil(life / len_x))
    else:
        x_ext = np.array([])
        for x_batch in split_in_batch(
            x, weeks_per_season_per_year * 7 * 24 * intervals_per_hour
        ):
            x_ext = np.append(
                x_ext, np.tile(x_batch, 20)[: 24 * 7 * 13 * intervals_per_hour]
            )
        x_ext = np.append(x_ext, x_batch[-24 * intervals_per_hour :])
        x = x_ext
        N_repeats = int(np.ceil(life / (365 * 24 * intervals_per_hour)))
    return np.tile(x, N_repeats)[:life]


def ems_cplex(
{_render_function_signature(BASELINE_SOLVER_PARAM_ORDER, defaults={"peak_hr_quantile": "0.9", "cost_of_battery_P_fluct_in_peak_price_ratio": "0.5", "n_full_power_hours_expected_per_day_at_peak_price": "3"}, indent=4)}
    batch_size=110,
):
    batches_all = split_in_batch(list(range(len(wind_ts))), batch_size)
    batches = batches_all
{_render_lines(batch_array_lines, indent=4)}

    for batch in batches:
{_render_lines(batch_slice_lines, indent=8)}
        (
            P_HPP_ts_batch,
            P_curtailment_ts_batch,
            P_charge_discharge_ts_batch,
            E_SOC_ts_batch,
            penalty_batch,
        ) = ems_cplex_parts(
{_render_items(batch_call_args, indent=12)}
        )
{_render_lines(batch_assign_lines, indent=8)}

{_render_lines(finalize_batch_lines, indent=4)}
    return (
        P_HPP_ts,
        P_curtailment_ts,
        P_charge_discharge_ts,
        E_SOC_ts,
        penalty_ts,
    )


def ems_cplex_parts(
{_render_function_signature(BASELINE_SOLVER_PARAM_ORDER, defaults={"peak_hr_quantile": "0.9", "cost_of_battery_P_fluct_in_peak_price_ratio": "0.5", "n_full_power_hours_expected_per_day_at_peak_price": "3"}, indent=4)}
):
    N_t = len(price_ts.index)
    N_days = N_t / 24
    e_peak_day_expected = (
        n_full_power_hours_expected_per_day_at_peak_price * hpp_grid_connection
    )
    e_peak_period_expected = e_peak_day_expected * N_days
    price_peak = np.quantile(price_ts.values, peak_hr_quantile)
    peak_hours_index = np.where(price_ts.values >= price_peak)[0]

    price_ts_to_max = price_peak - price_ts
    price_ts_to_max.loc[price_ts_to_max < 0] = 0
    price_ts_to_max.iloc[:-1] = (
        0.5 * price_ts_to_max.iloc[:-1].values + 0.5 * price_ts_to_max.iloc[1:].values
    )

    mdl = Model(name="EMS")
    mdl.context.cplex_parameters.threads = 1
    mdl.context.cplex_parameters.emphasis.mip = 1

    time = price_ts.index
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta("1hour")]))

{_render_lines(cplex_setup_lines, indent=4)}

    mdl.maximize(
        mdl.sum({objective_sum_expr} for t in time)
        - penalty{objective_tail_expr}
    )

{_render_lines(pre_loop_constraint_lines, indent=4)}
    mdl.add_constraint(penalty == {penalty_expr})

    for t in time:
        tt = t + pd.Timedelta("1hour")
        dt = 1
        mdl.add_constraint(P_HPP_t[t] == {power_balance_expr})
{_render_lines(loop_constraint_lines, indent=8)}

    sol = mdl.solve(log_output=False)

{_render_lines(extract_df_lines, indent=4)}

    penalty_2 = sol.get_value(penalty)
    penalty_ts = np.ones(N_t) * (penalty_2 / N_t)

    mdl.end()

{_render_lines(reindex_lines, indent=4)}

    if len(P_HPP_ts_df) < len(wind_ts):
        P_HPP_ts = {recompute_expr}

{_render_lines(post_process_lines, indent=4)}
    return (
        P_HPP_ts,
        P_curtailment_ts,
        P_charge_discharge_ts,
        E_SOC_ts,
        penalty_ts,
    )


def split_in_batch(array, N):
    batch = []
    counter = 0
    while counter * N < len(array):
        if (counter + 1) * N > len(array):
            end = len(array)
        else:
            end = (counter + 1) * N
        batch += [array[counter * N : end]]
        counter += 1
    return batch
'''


def project_package_init_text() -> str:
    return (
        "from .ems import ems, ems_comp, ems_cplex, ems_cplex_parts\n"
        "\n"
        "__all__ = ['ems', 'ems_comp', 'ems_cplex', 'ems_cplex_parts']\n"
    )


def default_project_name() -> str:
    return "wind_solar_battery_cplex"


def generated_package_path(output_root: Path, project_slug: str) -> Path:
    return output_root / project_slug
