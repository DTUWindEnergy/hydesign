from __future__ import annotations

from dataclasses import dataclass

from hydesign.modules.ems.contracts import EMSModuleSpec


@dataclass(frozen=True)
class EMSPortSpec:
    name: str
    kwargs_code: str


@dataclass(frozen=True)
class EMSArtifactModule:
    spec: EMSModuleSpec
    solver_params: tuple[str, ...] = ()
    component_inputs: tuple[EMSPortSpec, ...] = ()
    component_outputs: tuple[EMSPortSpec, ...] = ()
    compute_prepare_lines: tuple[str, ...] = ()
    dataframe_assignment_lines: tuple[str, ...] = ()
    compute_call_args: tuple[str, ...] = ()
    batch_array_lines: tuple[str, ...] = ()
    batch_slice_lines: tuple[str, ...] = ()
    batch_call_args: tuple[str, ...] = ()
    batch_assign_lines: tuple[str, ...] = ()
    finalize_batch_lines: tuple[str, ...] = ()
    cplex_setup_lines: tuple[str, ...] = ()
    power_balance_terms: tuple[str, ...] = ()
    objective_sum_terms: tuple[str, ...] = ()
    objective_tail_terms: tuple[str, ...] = ()
    penalty_expression_terms: tuple[str, ...] = ()
    pre_loop_constraint_lines: tuple[str, ...] = ()
    loop_constraint_lines: tuple[str, ...] = ()
    extract_df_lines: tuple[str, ...] = ()
    reindex_lines: tuple[str, ...] = ()
    recompute_power_terms: tuple[str, ...] = ()
    post_process_lines: tuple[str, ...] = ()
    lifetime_output_lines: tuple[str, ...] = ()
