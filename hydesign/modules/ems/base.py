from hydesign.modules.ems.artifact_contracts import EMSArtifactModule, EMSPortSpec
from hydesign.modules.ems.contracts import EMSExecutionStage, EMSModuleSpec

BASE_EMS_MODULE = EMSModuleSpec(
    name="base",
    kind="core",
    description=(
        "Shared CPLEX EMS orchestration from the original wind-solar-battery "
        "implementation: synthetic time index creation, batching, peak-price "
        "penalty, HPP output, curtailment, and lifetime expansion."
    ),
    source_of_truth=(
        "hydesign/ems/ems.py:13-193",
        "hydesign/ems/ems.py:604-927",
    ),
    component_inputs=(
        "price_t",
        "G_MW",
        "peak_hr_quantile",
        "n_full_power_hours_expected_per_day_at_peak_price",
    ),
    component_outputs=(
        "price_t_ext",
        "hpp_t",
        "hpp_curt_t",
        "penalty_t",
    ),
    batch_outputs=(
        "P_HPP_ts",
        "P_curtailment_ts",
        "penalty_ts",
    ),
    stages=(
        EMSExecutionStage.OPENMDAO_INPUTS,
        EMSExecutionStage.OPENMDAO_OUTPUTS,
        EMSExecutionStage.COMPUTE_PREPARE,
        EMSExecutionStage.BATCH_ALLOCATE,
        EMSExecutionStage.BATCH_SLICE,
        EMSExecutionStage.CPLEX_VARIABLES,
        EMSExecutionStage.CPLEX_POWER_BALANCE,
        EMSExecutionStage.CPLEX_OBJECTIVE,
        EMSExecutionStage.CPLEX_CONSTRAINTS,
        EMSExecutionStage.CPLEX_EXTRACT,
        EMSExecutionStage.FINALIZE_BATCH_OUTPUTS,
        EMSExecutionStage.LIFETIME_OUTPUTS,
    ),
    notes=(
        "This is always included. It is not a user-selected technology. "
        "Phase 1 must stay aligned with the original CPLEX path in hydesign/ems/ems.py."
    ),
)


BASE_EMS_ARTIFACT = EMSArtifactModule(
    spec=BASE_EMS_MODULE,
    solver_params=(
        "price_ts",
        "hpp_grid_connection",
        "peak_hr_quantile",
        "n_full_power_hours_expected_per_day_at_peak_price",
    ),
    component_inputs=(
        EMSPortSpec(
            "price_t",
            "{'desc': 'Electricity price time series', 'shape': [model.N_time]}",
        ),
        EMSPortSpec("G_MW", "{'units': 'MW', 'desc': 'Grid capacity'}"),
        EMSPortSpec(
            "peak_hr_quantile",
            "{'desc': 'Quantile of price tim sereis to define peak price hours (above this quantile).\\nOnly used for peak production penalty and for cost of battery degradation.'}",
        ),
        EMSPortSpec(
            "n_full_power_hours_expected_per_day_at_peak_price",
            "{'desc': 'Pnealty occurs if nunmber of full power hours expected per day at peak price are not reached.'}",
        ),
    ),
    component_outputs=(
        EMSPortSpec(
            "price_t_ext",
            "{'desc': 'Electricity price time series', 'shape': [model.life_h]}",
        ),
        EMSPortSpec(
            "hpp_t",
            "{'units': 'MW', 'desc': 'HPP power time series', 'shape': [model.life_h]}",
        ),
        EMSPortSpec(
            "hpp_curt_t",
            "{'units': 'MW', 'desc': 'HPP curtailed power time series', 'shape': [model.life_h]}",
        ),
        EMSPortSpec(
            "penalty_t",
            "{'desc': 'penalty for not reaching expected energy productin at peak hours', 'shape': [model.life_h]}",
        ),
    ),
    compute_prepare_lines=(
        "peak_hr_quantile = np.asarray(peak_hr_quantile).item()",
        "n_full_power_hours_expected_per_day_at_peak_price = np.asarray(n_full_power_hours_expected_per_day_at_peak_price).item()",
    ),
    dataframe_assignment_lines=('WSPr_df["price_t"] = price_t',),
    compute_call_args=(
        "price_ts=WSPr_df.price_t",
        "hpp_grid_connection=np.asarray(G_MW).item()",
        "peak_hr_quantile=peak_hr_quantile",
        "n_full_power_hours_expected_per_day_at_peak_price=n_full_power_hours_expected_per_day_at_peak_price",
    ),
    batch_array_lines=(
        "P_HPP_ts = np.zeros(len(price_ts))",
        "P_curtailment_ts = np.zeros(len(price_ts))",
        "penalty_ts = np.zeros(len(price_ts))",
    ),
    batch_slice_lines=("price_ts_sel = price_ts.iloc[batch]",),
    batch_call_args=(
        "price_ts=price_ts_sel",
        "hpp_grid_connection=hpp_grid_connection",
        "peak_hr_quantile=peak_hr_quantile",
        "n_full_power_hours_expected_per_day_at_peak_price=n_full_power_hours_expected_per_day_at_peak_price",
    ),
    batch_assign_lines=(
        "P_HPP_ts[batch] = P_HPP_ts_batch",
        "P_curtailment_ts[batch] = P_curtailment_ts_batch",
        "penalty_ts[batch] = penalty_batch",
    ),
    cplex_setup_lines=(
        "P_HPP_t = mdl.continuous_var_dict(time, lb=0, ub=hpp_grid_connection, name='HPP power output')",
        "P_curtailment_t = mdl.continuous_var_dict(time, lb=0, name='Curtailment')",
        "penalty = mdl.continuous_var(name='penalty', lb=-1e12)",
        "e_penalty = mdl.continuous_var(name='e_penalty', lb=-1e12)",
    ),
    objective_sum_terms=("price_ts[t] * P_HPP_t[t]",),
    penalty_expression_terms=("price_peak * f1(e_penalty)",),
    pre_loop_constraint_lines=(
        "mdl.add_constraint(e_penalty == (e_peak_period_expected - mdl.sum(P_HPP_t[time[i]] for i in peak_hours_index)))",
        "f1 = mdl.piecewise(0, [(0, 0)], 1)",
    ),
    extract_df_lines=(
        "P_HPP_ts_df = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_t), orient='index').loc[:, 0]",
        "P_curtailment_ts_df = pd.DataFrame.from_dict(sol.get_value_dict(P_curtailment_t), orient='index').loc[:, 0]",
    ),
    reindex_lines=(
        "P_HPP_ts = P_HPP_ts_df.reindex(time, fill_value=0).values",
        "P_curtailment_ts = P_curtailment_ts_df.reindex(time, fill_value=0).values",
    ),
    lifetime_output_lines=(
        "outputs['price_t_ext'] = expand_to_lifetime(price_t, life_y=self.life_y, intervals_per_hour=self.intervals_per_hour, weeks_per_season_per_year=self.weeks_per_season_per_year)",
        "outputs['hpp_t'] = expand_to_lifetime(P_HPP_ts, life_y=self.life_y, intervals_per_hour=self.intervals_per_hour, weeks_per_season_per_year=self.weeks_per_season_per_year)",
        "outputs['hpp_curt_t'] = expand_to_lifetime(P_curtailment_ts, life_y=self.life_y, intervals_per_hour=self.intervals_per_hour, weeks_per_season_per_year=self.weeks_per_season_per_year)",
        "outputs['penalty_t'] = expand_to_lifetime(penalty_ts, life_y=self.life_y, intervals_per_hour=self.intervals_per_hour, weeks_per_season_per_year=self.weeks_per_season_per_year)",
    ),
)
