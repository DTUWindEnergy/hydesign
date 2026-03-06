from hydesign.modular.modules.ems.artifact_contracts import (
    EMSArtifactModule,
    EMSPortSpec,
)
from hydesign.modular.modules.ems.contracts import EMSExecutionStage, EMSModuleSpec

BATTERY_EMS_MODULE = EMSModuleSpec(
    name="battery",
    kind="technology",
    description=(
        "Battery dispatch, state-of-charge dynamics, and fluctuation cost "
        "contribution for the original CPLEX EMS."
    ),
    source_of_truth=(
        "hydesign/ems/ems.py:66-117",
        "hydesign/ems/ems.py:807-919",
    ),
    depends_on=("base",),
    component_inputs=(
        "b_P",
        "b_E",
        "battery_depth_of_discharge",
        "battery_charge_efficiency",
        "cost_of_battery_P_fluct_in_peak_price_ratio",
    ),
    component_outputs=(
        "b_t",
        "b_E_SOC_t",
    ),
    batch_outputs=(
        "P_charge_discharge_ts",
        "E_SOC_ts",
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
        "Battery is the first stateful technology module. In the original EMS "
        "it owns SOC variables, charge/discharge bounds, efficiency modeling, "
        "and the smoothing penalty in the objective."
    ),
)


BATTERY_EMS_ARTIFACT = EMSArtifactModule(
    spec=BATTERY_EMS_MODULE,
    solver_params=(
        "P_batt_MW",
        "E_batt_MWh_t",
        "battery_depth_of_discharge",
        "charge_efficiency",
        "cost_of_battery_P_fluct_in_peak_price_ratio",
    ),
    component_inputs=(
        EMSPortSpec("b_P", "{'units': 'MW', 'desc': 'Battery power capacity'}"),
        EMSPortSpec("b_E", "{'desc': 'Battery energy storage capacity'}"),
        EMSPortSpec(
            "battery_depth_of_discharge",
            "{'units': 'MW', 'desc': 'battery depth of discharge'}",
        ),
        EMSPortSpec(
            "battery_charge_efficiency",
            "{'desc': 'battery charge efficiency'}",
        ),
        EMSPortSpec(
            "cost_of_battery_P_fluct_in_peak_price_ratio",
            "{'desc': 'cost of battery power fluctuations computed as a peak price ratio.'}",
        ),
    ),
    component_outputs=(
        EMSPortSpec(
            "b_t",
            "{'units': 'MW', 'desc': 'Battery charge/discharge power time series', 'shape': [model.life_h]}",
        ),
        EMSPortSpec(
            "b_E_SOC_t",
            "{'desc': 'Battery energy SOC time series', 'shape': [model.life_h + 1]}",
        ),
    ),
    compute_prepare_lines=(
        "battery_depth_of_discharge = np.asarray(battery_depth_of_discharge).item()",
        "battery_charge_efficiency = np.asarray(battery_charge_efficiency).item()",
        "cost_of_battery_P_fluct_in_peak_price_ratio = np.asarray(cost_of_battery_P_fluct_in_peak_price_ratio).item()",
    ),
    dataframe_assignment_lines=('WSPr_df["E_batt_MWh_t"] = np.asarray(b_E).item()',),
    compute_call_args=(
        "P_batt_MW=np.asarray(b_P).item()",
        "E_batt_MWh_t=WSPr_df.E_batt_MWh_t",
        "battery_depth_of_discharge=float(battery_depth_of_discharge)",
        "charge_efficiency=float(battery_charge_efficiency)",
        "cost_of_battery_P_fluct_in_peak_price_ratio=cost_of_battery_P_fluct_in_peak_price_ratio",
    ),
    batch_array_lines=(
        "P_charge_discharge_ts = np.zeros(len(price_ts))",
        "E_SOC_ts = np.zeros(len(price_ts) + 1)",
    ),
    batch_slice_lines=("E_batt_MWh_t_sel = E_batt_MWh_t.iloc[batch]",),
    batch_call_args=(
        "P_batt_MW=P_batt_MW",
        "E_batt_MWh_t=E_batt_MWh_t_sel",
        "battery_depth_of_discharge=battery_depth_of_discharge",
        "charge_efficiency=charge_efficiency",
        "cost_of_battery_P_fluct_in_peak_price_ratio=cost_of_battery_P_fluct_in_peak_price_ratio",
    ),
    batch_assign_lines=(
        "P_charge_discharge_ts[batch] = P_charge_discharge_ts_batch",
        "E_SOC_ts[batch] = E_SOC_ts_batch[:-1]",
    ),
    finalize_batch_lines=("E_SOC_ts[-1] = E_SOC_ts[0]",),
    cplex_setup_lines=(
        "P_charge_discharge_t = mdl.continuous_var_dict(time, lb=-P_batt_MW / charge_efficiency, ub=P_batt_MW * charge_efficiency, name='Battery power')",
        "E_SOC_t = mdl.continuous_var_dict(SOCtime, lb=0, name='Energy level')",
        "fabs = mdl.piecewise(-1, [(0, 0)], 1)",
    ),
    power_balance_terms=("P_charge_discharge_t[t]",),
    objective_tail_terms=(
        "mdl.sum(fabs(P_charge_discharge_t[t + pd.Timedelta('1hour')] - P_charge_discharge_t[t]) * cost_of_battery_P_fluct_in_peak_price_ratio * price_ts_to_max[t] for t in time[:-1])",
    ),
    pre_loop_constraint_lines=(
        "mdl.add_constraint(E_SOC_t[SOCtime[0]] == 0.5 * E_batt_MWh_t[time[0]])",
        "mdl.add_constraint(E_SOC_t[SOCtime[-1]] == 0.5 * E_batt_MWh_t[time[0]])",
        "f2 = mdl.piecewise(charge_efficiency, [(0, 0)], 1 / charge_efficiency)",
    ),
    loop_constraint_lines=(
        "mdl.add_constraint(E_SOC_t[tt] == E_SOC_t[t] - f2(P_charge_discharge_t[t]) * dt)",
        "mdl.add_constraint(E_SOC_t[t] >= (1 - battery_depth_of_discharge) * E_batt_MWh_t[t])",
        "mdl.add_constraint(E_SOC_t[t] <= E_batt_MWh_t[t])",
        "mdl.add_constraint(P_charge_discharge_t[t] <= P_batt_MW * charge_efficiency)",
        "mdl.add_constraint(P_charge_discharge_t[t] >= -P_batt_MW / charge_efficiency)",
    ),
    extract_df_lines=(
        "P_charge_discharge_ts_df = pd.DataFrame.from_dict(sol.get_value_dict(P_charge_discharge_t), orient='index').loc[:, 0]",
        "E_SOC_ts_df = pd.DataFrame.from_dict(sol.get_value_dict(E_SOC_t), orient='index').loc[:, 0]",
    ),
    reindex_lines=(
        "P_charge_discharge_ts = P_charge_discharge_ts_df.reindex(time, fill_value=0).values",
        "E_SOC_ts = E_SOC_ts_df.reindex(SOCtime, fill_value=0).values",
    ),
    recompute_power_terms=("P_charge_discharge_ts",),
    lifetime_output_lines=(
        "outputs['b_t'] = expand_to_lifetime(P_charge_discharge_ts, life_y=self.life_y, intervals_per_hour=self.intervals_per_hour, weeks_per_season_per_year=self.weeks_per_season_per_year)",
        "outputs['b_E_SOC_t'] = expand_to_lifetime(E_SOC_ts[:-1], life_y=self.life_y, intervals_per_hour=self.intervals_per_hour, weeks_per_season_per_year=self.weeks_per_season_per_year, additional_intervals=1)",
    ),
)
