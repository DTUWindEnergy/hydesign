from hydesign.modular.modules.ems.contracts import EMSExecutionStage, EMSModuleSpec

P2X_EMS_MODULE = EMSModuleSpec(
    name="p2x",
    kind="technology",
    description=(
        "Electrolyzer and hydrogen storage contribution extending the original "
        "wind-solar-battery CPLEX EMS."
    ),
    source_of_truth=(
        "hydesign/ems/ems_P2X.py:10-415",
        "hydesign/ems/ems_P2X.py:418-858",
    ),
    depends_on=("base",),
    component_inputs=(
        "ptg_MW",
        "m_H2_demand_t",
        "HSS_kg",
    ),
    component_outputs=(
        "P_ptg_t",
        "P_ptg_SB_t",
        "m_H2_t",
        "m_H2_offtake_t",
        "LoS_H2_t",
        "total_curtailment",
        "m_H2_demand_t_ext",
    ),
    config_inputs=(
        "eff_curve",
        "price_H2",
        "storage_eff",
        "hhv",
        "penalty_factor_H2",
        "min_power_standby",
        "electrolyzer_eff_curve_type",
        "load_min_penalty_factor",
    ),
    batch_outputs=(
        "P_ptg_ts",
        "P_ptg_SB_ts",
        "m_H2_ts",
        "m_H2_offtake_ts",
        "LoS_H2_ts",
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
        "P2X extends the base EMS with hydrogen revenue, hydrogen demand penalty, "
        "electrolyzer piecewise efficiency, and hydrogen storage state updates."
    ),
)
