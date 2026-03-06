from hydesign.modular.modules.ems.artifact_contracts import (
    EMSArtifactModule,
    EMSPortSpec,
)
from hydesign.modular.modules.ems.contracts import EMSExecutionStage, EMSModuleSpec

SOLAR_EMS_MODULE = EMSModuleSpec(
    name="solar",
    kind="technology",
    description="Solar generation contribution to the EMS power balance.",
    source_of_truth=(
        "hydesign/ems/ems.py:66-117",
        "hydesign/ems/ems.py:865-869",
    ),
    depends_on=("base",),
    component_inputs=("solar_t",),
    component_outputs=("solar_t_ext",),
    stages=(
        EMSExecutionStage.OPENMDAO_INPUTS,
        EMSExecutionStage.OPENMDAO_OUTPUTS,
        EMSExecutionStage.COMPUTE_PREPARE,
        EMSExecutionStage.BATCH_SLICE,
        EMSExecutionStage.CPLEX_POWER_BALANCE,
        EMSExecutionStage.LIFETIME_OUTPUTS,
    ),
    notes=(
        "Solar mirrors wind in phase 1: it contributes production to the "
        "dispatch balance without introducing extra CPLEX state variables."
    ),
)


SOLAR_EMS_ARTIFACT = EMSArtifactModule(
    spec=SOLAR_EMS_MODULE,
    solver_params=("solar_ts",),
    component_inputs=(
        EMSPortSpec(
            "solar_t",
            "{'units': 'MW', 'desc': 'PVP power time series', 'shape': [model.N_time]}",
        ),
    ),
    component_outputs=(
        EMSPortSpec(
            "solar_t_ext",
            "{'units': 'MW', 'desc': 'PVP power time series', 'shape': [model.life_h]}",
        ),
    ),
    dataframe_assignment_lines=('WSPr_df["solar_t"] = solar_t',),
    compute_call_args=("solar_ts=WSPr_df.solar_t",),
    batch_slice_lines=("solar_ts_sel = solar_ts.iloc[batch]",),
    batch_call_args=("solar_ts=solar_ts_sel",),
    power_balance_terms=("solar_ts[t]",),
    recompute_power_terms=("solar_ts.values",),
    lifetime_output_lines=(
        "outputs['solar_t_ext'] = expand_to_lifetime(solar_t, life_y=self.life_y, intervals_per_hour=self.intervals_per_hour, weeks_per_season_per_year=self.weeks_per_season_per_year)",
    ),
)
