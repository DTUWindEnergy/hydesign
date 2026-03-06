from hydesign.modular.modules.ems.artifact_contracts import (
    EMSArtifactModule,
    EMSPortSpec,
)
from hydesign.modular.modules.ems.contracts import EMSExecutionStage, EMSModuleSpec

WIND_EMS_MODULE = EMSModuleSpec(
    name="wind",
    kind="technology",
    description="Wind generation contribution to the EMS power balance.",
    source_of_truth=(
        "hydesign/ems/ems.py:66-117",
        "hydesign/ems/ems.py:865-869",
    ),
    depends_on=("base",),
    component_inputs=("wind_t",),
    component_outputs=("wind_t_ext",),
    stages=(
        EMSExecutionStage.OPENMDAO_INPUTS,
        EMSExecutionStage.OPENMDAO_OUTPUTS,
        EMSExecutionStage.COMPUTE_PREPARE,
        EMSExecutionStage.BATCH_SLICE,
        EMSExecutionStage.CPLEX_POWER_BALANCE,
        EMSExecutionStage.LIFETIME_OUTPUTS,
    ),
    notes=(
        "Wind is a pure supply-side module in phase 1. "
        "It adds no standalone solver variables in the original CPLEX EMS."
    ),
)


WIND_EMS_ARTIFACT = EMSArtifactModule(
    spec=WIND_EMS_MODULE,
    solver_params=("wind_ts",),
    component_inputs=(
        EMSPortSpec(
            "wind_t",
            "{'units': 'MW', 'desc': 'WPP power time series', 'shape': [model.N_time]}",
        ),
    ),
    component_outputs=(
        EMSPortSpec(
            "wind_t_ext",
            "{'units': 'MW', 'desc': 'WPP power time series', 'shape': [model.life_h]}",
        ),
    ),
    dataframe_assignment_lines=('WSPr_df["wind_t"] = wind_t',),
    compute_call_args=("wind_ts=WSPr_df.wind_t",),
    batch_slice_lines=("wind_ts_sel = wind_ts.iloc[batch]",),
    batch_call_args=("wind_ts=wind_ts_sel",),
    power_balance_terms=("wind_ts[t]",),
    recompute_power_terms=("wind_ts.values",),
    lifetime_output_lines=(
        "outputs['wind_t_ext'] = expand_to_lifetime(wind_t, life_y=self.life_y, intervals_per_hour=self.intervals_per_hour, weeks_per_season_per_year=self.weeks_per_season_per_year)",
    ),
)
