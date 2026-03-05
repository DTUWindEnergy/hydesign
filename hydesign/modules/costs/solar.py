from hydesign.modules.costs.contracts import CostsModuleSpec

SOLAR_COSTS_MODULE = CostsModuleSpec(
    name="solar",
    kind="technology",
    description="Solar PV cost component for plant CAPEX and OPEX.",
    source_of_truth=("hydesign/costs/costs.py:143-242",),
    symbols=("pvp_cost", "pvp_cost_comp"),
    component_inputs=("solar_MW", "DC_AC_ratio"),
    component_outputs=("CAPEX_s", "OPEX_s"),
    notes="Source of truth is the original PV cost model in hydesign/costs/costs.py.",
)
