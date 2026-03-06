from hydesign.modular.modules.costs.contracts import CostsModuleSpec

WIND_COSTS_MODULE = CostsModuleSpec(
    name="wind",
    kind="technology",
    description="Wind cost component for WPP CAPEX and OPEX.",
    source_of_truth=("hydesign/costs/costs.py:9-141",),
    symbols=("wpp_cost", "wpp_cost_comp"),
    component_inputs=("Nwt", "hh", "d", "p_rated", "wind_t"),
    component_outputs=("CAPEX_w", "OPEX_w"),
    notes="Source of truth is the original wind cost model in hydesign/costs/costs.py.",
)
