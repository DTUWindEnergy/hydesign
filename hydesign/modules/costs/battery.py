from hydesign.modules.costs.contracts import CostsModuleSpec

BATTERY_COSTS_MODULE = CostsModuleSpec(
    name="battery",
    kind="technology",
    description="Battery cost component for storage CAPEX and OPEX.",
    source_of_truth=("hydesign/costs/costs.py:244-359",),
    symbols=("battery_cost", "battery_cost_comp"),
    component_inputs=("b_P", "b_E", "SoH"),
    component_outputs=("CAPEX_b", "OPEX_b"),
    notes="Source of truth is the original battery cost model in hydesign/costs/costs.py.",
)
