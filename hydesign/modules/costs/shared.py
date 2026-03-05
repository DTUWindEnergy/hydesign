from hydesign.modules.costs.contracts import CostsModuleSpec

SHARED_COSTS_MODULE = CostsModuleSpec(
    name="shared",
    kind="shared_asset",
    description="Shared electrical infrastructure and land-rent cost component.",
    source_of_truth=("hydesign/costs/costs.py:361-438",),
    symbols=("shared_cost", "shared_cost_comp"),
    component_inputs=("G_MW", "Awpp", "Apvp"),
    component_outputs=("CAPEX_sh", "OPEX_sh"),
    notes="Always included for the WSB baseline because finance depends on shared CAPEX and OPEX.",
)
