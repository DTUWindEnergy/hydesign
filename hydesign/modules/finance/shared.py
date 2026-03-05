from hydesign.modules.finance.contracts import FinanceModuleSpec

SHARED_FINANCE_MODULE = FinanceModuleSpec(
    name="shared",
    kind="shared_asset",
    description="Shared electrical CAPEX/OPEX contribution to the WSB finance layer.",
    source_of_truth=("hydesign/finance/finance.py:67-155",),
    component_inputs=("CAPEX_el", "OPEX_el"),
    notes="Always included in the WSB baseline because calculate_WACC includes shared electrical CAPEX.",
)
