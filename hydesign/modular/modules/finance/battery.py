from hydesign.modular.modules.finance.contracts import FinanceModuleSpec

BATTERY_FINANCE_MODULE = FinanceModuleSpec(
    name="battery",
    kind="technology",
    description="Battery CAPEX/OPEX and WACC contribution to the WSB finance layer.",
    source_of_truth=("hydesign/finance/finance.py:67-155",),
    component_inputs=("CAPEX_b", "OPEX_b", "battery_WACC"),
    notes="Technology contribution metadata only in phase 1; code is still emitted from the original finance.py.",
)
