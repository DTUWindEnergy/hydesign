from hydesign.modular.modules.finance.contracts import FinanceModuleSpec

WIND_FINANCE_MODULE = FinanceModuleSpec(
    name="wind",
    kind="technology",
    description="Wind CAPEX/OPEX and WACC contribution to the WSB finance layer.",
    source_of_truth=("hydesign/finance/finance.py:67-155",),
    component_inputs=("CAPEX_w", "OPEX_w", "wind_WACC"),
    notes="Technology contribution metadata only in phase 1; code is still emitted from the original finance.py.",
)
