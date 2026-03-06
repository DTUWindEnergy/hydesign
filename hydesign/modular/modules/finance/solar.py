from hydesign.modular.modules.finance.contracts import FinanceModuleSpec

SOLAR_FINANCE_MODULE = FinanceModuleSpec(
    name="solar",
    kind="technology",
    description="Solar CAPEX/OPEX and WACC contribution to the WSB finance layer.",
    source_of_truth=("hydesign/finance/finance.py:67-155",),
    component_inputs=("CAPEX_s", "OPEX_s", "solar_WACC"),
    notes="Technology contribution metadata only in phase 1; code is still emitted from the original finance.py.",
)
