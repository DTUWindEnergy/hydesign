from hydesign.modules.finance.contracts import FinanceModuleSpec

BASE_FINANCE_MODULE = FinanceModuleSpec(
    name="base",
    kind="core",
    description="Shared WSB finance aggregation and helper functions.",
    source_of_truth=("hydesign/finance/finance.py:19-481",),
    symbols=(
        "finance",
        "finance_comp",
        "calculate_NPV_IRR",
        "calculate_WACC",
        "calculate_revenues",
        "calculate_break_even_PPA_price",
        "calculate_CAPEX_phasing",
        "get_inflation_index",
    ),
    component_inputs=("hpp_t_with_deg", "penalty_t", "price_t_ext", "tax_rate"),
    component_outputs=(
        "CAPEX",
        "OPEX",
        "revenues",
        "NPV",
        "IRR",
        "NPV_over_CAPEX",
        "LCOE",
        "mean_AEP",
        "penalty_lifetime",
        "break_even_PPA_price",
    ),
    notes="Source of truth is the original finance aggregator in hydesign/finance/finance.py.",
)
