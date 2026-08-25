"""User-facing settings for the settlement-period analysis."""

PROFILE = "Generic Industrial"
PROFILES = (
    "Generic Industrial",
    "Generic Private",
    "Inverse Solar",
    "Continuous Baseload",
)
SETTLEMENT_PERIODS = ("hourly", "daily", "weekly", "monthly")

P_B_MAX = 20.0
E_MAX = 60.0

C_PPA_VALUES = (120.0, 160.0, 200.0)
LAMBDA_PPA_VALUES = (40.0, 60.0, 80.0, 100.0)

BENCHMARK_C_PPA = 160.0
BENCHMARK_LAMBDA_PPA = 80.0

PALETTE_NAME = "colorblind4"
CASE_WORKERS = 4

# Set to True after changing the expected-profit or shortfall calculation.
REBUILD_ANALYSIS_DATA = False
