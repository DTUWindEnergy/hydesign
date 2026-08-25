"""User-facing assumptions for the NPV analysis."""

PROFILE = "Generic Industrial"
PROFILES = (
    "Generic Industrial",
    "Generic Private",
    "Inverse Solar",
    "Continuous Baseload",
)
SETTLEMENT_PERIODS = ("hourly", "daily", "weekly", "monthly")
C_PPA_VALUES = (120.0, 160.0, 200.0)
LAMBDA_PPA_VALUES = (40.0, 60.0, 80.0, 100.0)

P_B_MAX = 20.0
E_MAX = 60.0

# All monetary values are real EUR2016.
EUR2020_TO_EUR2016 = 0.9432
DISCOUNT_RATE = 0.05
PROJECT_LIFETIME_YEARS = 30

WIND_CAPACITY_MW = 37.8
WIND_CAPEX_EUR_PER_MW = 1_037_500.0
WIND_FIXED_OM_EUR_PER_MW_YEAR = 14_887.20
WIND_VARIABLE_OM_EUR_PER_MWH = 1.7647

BATTERY_ENERGY_CAPEX_EUR_PER_MWH = 51_215.76
BATTERY_PROJECT_COST_EUR_PER_MWH = 113_184.00
BATTERY_POWER_CAPEX_EUR_PER_MW = 68_287.68

# DEA central fixed-O&M estimates are linearly interpolated to 2045.
BATTERY_FIXED_OM_2040_KEUR2020_PER_MW_YEAR = 8.7
BATTERY_FIXED_OM_2050_KEUR2020_PER_MW_YEAR = 9.2
BATTERY_VARIABLE_OM_EUR_PER_MWH = 0.0

# No battery replacement and no terminal/salvage value are included.
INCLUDE_BATTERY_REPLACEMENT = False
SALVAGE_VALUE_EUR = 0.0

PALETTE_NAME = "colorblind3"
NPV_12_BAR_COLOR = "#0072B2"
