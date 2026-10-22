"""Settings for the settlement-period indifference-price comparison."""

C_PPA_VALUES = (120.0, 160.0, 200.0)
SETTLEMENT_PERIODS = ("hourly", "daily", "weekly", "monthly")
REFERENCE_PPA_PRICES = (40.0, 60.0, 80.0, 100.0)

PROFILE = "Generic Industrial"
P_B_MAX = 20.0
E_MAX = 60.0
PALETTE_NAME = "colorblind4"

# One optional direct optimization check of the shortcut equation.
VALIDATION_C_PPA = 200.0
VALIDATION_SETTLEMENT_PERIOD = "monthly"
VALIDATION_REFERENCE_PPA_PRICE = 80.0
