"""Settings for the targeted battery and settlement-period interaction."""

DELIVERY_PROFILE = "Generic Industrial Yearly"
SDP_LABEL = "Generic Industrial"

# One benchmark PPA configuration is used to isolate the interaction.
LAMBDA_PPA = 80.0
C_PPA = 160.0

# Hourly results already exist. These are the three additional periods.
SETTLEMENT_PERIODS = ("daily", "weekly", "monthly")

# The existing 20 MW/60 MWh settlement-period results are reused later.
# This runner solves only the two missing battery configurations.
BATTERY_CASES_TO_SOLVE = (
    (0.0, 0.0, 0.0),
    (10.0, 30.0, 15.0),
)

WEATHER_YEARS = (
    1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
    2000, 2002, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2013,
)
XI_WIND_FILES = (
    "P_forecast1.csv",
    "P_forecast2.csv",
    "P_forecast3.csv",
    "P_forecast4.csv",
)
STAGE2_WIND_METHOD = "mean"

SOLVER_NAME = "cplex"
SOLVER_THREADS = 16
SKIP_COMPLETED_CASES = True
