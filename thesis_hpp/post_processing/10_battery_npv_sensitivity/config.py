"""User-facing settings for the SoC-cycling sensitivity analysis."""

# Existing daily-cycling results are reused; only missing three-day cases run.
DELIVERY_PROFILE = "Generic Industrial Yearly"
SDP_LABEL = "Generic Industrial"
SETTLEMENT_PERIOD = "hourly"
LAMBDA_PPA = 80.0
C_PPA = 160.0

SOC_CYCLE_HOURS = 72

# Three-day cases required for a battery-size comparison. Existing completed
# summaries are reused, so rerunning the script solves only missing cases.
THREE_DAY_BATTERY_CASES = (
    (10.0, 30.0, 15.0),
    (20.0, 60.0, 30.0),
)
SKIP_EXISTING_THREE_DAY_RESULTS = True

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

# Existing battery configurations included in the daily-cycling comparison.
DAILY_CYCLING_BATTERY_CASES = (
    (0.0, 0.0),
    (10.0, 30.0),
    (20.0, 60.0),
)

PALETTE_NAME = "colorblind3"
