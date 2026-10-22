"""Settings for the benchmark risk-aversion sensitivity."""

DELIVERY_PROFILE = "Generic Industrial Yearly"
PROFILE_LABEL = "Generic Industrial"
LAMBDA_PPA = 80.0
C_PPA = 160.0
P_B_MAX = 20.0
E_MAX = 60.0
E0 = 30.0
SETTLEMENT_PERIOD = "hourly"
ALPHA = 0.9

BASELINE_BETA = 0.2
BETAS_TO_SOLVE = (0.0, 0.1, 0.5, 0.8, 1.0)

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
PALETTE_NAME = "colorblind4"
