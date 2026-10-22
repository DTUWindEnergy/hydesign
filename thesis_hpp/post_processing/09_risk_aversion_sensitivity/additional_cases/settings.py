"""Settings for the additional risk-aversion comparison cases."""

BETAS = (0.0, 0.1, 0.2, 0.5, 0.8, 1.0)
BASELINE_BETA = 0.2
ALPHA = 0.9

DELIVERY_PROFILE = "Generic Industrial Yearly"
SETTLEMENT_PERIOD = "hourly"
P_B_MAX = 20.0
E_MAX = 60.0
E0 = 30.0

# When optimize_s_ppa is False, the capacity is fixed at fixed_s_ppa.
CASES = (
    {
        "case_id": "high_contract",
        "label": "High-contract PPA",
        "lambda_ppa": 100.0,
        "c_ppa": 120.0,
        "optimize_s_ppa": True,
        "fixed_s_ppa": None,
        "betas": BETAS,
    },
    {
        "case_id": "merchant",
        "label": "Merchant operation",
        "lambda_ppa": 0.0,
        "c_ppa": 160.0,
        "optimize_s_ppa": False,
        "fixed_s_ppa": 0.0,
        "betas": BETAS,
    },
    {
        "case_id": "high_penalty",
        "label": "High-penalty operation",
        "lambda_ppa": 80.0,
        "c_ppa": 200.0,
        "optimize_s_ppa": True,
        "fixed_s_ppa": None,
        "betas": BETAS,
    },
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
