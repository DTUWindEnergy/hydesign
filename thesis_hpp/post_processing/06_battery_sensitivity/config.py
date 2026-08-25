"""User-facing settings for the battery-sensitivity analysis."""

# Main comparison shown in the results chapter.
SETTLEMENT_PERIOD = "hourly"
PROFILE = "Generic Industrial"

# Values are (battery power [MW], battery energy [MWh]).
BATTERY_CASES = (
    (0.0, 0.0),
    (10.0, 30.0),
    (20.0, 60.0),
)

C_PPA_VALUES = (120.0, 160.0, 200.0)
LAMBDA_PPA_VALUES = (40.0, 60.0, 80.0, 100.0)

# Select "daily_energy" for MWh/day or "capacity" for S_PPA in MW.
CONTRACT_FIGURE_METRIC = "daily_energy"

# Cases used for the detailed revenue and cost decomposition.
DECOMPOSITION_C_PPA_VALUES = C_PPA_VALUES
DECOMPOSITION_LAMBDA_PPA = 80.0

PALETTE_NAME = "colorblind3"

# Detailed stage-three result files are read in chunks.
CSV_CHUNK_SIZE = 200_000

# Independent expected-profit cases can be processed in parallel.
EXPECTED_PROFIT_CASE_WORKERS = 4

# Set to True after changing the decomposition calculation itself.
REBUILD_DECOMPOSITION_DATA = False
REBUILD_EXPECTED_PROFIT_DATA = False

# Numerical tolerance for reproducing the reported mean-CVaR objective.
OBJECTIVE_AUDIT_ABS_TOL_EUR = 5.0
OBJECTIVE_AUDIT_REL_TOL = 1e-7
