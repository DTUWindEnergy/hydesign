"""User-facing settings for the expected-profit decomposition figures."""

# Change this to ("weekly",), ("daily",), or ("monthly",) to process another
# available PPA deviation settlement period.
SETTLEMENT_PERIODS = ("hourly",)

# Values are (battery power [MW], battery energy [MWh]).
BATTERY_CASES = ((20.0, 60.0),)

C_PPA_VALUES = (120.0, 160.0, 200.0)
LAMBDA_PPA_VALUES = (40.0, 60.0, 80.0, 100.0)

# Detailed stage-three files are read in chunks to limit memory use.
CSV_CHUNK_SIZE = 200_000

# Independent result cases can be audited in parallel. Four workers keeps
# memory use moderate while avoiding slow sequential reads from the shared
# drive.
CASE_WORKERS = 4

# Reuse the audited decomposition CSV when it already contains exactly the
# requested cases. Set this to True after changing the calculation itself.
REBUILD_DECOMPOSITION_DATA = False

# Full-width grouped figures are the main presentation. The compressed 2x2
# version and the existing stacked figures are retained but disabled.
GENERATE_INDIVIDUAL_GROUPED_FIGURES = True
GENERATE_GROUPED_2X2_FIGURES = False
GENERATE_STACKED_FIGURES = False

# Four colors are used for lambda_PPA = 40, 60, 80, and 100 EUR/MWh.
GROUPED_PLOT_PALETTE = "colorblind4"

# Labels smaller than this are omitted from the grouped figures only. Their
# exact values remain available in expected_profit_decomposition.csv.
GROUPED_MIN_LABEL_EUR = 1_000.0

# Segment labels are reported in MEUR/year. Values below this threshold cannot
# be placed legibly inside a stacked segment and remain available in the CSV.
MIN_SEGMENT_LABEL_MEUR = 0.15

# Numerical audit tolerance when reconstructing the reported mean-CVaR
# objective from the exported decision variables.
OBJECTIVE_AUDIT_ABS_TOL_EUR = 5.0
OBJECTIVE_AUDIT_REL_TOL = 1e-7


COMPONENTS = (
    ("ppa_revenue", "PPA revenue", "#0072B2"),
    ("spot_revenue", "Day-ahead spot revenue", "#009E73"),
    ("imbalance_settlement", "Imbalance settlement", "#777777"),
    ("imbalance_penalty", "Imbalance penalty", "#999999"),
    ("mfrr_up", "mFRR-up revenue", "#E69F00"),
    ("mfrr_down", "mFRR-down revenue", "#CC79A7"),
    ("ppa_shortfall_cost", "PPA shortfall cost", "#D55E00"),
    ("battery_degradation", "Battery degradation cost", "#9A6A2F"),
)
