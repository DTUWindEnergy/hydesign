"""Run the complete NPV post-processing workflow."""

from export_appendix_tables import export_appendix_tables
from npv_analysis import calculate_npv_results
from plot_npv_12_bars_by_settlement_period import plot_all_periods_12_bars
from plot_npv_by_settlement_period import plot_all_periods
from plot_npv_heatmap_by_settlement_period import plot_npv_heatmap
from plot_positive_npv_counts import plot_counts


def main() -> None:
    data = calculate_npv_results()
    outputs = (
        plot_all_periods(data)
        + plot_all_periods_12_bars(data)
        + plot_npv_heatmap(data)
        + plot_counts(data)
        + export_appendix_tables(data)
    )
    print("\nAll NPV outputs are ready:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
