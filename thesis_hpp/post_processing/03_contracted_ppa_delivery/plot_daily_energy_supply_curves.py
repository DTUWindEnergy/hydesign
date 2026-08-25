"""Generate the daily contracted-energy supply curves only."""

from generate_all import plot_daily_energy_supply_curves, prepare_analysis_data


def main() -> None:
    case_data, _, _ = prepare_analysis_data()
    outputs = plot_daily_energy_supply_curves(case_data)
    print(
        f"Created {len(outputs) // 2} daily-energy supply curves "
        "in PDF and PNG."
    )


if __name__ == "__main__":
    main()
