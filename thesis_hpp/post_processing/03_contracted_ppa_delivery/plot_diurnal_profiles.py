"""Generate the contracted PPA diurnal-profile figures only."""

from generate_all import plot_diurnal_cases, prepare_analysis_data


def main() -> None:
    _, hourly_data, _ = prepare_analysis_data()
    outputs = plot_diurnal_cases(hourly_data)
    print(f"Created {len(outputs) // 2} diurnal figures in PDF and PNG.")


if __name__ == "__main__":
    main()

