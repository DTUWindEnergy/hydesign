"""Build the contracted PPA delivery CSV datasets only."""

from generate_all import prepare_analysis_data, write_datasets


def main() -> None:
    case_data, hourly_data, definitions = prepare_analysis_data()
    outputs = write_datasets(case_data, hourly_data, definitions)
    print(f"Validated {len(case_data)} optimization cases.")
    print(f"Created {len(hourly_data)} hourly contracted-delivery rows.")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

