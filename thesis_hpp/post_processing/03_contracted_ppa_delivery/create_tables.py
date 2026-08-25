"""Generate the contracted PPA delivery LaTeX tables only."""

from generate_all import prepare_analysis_data, write_tables


def main() -> None:
    case_data, _, definitions = prepare_analysis_data()
    outputs = write_tables(case_data, definitions)
    print(f"Created {len(outputs)} LaTeX table files.")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

