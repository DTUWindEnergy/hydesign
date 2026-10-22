"""Build a compact catalog from case-level optimization summaries."""

from pathlib import Path

import pandas as pd


PROJECT_FOLDERS = {
    "Generic Industrial": "thesis_hpp",
    "Generic Private": "thesis_hpp_Private",
    "Inverse Solar": "thesis_hpp_Inverse_Solar",
    "Continuous Baseload": "thesis_hpp_Baseload",
}


def post_processing_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_roots() -> dict[str, Path]:
    thesis_root = post_processing_root().parent
    projects_parent = thesis_root.parent
    return {
        label: projects_parent / folder
        for label, folder in PROJECT_FOLDERS.items()
    }


def build_results_catalog(save: bool = True) -> pd.DataFrame:
    """Read every case-level summary.csv from the four SDP projects."""
    frames = []

    for sdp_label, project_root in project_roots().items():
        results_root = project_root / "results"
        if not results_root.exists():
            raise FileNotFoundError(f"Missing results directory: {results_root}")

        summary_paths = sorted(results_root.glob("settlement_*/**/summary.csv"))
        if not summary_paths:
            raise FileNotFoundError(f"No case-level summary.csv files under {results_root}")

        for summary_path in summary_paths:
            frame = pd.read_csv(summary_path)
            if len(frame) != 1:
                raise ValueError(
                    f"Expected one row in {summary_path}, found {len(frame)}."
                )
            frame["sdp_label"] = sdp_label
            frame["project_root"] = str(project_root)
            frame["case_dir"] = str(summary_path.parent)
            frames.append(frame)

    catalog = pd.concat(frames, ignore_index=True)

    numeric_columns = [
        "lambda_PPA",
        "c_ppa",
        "P_b_max",
        "E_max",
        "objective_value",
        "S_PPA",
        "elapsed_seconds",
    ]
    for column in numeric_columns:
        if column in catalog.columns:
            catalog[column] = pd.to_numeric(catalog[column], errors="raise")

    key_columns = [
        "sdp_label",
        "settlement_period",
        "lambda_PPA",
        "c_ppa",
        "P_b_max",
        "E_max",
    ]
    duplicates = catalog.duplicated(key_columns, keep=False)
    if duplicates.any():
        duplicate_rows = catalog.loc[duplicates, key_columns + ["case_dir"]]
        raise ValueError(
            "Duplicate result cases found:\n"
            + duplicate_rows.to_string(index=False)
        )

    catalog = catalog.sort_values(key_columns).reset_index(drop=True)

    if save:
        output_path = post_processing_root() / "common" / "results_catalog.csv"
        catalog.to_csv(output_path, index=False)
        print(f"Saved results catalog with {len(catalog)} cases: {output_path}")

    return catalog


def hourly_supply_data(
    p_b_max: float = 20.0,
    e_max: float = 60.0,
) -> pd.DataFrame:
    catalog = build_results_catalog(save=True)
    selected = catalog.loc[
        (catalog["settlement_period"].str.lower() == "hourly")
        & (catalog["P_b_max"] == p_b_max)
        & (catalog["E_max"] == e_max)
    ].copy()

    expected_rows = len(PROJECT_FOLDERS) * 3 * 4
    if len(selected) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} hourly supply-curve cases, "
            f"found {len(selected)}."
        )

    return selected
