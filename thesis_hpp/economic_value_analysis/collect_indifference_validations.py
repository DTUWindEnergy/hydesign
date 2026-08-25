from __future__ import annotations

import csv
import json

from common import RESULTS_DIR


def main() -> None:
    root = RESULTS_DIR / "indifference_validations"
    summaries = []
    for path in sorted(root.glob("reference_lambda_*/summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    if not summaries:
        raise FileNotFoundError(f"No validation summaries found below {root}.")

    summaries.sort(key=lambda row: float(row["reference_lambda_ppa"]))
    fieldnames = []
    for summary in summaries:
        for key in summary:
            if key not in fieldnames:
                fieldnames.append(key)

    output_path = RESULTS_DIR / "indifference_validations_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Collected {len(summaries)} validation summaries in {output_path}")


if __name__ == "__main__":
    main()
