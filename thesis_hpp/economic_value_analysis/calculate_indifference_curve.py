from __future__ import annotations

import csv
import json

from common import INPUTS_DIR, RESULTS_DIR, load_analysis_config


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def profile_energy_factor_mwh_per_mw() -> float:
    profile_path = INPUTS_DIR / "ppa_profile.csv"
    with profile_path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return sum(float(row["sdp"]) for row in rows)


def main() -> None:
    cfg = load_analysis_config()
    merchant = read_json(RESULTS_DIR / "merchant" / "summary.json")
    merchant_objective = float(merchant["objective_value"])
    energy_factor = profile_energy_factor_mwh_per_mw()

    rows = []
    for point in cfg["validation_points"]:
        reference_price = float(point["lambda_ppa_reference"])
        capacity = float(point["s_ppa_fixed"])
        reference_objective = float(point["reference_objective"])
        contracted_energy = capacity * energy_factor
        objective_surplus = reference_objective - merchant_objective
        price_surplus = objective_surplus / contracted_energy
        indifference_price = reference_price - price_surplus
        rows.append(
            {
                "reference_lambda_ppa": reference_price,
                "s_ppa_fixed": capacity,
                "contracted_energy_mwh": contracted_energy,
                "reference_objective": reference_objective,
                "merchant_objective": merchant_objective,
                "objective_surplus": objective_surplus,
                "price_surplus_eur_per_mwh": price_surplus,
                "risk_adjusted_indifference_price_eur_per_mwh": indifference_price,
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "indifference_curve_summary.json"
    csv_path = RESULTS_DIR / "indifference_curve_summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== RISK-ADJUSTED PPA INDIFFERENCE CURVE ===")
    for row in rows:
        print(
            f"Reference {row['reference_lambda_ppa']:.0f} EUR/MWh: "
            f"S_PPA={row['s_ppa_fixed']:.6f} MW, "
            "indifference price="
            f"{row['risk_adjusted_indifference_price_eur_per_mwh']:.6f} EUR/MWh"
        )
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
