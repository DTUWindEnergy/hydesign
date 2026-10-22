import json

from common import RESULTS_DIR, load_analysis_config, write_summary


def read_summary(relative_path: str) -> dict:
    path = RESULTS_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the required solve first.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    cfg = load_analysis_config()
    merchant = read_summary("merchant/summary.json")
    obligation = read_summary("fixed_obligation/summary.json")
    j_merchant = float(merchant["objective_value"])
    j_obligation = float(obligation["objective_value"])
    energy = float(obligation["contracted_energy_mwh"])
    lambda_reference = float(cfg["lambda_ppa_reference"])
    j_reference = float(cfg["reference_objective"])
    if energy <= 0:
        raise ValueError("Contracted PPA energy must be positive.")

    direct_price = (j_merchant - j_obligation) / energy
    shortcut_surplus = (j_reference - j_merchant) / energy
    shortcut_price = lambda_reference - shortcut_surplus
    implied_obligation = j_reference - lambda_reference * energy
    summary = {
        "s_ppa_fixed": float(cfg["s_ppa_fixed"]),
        "contracted_energy_mwh": energy,
        "merchant_objective": j_merchant,
        "fixed_obligation_objective": j_obligation,
        "reference_lambda_ppa": lambda_reference,
        "reference_objective": j_reference,
        "direct_indifference_price_eur_per_mwh": direct_price,
        "shortcut_price_surplus_eur_per_mwh": shortcut_surplus,
        "shortcut_indifference_price_eur_per_mwh": shortcut_price,
        "implied_obligation_objective_from_shortcut": implied_obligation,
        "obligation_objective_difference": j_obligation - implied_obligation,
        "indifference_price_difference": direct_price - shortcut_price,
    }
    output_path = write_summary("indifference_price_summary.json", summary)
    print("\n=== RISK-ADJUSTED INDIFFERENCE PRICE ===")
    print(f"Direct price: {direct_price:.10f} EUR/MWh")
    print(f"Shortcut price: {shortcut_price:.10f} EUR/MWh")
    print(f"Difference: {direct_price - shortcut_price:.10f} EUR/MWh")
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
