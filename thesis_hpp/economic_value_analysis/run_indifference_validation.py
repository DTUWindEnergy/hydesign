import json

from common import (
    RESULTS_DIR,
    base_summary,
    load_analysis_config,
    solve_fixed_capacity,
    write_summary,
)


def main() -> None:
    cfg = load_analysis_config()
    price_path = RESULTS_DIR / "indifference_price_summary.json"
    if not price_path.exists():
        raise FileNotFoundError(
            f"Missing {price_path}. Run calculate_indifference_price.py first."
        )
    with price_path.open("r", encoding="utf-8") as handle:
        price_summary = json.load(handle)

    price = float(price_summary["direct_indifference_price_eur_per_mwh"])
    model, results, elapsed_seconds, cfg = solve_fixed_capacity(
        price, float(cfg["s_ppa_fixed"])
    )
    summary = base_summary(model, results, elapsed_seconds, cfg)
    summary["run_type"] = "fixed_capacity_indifference_price_validation"
    summary["merchant_objective"] = float(price_summary["merchant_objective"])
    summary["objective_difference_from_merchant"] = (
        summary["objective_value"] - summary["merchant_objective"]
    )
    output_path = write_summary("indifference_validation/summary.json", summary)
    print("\n=== INDIFFERENCE-PRICE VALIDATION ===")
    print(f"Fixed S_PPA: {summary['s_ppa']}")
    print(f"Indifference price: {summary['lambda_ppa']}")
    print(f"Objective: {summary['objective_value']}")
    print(f"Merchant objective: {summary['merchant_objective']}")
    print(f"Difference: {summary['objective_difference_from_merchant']:.10f}")
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
