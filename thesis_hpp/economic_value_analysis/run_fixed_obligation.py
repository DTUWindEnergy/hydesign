from common import base_summary, load_analysis_config, solve_fixed_capacity, write_summary


def main() -> None:
    cfg = load_analysis_config()
    model, results, elapsed_seconds, cfg = solve_fixed_capacity(
        0.0, float(cfg["s_ppa_fixed"])
    )
    summary = base_summary(model, results, elapsed_seconds, cfg)
    summary["run_type"] = "fixed_obligation_zero_price"
    summary["reference_lambda_ppa"] = float(cfg["lambda_ppa_reference"])
    summary["reference_objective"] = float(cfg["reference_objective"])
    output_path = write_summary("fixed_obligation/summary.json", summary)
    print("\n=== FIXED OBLIGATION AT ZERO PPA PRICE ===")
    print(f"Fixed S_PPA: {summary['s_ppa']}")
    print(f"Objective: {summary['objective_value']}")
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
