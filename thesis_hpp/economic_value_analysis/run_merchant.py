from common import base_summary, solve_fixed_capacity, write_summary


def main() -> None:
    model, results, elapsed_seconds, cfg = solve_fixed_capacity(0.0, 0.0)
    summary = base_summary(model, results, elapsed_seconds, cfg)
    summary["run_type"] = "merchant"
    output_path = write_summary("merchant/summary.json", summary)
    print("\n=== MERCHANT CASE ===")
    print(f"Objective: {summary['objective_value']}")
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
