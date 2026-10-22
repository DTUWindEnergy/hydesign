"""Calculate delivery-profile-weighted market and wind indicators."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
THESIS_ROOT = HERE.parents[1]
INPUTS = THESIS_ROOT / "inputs" / "Input_files_created"
HOURLY_DELIVERY = HERE / "outputs" / "data" / "contracted_ppa_delivery_hourly.csv"
OUTPUT = HERE / "outputs" / "data" / "delivery_profile_market_indicators.csv"
HOURS_PER_DAY = 24


def read_profiles() -> dict[str, dict[int, float]]:
    profiles: dict[str, dict[int, float]] = {}
    with HOURLY_DELIVERY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            profiles.setdefault(row["sdp_label"], {}).setdefault(
                int(row["hour"]), float(row["profile_factor"])
            )
    for name, shape in profiles.items():
        if set(shape) != set(range(HOURS_PER_DAY)):
            raise ValueError(f"Profile {name} does not contain exactly 24 hours.")
    return profiles


def read_probabilities() -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    with (INPUTS / "omega_probabilities.csv").open(newline="", encoding="utf-8") as handle:
        omega = {
            int(row["omega"]): float(row["prob"])
            for row in csv.DictReader(handle)
        }
    with (INPUTS / "xi_probabilities.csv").open(newline="", encoding="utf-8") as handle:
        xi = {
            (int(row["omega"]), int(row["xi"])): float(row["prob"])
            for row in csv.DictReader(handle)
        }
    return omega, xi


def aggregate_spot(omega_prob: dict[int, float]) -> tuple[list[float], list[float]]:
    weights = [0.0] * HOURS_PER_DAY
    values = [0.0] * HOURS_PER_DAY
    with (INPUTS / "spot_prices.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            hour = (int(row["t"]) - 1) % HOURS_PER_DAY
            probability = omega_prob[int(row["omega"])]
            weights[hour] += probability
            values[hour] += probability * float(row["price"])
    return weights, values


def read_unique_scenario_hour_prices(filename: str) -> dict[tuple[int, int], float]:
    """Read one price per (t, omega), dropping the repeated xi rows."""
    prices = {}
    with (INPUTS / filename).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["t"]), int(row["omega"]))
            if key not in prices:
                prices[key] = float(row["price"])
    return prices


def aggregate_system_regulation(
    omega_prob: dict[int, float],
) -> tuple[list[float], list[float], list[float]]:
    """Aggregate system-level up/down values using the dominant direction."""
    all_weights = [0.0] * HOURS_PER_DAY
    up_values = [0.0] * HOURS_PER_DAY
    down_values = [0.0] * HOURS_PER_DAY

    spot = read_unique_scenario_hour_prices("spot_prices.csv")
    up = read_unique_scenario_hour_prices("mfrr_up_prices.csv")
    down = read_unique_scenario_hour_prices("mfrr_down_prices.csv")
    imbalance = read_unique_scenario_hour_prices("imbalance_prices.csv")

    if not (spot.keys() == up.keys() == down.keys() == imbalance.keys()):
        raise ValueError("The prepared market-price files do not contain matching hours.")

    for (t, omega), spot_price in spot.items():
        up_price = up[(t, omega)]
        down_price = down[(t, omega)]
        imbalance_price = imbalance[(t, omega)]
        up_signal = up_price > spot_price
        down_signal = down_price < spot_price

        system_y_up = False
        system_y_down = False
        if up_signal and not down_signal:
            system_y_up = True
        elif down_signal and not up_signal:
            system_y_down = True
        elif up_signal and down_signal:
            # The prepared imbalance price encodes the dominant direction.
            up_distance = abs(imbalance_price - up_price)
            down_distance = abs(imbalance_price - down_price)
            if up_distance <= down_distance:
                system_y_up = True
            else:
                system_y_down = True

        hour = (t - 1) % HOURS_PER_DAY
        probability = omega_prob[omega]
        all_weights[hour] += probability
        if system_y_up:
            up_values[hour] += probability * up_price
        if system_y_down:
            down_values[hour] += probability * down_price

    return all_weights, up_values, down_values


def aggregate_wind(
    omega_prob: dict[int, float],
    xi_prob: dict[tuple[int, int], float],
    p_nom: float,
) -> tuple[list[float], list[float]]:
    weights = [0.0] * HOURS_PER_DAY
    values = [0.0] * HOURS_PER_DAY
    with (INPUTS / "wind_realization.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            omega = int(row["omega"])
            xi = int(row["xi"])
            hour = (int(row["t"]) - 1) % HOURS_PER_DAY
            probability = omega_prob[omega] * xi_prob[(omega, xi)]
            weights[hour] += probability
            values[hour] += probability * float(row["wind"]) / p_nom
    return weights, values


def weighted_sum(shape: dict[int, float], values: list[float]) -> float:
    return sum(shape[hour] * values[hour] for hour in range(HOURS_PER_DAY))


def main() -> None:
    profiles = read_profiles()
    omega_prob, xi_prob = read_probabilities()
    with (INPUTS / "config.json").open(encoding="utf-8") as handle:
        p_nom = float(json.load(handle)["P_nom"])

    spot_weight, spot_value = aggregate_spot(omega_prob)
    regulation_weight, up_price, down_price = aggregate_system_regulation(
        omega_prob
    )
    wind_weight, wind_value = aggregate_wind(omega_prob, xi_prob, p_nom)

    rows = []
    for name, shape in profiles.items():
        spot_denominator = weighted_sum(shape, spot_weight)
        regulation_denominator = weighted_sum(shape, regulation_weight)
        wind_denominator = weighted_sum(shape, wind_weight)
        rows.append(
            {
                "profile": name,
                "profile_weighted_spot_price": weighted_sum(shape, spot_value)
                / spot_denominator,
                "profile_weighted_expected_mfrr_up_value": weighted_sum(
                    shape, up_price
                )
                / regulation_denominator,
                "profile_weighted_expected_mfrr_down_value": -weighted_sum(
                    shape, down_price
                )
                / regulation_denominator,
                "delivery_weighted_wind_cf": weighted_sum(shape, wind_value)
                / wind_denominator,
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {OUTPUT}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
