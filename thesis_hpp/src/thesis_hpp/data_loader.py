from pathlib import Path
import json
import pandas as pd


def _series_to_dict(df, key_cols, value_col):
    out = {}
    for _, row in df.iterrows():
        key = tuple(row[c] for c in key_cols)
        if len(key) == 1:
            key = key[0]
        out[key] = float(row[value_col])
    return out


def load_all_data(inputs_path: Path):
    inputs_path = Path(inputs_path)

    with open(inputs_path / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    spot = pd.read_csv(inputs_path / "spot_prices.csv")
    mfrr_up = pd.read_csv(inputs_path / "mfrr_up_prices.csv")
    mfrr_down = pd.read_csv(inputs_path / "mfrr_down_prices.csv")
    imb = pd.read_csv(inputs_path / "imbalance_prices.csv")
    ppa_profile = pd.read_csv(inputs_path / "ppa_profile.csv")
    omega_prob = pd.read_csv(inputs_path / "omega_probabilities.csv")
    xi_prob = pd.read_csv(inputs_path / "xi_probabilities.csv")
    wind_forecast = pd.read_csv(inputs_path / "wind_forecast.csv")
    wind_real = pd.read_csv(inputs_path / "wind_realization.csv")
    activation_limits = pd.read_csv(inputs_path / "activation_limits.csv")

    T = sorted(spot["t"].unique().tolist())
    Omega = sorted(spot["omega"].unique().tolist())
    Xi = sorted(xi_prob["xi"].unique().tolist())

    data = {
        "config": config,
        "T": T,
        "Omega": Omega,
        "Xi": Xi,

        "lambda_spot": _series_to_dict(spot, ["t", "omega"], "price"),
        "lambda_mfrr_up": _series_to_dict(mfrr_up, ["t", "omega", "xi"], "price"),
        "lambda_mfrr_down": _series_to_dict(mfrr_down, ["t", "omega", "xi"], "price"),
        "lambda_imb": _series_to_dict(imb, ["t", "omega", "xi"], "price"),
        "y_up": _series_to_dict(mfrr_up, ["t", "omega", "xi"], "y_up"),
        "y_down": _series_to_dict(mfrr_down, ["t", "omega", "xi"], "y_down"),

        "sdp": _series_to_dict(ppa_profile, ["t"], "sdp"),
        "pi_omega": _series_to_dict(omega_prob, ["omega"], "prob"),
        "pi_xi_given_omega": _series_to_dict(xi_prob, ["omega", "xi"], "prob"),

        "wind_forecast": _series_to_dict(wind_forecast, ["t", "omega"], "wind"),
        "wind_real": _series_to_dict(wind_real, ["t", "omega", "xi"], "wind"),
        "A_up_max": _series_to_dict(activation_limits, ["t", "omega"], "A_up_max"),
        "A_down_max": _series_to_dict(activation_limits, ["t", "omega"], "A_down_max"),
    }

    return data
