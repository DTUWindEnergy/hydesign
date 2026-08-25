from dataclasses import dataclass
import re
import pandas as pd


@dataclass(frozen=True)
class DeliveryProfile:
    name: str
    profile_id: int
    hourly_shape: list[float] | None


DELIVERY_PROFILES = {
    "Seasonally Free": DeliveryProfile("Seasonally Free", 0, None),
    "Monthly Free": DeliveryProfile("Monthly Free", 1, None),

    "Monthly Continuous Baseload": DeliveryProfile("Monthly Continuous Baseload", 2, [1] * 24),
    "Seasonal Continuous Baseload": DeliveryProfile("Seasonal Continuous Baseload", 3, [1] * 24),
    "Yearly Continuous Baseload": DeliveryProfile("Yearly Continuous Baseload", 4, [1] * 24),

    "Generic Industrial Monthly": DeliveryProfile("Generic Industrial Monthly", 5, [
        0.593030735, 0.554395406, 0.546245849, 0.53915524, 0.552433083, 0.620155643,
        0.732682151, 0.84368243, 0.927410869, 0.975923914, 0.990102265, 0.987748587,
        0.988789471, 1, 0.996437111, 0.98324188, 0.960533801, 0.946952824,
        0.911729005, 0.878990223, 0.811109466, 0.750608622, 0.682631749, 0.632859476
    ]),
    "Generic Industrial Seasonal": DeliveryProfile("Generic Industrial Seasonal", 6, [
        0.593030735, 0.554395406, 0.546245849, 0.53915524, 0.552433083, 0.620155643,
        0.732682151, 0.84368243, 0.927410869, 0.975923914, 0.990102265, 0.987748587,
        0.988789471, 1, 0.996437111, 0.98324188, 0.960533801, 0.946952824,
        0.911729005, 0.878990223, 0.811109466, 0.750608622, 0.682631749, 0.632859476
    ]),
    "Generic Industrial Yearly": DeliveryProfile("Generic Industrial Yearly", 7, [
        0.593030735, 0.554395406, 0.546245849, 0.53915524, 0.552433083, 0.620155643,
        0.732682151, 0.84368243, 0.927410869, 0.975923914, 0.990102265, 0.987748587,
        0.988789471, 1, 0.996437111, 0.98324188, 0.960533801, 0.946952824,
        0.911729005, 0.878990223, 0.811109466, 0.750608622, 0.682631749, 0.632859476
    ]),

    "Generic Private Monthly": DeliveryProfile("Generic Private Monthly", 8, [
        0.405392308, 0.363868092, 0.331129724, 0.302547855, 0.285679627, 0.299305534,
        0.411206526, 0.52205873, 0.507150756, 0.490946791, 0.486622381, 0.495561923,
        0.51057742, 0.511541268, 0.521249321, 0.551870621, 0.655318008, 0.892684358,
        1, 0.938593833, 0.829571565, 0.732613459, 0.604970492, 0.477711519
    ]),
    "Generic Private Seasonal": DeliveryProfile("Generic Private Seasonal", 9, [
        0.405392308, 0.363868092, 0.331129724, 0.302547855, 0.285679627, 0.299305534,
        0.411206526, 0.52205873, 0.507150756, 0.490946791, 0.486622381, 0.495561923,
        0.51057742, 0.511541268, 0.521249321, 0.551870621, 0.655318008, 0.892684358,
        1, 0.938593833, 0.829571565, 0.732613459, 0.604970492, 0.477711519
    ]),
    "Generic Private Yearly": DeliveryProfile("Generic Private Yearly", 10, [
        0.405392308, 0.363868092, 0.331129724, 0.302547855, 0.285679627, 0.299305534,
        0.411206526, 0.52205873, 0.507150756, 0.490946791, 0.486622381, 0.495561923,
        0.51057742, 0.511541268, 0.521249321, 0.551870621, 0.655318008, 0.892684358,
        1, 0.938593833, 0.829571565, 0.732613459, 0.604970492, 0.477711519
    ]),

    "Inverse Solar Monthly": DeliveryProfile("Inverse Solar Monthly", 11, [
        1, 1, 1, 0.997772592, 0.919229953, 0.757901994, 0.558071701, 0.373355961,
        0.201633432, 0.092013152, 0.010288502, 0, 0.069633008, 0.145736105,
        0.277047094, 0.460065762, 0.650774289, 0.848589308, 0.97300594,
        0.999363598, 1, 1, 1, 1
    ]),
    "Inverse Solar Seasonal": DeliveryProfile("Inverse Solar Seasonal", 12, [
        1, 1, 1, 0.997772592, 0.919229953, 0.757901994, 0.558071701, 0.373355961,
        0.201633432, 0.092013152, 0.010288502, 0, 0.069633008, 0.145736105,
        0.277047094, 0.460065762, 0.650774289, 0.848589308, 0.97300594,
        0.999363598, 1, 1, 1, 1
    ]),
    "Inverse Solar Yearly": DeliveryProfile("Inverse Solar Yearly", 13, [
        1, 1, 1, 0.997772592, 0.919229953, 0.757901994, 0.558071701, 0.373355961,
        0.201633432, 0.092013152, 0.010288502, 0, 0.069633008, 0.145736105,
        0.277047094, 0.460065762, 0.650774289, 0.848589308, 0.97300594,
        0.999363598, 1, 1, 1, 1
    ]),

    "Peak Monthly": DeliveryProfile("Peak Monthly", 14, [
        0.653050827, 0.625423729, 0.62006779, 0.663525441, 0.781966078, 0.87440682,
        0.882271149, 0.789559315, 0.779661017, 0.778881369, 0.728813559, 0.703389831,
        0.682305081, 0.68572879, 0.714915234, 0.765084736, 0.818169495, 0.986915305,
        1, 0.903254207, 0.826033875, 0.771186441, 0.681050861, 0.620338983
    ]),
    "Peak Seasonal": DeliveryProfile("Peak Seasonal", 15, [
        0.653050827, 0.625423729, 0.62006779, 0.663525441, 0.781966078, 0.87440682,
        0.882271149, 0.789559315, 0.779661017, 0.778881369, 0.728813559, 0.703389831,
        0.682305081, 0.68572879, 0.714915234, 0.765084736, 0.818169495, 0.986915305,
        1, 0.903254207, 0.826033875, 0.771186441, 0.681050861, 0.620338983
    ]),
    "Peak Yearly": DeliveryProfile("Peak Yearly", 16, [
        0.653050827, 0.625423729, 0.62006779, 0.663525441, 0.781966078, 0.87440682,
        0.882271149, 0.789559315, 0.779661017, 0.778881369, 0.728813559, 0.703389831,
        0.682305081, 0.68572879, 0.714915234, 0.765084736, 0.818169495, 0.986915305,
        1, 0.903254207, 0.826033875, 0.771186441, 0.681050861, 0.620338983
    ]),

    "Binary Peak Monthly": DeliveryProfile("Binary Peak Monthly", 17, [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    "Binary Peak Seasonal": DeliveryProfile("Binary Peak Seasonal", 18, [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
    "Binary Peak Yearly": DeliveryProfile("Binary Peak Yearly", 19, [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
}


def get_delivery_profile(name: str) -> DeliveryProfile:
    try:
        profile = DELIVERY_PROFILES[name]
    except KeyError as exc:
        valid = ", ".join(DELIVERY_PROFILES)
        raise KeyError(f"Unknown delivery profile '{name}'. Valid profiles: {valid}") from exc

    if profile.hourly_shape is None:
        raise ValueError(f"Delivery profile '{name}' has no fixed hourly shape.")

    if len(profile.hourly_shape) != 24:
        raise ValueError(f"Delivery profile '{name}' must have exactly 24 hourly values.")

    return profile


def make_ppa_profile(t_values, profile_name: str) -> pd.DataFrame:
    profile = get_delivery_profile(profile_name)
    rows = []

    for t in t_values:
        hour_index = (int(t) - 1) % 24
        rows.append({
            "t": int(t),
            "sdp": float(profile.hourly_shape[hour_index])
        })

    return pd.DataFrame(rows)


def profile_filename(profile_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", profile_name.lower()).strip("_")
    return f"ppa_profile_{slug}.csv"
