"""Regenerate the compact HKN data used by the coupled-operation tutorial.

This script is intentionally not executed by the notebook. It requires the
optional design-friendly control and wind-farm-load surrogate repositories and
takes several minutes. The resulting CSV keeps the public tutorial lightweight.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from design_friendly.utils.get_flowmodel import get_flowmodel
from design_friendly.utils.iea22s import IEA22s
from design_friendly.utils.sites import geometric_median, polygon_area, scale_by_D
from design_friendly.utils.sites_data import HKN_boundaries, HKN_x, HKN_y
from wind_farm_loads.py_wake import PropagateDownwindNoSelfInduction

import hydesign
from hydesign.weather.weather import interpolate_WD, interpolate_WS_loglog
from hydesign.wind.pywake_wpp import PyWakeWPP

EXAMPLE_ROOT = Path(hydesign.__file__).resolve().parent / "examples"
OUTPUT_ROOT = EXAMPLE_ROOT / "SUDOCO"
START_HOUR = 312
N_HOURS = 168


def hkn_layout(diameter=284.0):
    """Return the HKN layout and boundary, scaled to the IEA 22 MW turbine."""
    center_x, center_y = geometric_median(HKN_x, HKN_y)
    x, y = scale_by_D(
        200.0, HKN_x - center_x, HKN_y - center_y, diameter, center=(0, 0)
    )
    boundary_x, boundary_y = scale_by_D(
        200.0,
        HKN_boundaries[:, 0] - center_x,
        HKN_boundaries[:, 1] - center_y,
        diameter,
        center=(0, 0),
    )
    return x, y, np.column_stack([boundary_x, boundary_y])


weather = pd.read_csv(
    EXAMPLE_ROOT / "Europe" / "GWA2" / "input_ts_HKN.csv",
    index_col=0,
    parse_dates=True,
).iloc[START_HOUR : START_HOUR + N_HOURS]
mdf = pd.read_csv(
    EXAMPLE_ROOT / "Europe" / "df_mdf_wind_nl_20251219_1451_[2023, 2024].csv",
    index_col=0,
)
mdf.index = pd.to_datetime(mdf.index, format="%d/%m/%Y %H:%M")
mdf = mdf.loc[weather.index, "MDFwind_kgCO2eperMWh"]

wt = IEA22s()
diameter = float(wt.diameter().item())
hub_height = float(wt.hub_height().item())
x, y, boundary = hkn_layout(diameter)
n_wt = len(x)
rated_power_mw = 22.0
grid_mw = n_wt * rated_power_mw

components = [
    {
        "name": "Main bearing",
        "beta": 2.5,
        "C_replace": 0.45,
        "T_replace_days": 90,
        "number": 1,
        "failed_after_design_life": 0.25,
        "cost_fraction_of_capex": 0.6,
        "downtime": 270,
    },
    {
        "name": "Blades",
        "beta": 2.5,
        "C_replace": 0.45,
        "T_replace_days": 90,
        "number": 3,
        "failed_after_design_life": 0.1,
        "cost_fraction_of_capex": 0.2,
        "downtime": 90,
    },
    {
        "name": "Pitch bearing",
        "beta": 2.5,
        "C_replace": 0.15,
        "T_replace_days": 30,
        "number": 3,
        "failed_after_design_life": 0.1,
        "cost_fraction_of_capex": 0.2,
        "downtime": 90,
    },
    {
        "name": "Yaw bearing",
        "beta": 2.5,
        "C_replace": 0.3,
        "T_replace_days": 60,
        "number": 0,
        "failed_after_design_life": 0.1,
        "cost_fraction_of_capex": 0.6,
        "downtime": 270,
    },
]
load_sensors = ["RA_fbrm", "RA_tbfa", "RA_tbss", "RA_ttfa"]
load_sensors_used = ["RA_fbrm", "RA_fbrm", "RA_fbrm", "RA_tbss"]

farm = get_flowmodel(wt=wt, propagate=PropagateDownwindNoSelfInduction)
wpp = PyWakeWPP(
    n_wt=n_wt,
    x=x,
    y=y,
    tilt=np.zeros(N_HOURS),
    time_stamp=np.arange(N_HOURS),
    sim_pars={"TI": 0.06},
    load_sensors=load_sensors,
    load_sensors_used=load_sensors_used,
    components=components,
    _LIFETIME=25,
    wpp_efficiency=0.95,
    farm=farm,
    N_time=N_HOURS,
    G_MW=grid_mw,
    _OPEX=1.42,
)

wind_controlled, wind_reference, yaw, variable_opex, loads, loads_ref = wpp.compute(
    wst=interpolate_WS_loglog(weather, hub_height).WS.to_numpy(),
    wd=interpolate_WD(weather, hub_height),
)

data = pd.DataFrame(
    {
        "timestamp": weather.index,
        "source_hour": np.arange(START_HOUR, START_HOUR + N_HOURS),
        "price_eur_per_mwh": weather["Price"].to_numpy(),
        "mdf_kgco2e_per_mwh": mdf.to_numpy(),
        "wind_reference_mw": wind_reference,
        "wind_controlled_mw": wind_controlled,
        "variable_opex_eur": variable_opex,
        "yaw_rms_deg": np.sqrt(np.mean(np.square(yaw), axis=0)),
        "load_ratio": np.mean(loads[0], axis=0) / np.mean(loads_ref[0], axis=0),
    }
)
data.to_csv(OUTPUT_ROOT / "hkn_yaw_modes_168h.csv", index=False)
pd.DataFrame({"x_m": x, "y_m": y}).to_csv(OUTPUT_ROOT / "hkn_layout.csv", index=False)

print(f"HKN turbines: {n_wt}")
print(f"HKN area: {polygon_area(boundary) / 1e6:.1f} km2")
print(f"Saved {len(data)} hours to {OUTPUT_ROOT}")
