"""Generate consistent N2 and XDSM views of the HKN GNN-loads assembly.

The OpenMDAO runtime assembly in ``hpp_pywake_gnn_loads.hpp_model`` promotes
all variables and therefore hides both its constructor arguments and the
pre-computed Design-Friendly yaw map.  This script builds a *viewer-only*
OpenMDAO problem from the same ordered component interfaces.  It adds:

* EGO as the sizing driver, including the ``Finance.NPV -> EGO`` objective;
* HKN data as the source of fixed and constructor-time inputs; and
* the Design-Friendly GNN yaw-map preprocessing step.

No component is evaluated.  The proxy merely lets OpenMDAO write a genuine,
interactive N2 diagram containing the complete interface.  The XDSM is made
from the exact same ``SYSTEMS`` and ``CONNECTIONS`` manifest.

Naming convention
-----------------
Scalar integers/floats use capitals (``H``, ``N_WT``, ``B_E``).  Hourly
vectors for the representative year end in ``_y`` and hourly lifetime vectors
end in ``_l``.  Other arrays have meaningful subscripts such as ``_a`` for an
annual vector, ``_g`` for a lookup grid, and ``_i``/``_s`` for turbine/sensor.

Run from the repository root with::

    python hydesign/examples/SUDOCO/make_xdsm_coupled_hpp_gnn_loads.py

Use ``--tex`` to additionally write PyXDSM sources, or ``--pdf`` to compile
them when a LaTeX installation is available.  The generator also writes an
editable CSV mapping every abbreviated system/variable to its HyDesign name.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om
from matplotlib.patches import Polygon, Rectangle
from pyxdsm.XDSM import FUNC, METAMODEL, OPT, XDSM


@dataclass(frozen=True)
class System:
    key: str
    label: str
    kind: str = FUNC


@dataclass(frozen=True)
class Connection:
    source: str
    target: str
    # A (source, target) pair represents an OpenMDAO promoted-name remapping.
    variables: tuple[str | tuple[str, str], ...]


# The 17 runtime subsystems retain the order in hpp_pywake_gnn_loads.hpp_model.
# EGO, HKN data, GNN and the evaluate mapping are visible orchestration steps.
SYSTEMS = (
    System("EGO", "EGO", OPT),
    System("HKN", "HKN data"),
    System("GNN", "GNN yaw", METAMODEL),
    System("EVAL", "Eval. map"),
    System("ABL", "ABL"),
    System("PW", "PW + VOM"),
    System("PV", "PV"),
    System("EMS", "EMS", OPT),
    System("PW_ACT", "PW act."),
    System("B_DEG", "B deg."),
    System("B_TMP", "B temp."),
    System("PV_DEG", "PV deg."),
    System("B_REL", "B rel."),
    System("W_REL", "W rel."),
    System("PV_REL", "PV rel."),
    System("LT_EMS", "LT EMS"),
    System("W_COST", "W cost"),
    System("PV_COST", "PV cost"),
    System("B_COST", "B cost"),
    System("SH_COST", "Sh. cost"),
    System("FIN", "Finance"),
)


# Every input in the viewer problem is connected.  Variables on HKN edges are
# constructor-time or fixed runtime values; the remaining edges mirror the
# promoted-variable links of the real OpenMDAO model.
CONNECTIONS = (
    Connection(
        "EGO",
        "EVAL",
        (
            "CLR",
            "SP",
            "P_R",
            "N_WT",
            "RHO_W",
            "B_P",
            "B_E_H",
        ),
    ),
    Connection("EGO", "PV", ("TILT", "AZI", "R_DC_AC", "P_PV")),
    Connection("EGO", "EMS", ("B_P", "C_BFLUCT")),
    Connection("EGO", "LT_EMS", ("B_P",)),
    Connection("EGO", "W_COST", ("N_WT", "P_R")),
    Connection("EGO", "PV_COST", ("P_PV", "R_DC_AC")),
    Connection("EGO", "B_COST", ("B_P",)),
    Connection("HKN", "GNN", ("x_i", "y_i", "ws_g", "wd_g", "WT_GNN", "TI")),
    Connection("HKN", "ABL", ("WEATHER_FN", "N_Y", "INTERP_WD")),
    Connection(
        "HKN",
        "PW",
        (
            "FARM",
            "FARM_LOAD",
            "x_i",
            "y_i",
            "tilt_y",
            "t_y",
            "TI",
            "ETA_W",
            "CAPEX_REF",
            "LIFE",
            "OPEX_REF",
            "LOAD_CFG",
        ),
    ),
    Connection(
        "HKN",
        "PV",
        ("WEATHER_FN", "N_Y", "LAT", "LON", "ALT", "TRACKING", "A_PV_MW"),
    ),
    Connection(
        "HKN",
        "EMS",
        (
            "N_Y",
            "WEEKS",
            "LIFE",
            "EMS_TYPE",
            "YAW_MODE",
            "BATCH",
            "price_y",
            "mdf_y",
            "V_CO2",
            "DOD",
            "ETA_B",
            "Q_PK",
            "H_PK",
        ),
    ),
    Connection("HKN", "B_DEG", ("WEATHER_FN", "N_B_MAX", "LIFE", "WEEKS", "MIN_LOH")),
    Connection("HKN", "B_TMP", ("WEATHER_FN", "LIFE", "WEEKS")),
    Connection("HKN", "PV_DEG", ("LIFE", "pv_deg_yr_a", "pv_deg_a")),
    Connection("HKN", "B_REL", ("LIFE", "rel_b_l", "rel_tr_l")),
    Connection("HKN", "W_REL", ("LIFE", "rel_w_l", "rel_tr_l")),
    Connection("HKN", "PV_REL", ("LIFE", "rel_pv_l", "rel_tr_l")),
    Connection(
        "HKN", "LT_EMS", ("N_Y", "LIFE", "LOAD_MIN", "DOD", "ETA_B", "Q_PK", "H_PK")
    ),
    Connection(
        "HKN",
        "W_COST",
        ("C_WT", "C_W_CIV", "C_W_FIX", "C_W_VAR", "D_REF", "H_REF", "P_R_REF", "N_Y"),
    ),
    Connection("HKN", "PV_COST", ("C_PV", "C_PV_INST", "C_INV", "C_PV_FIX")),
    Connection(
        "HKN",
        "B_COST",
        ("C_B_E", "C_B_P", "C_B_BOP", "C_B_CTRL", "C_B_ONM", "LIFE", "R_B_COST"),
    ),
    Connection("HKN", "SH_COST", ("C_BOS", "C_GRID", "C_LAND")),
    Connection(
        "HKN",
        "FIN",
        (
            "N_Y",
            "LIFE",
            "dep_yr_a",
            "dep_a",
            "infl_yr_a",
            "infl_a",
            "REF_INFL_YR",
            "phase_yr_a",
            "phase_capex_a",
            "WACC_W",
            "WACC_PV",
            "WACC_B",
            "TAX",
            "DECOM_W",
            "DECOM_PV",
        ),
    ),
    Connection("GNN", "PW", ("yaw_i_g",)),
    Connection("EVAL", "ABL", ("H",)),
    Connection("EVAL", "EMS", ("B_E",)),
    Connection("EVAL", "LT_EMS", ("B_E",)),
    Connection("EVAL", "W_COST", ("H", "D")),
    Connection("EVAL", "B_COST", ("B_E",)),
    Connection("EVAL", "SH_COST", ("A_W",)),
    Connection("HKN", "EMS", ("G",)),
    Connection("HKN", "LT_EMS", ("G",)),
    Connection("HKN", "SH_COST", ("G",)),
    Connection("ABL", "PW", ("ws_y", "wd_y")),
    Connection("PW", "EMS", ("p_w_y", "p_w_ref_y", "m_y")),
    Connection("PV", "EMS", ("p_pv_y",)),
    Connection(
        "PW",
        "PW_ACT",
        (
            "p_w_y",
            "p_w_ref_y",
            "yaw_i_y",
            "m_y",
            "m_exp_s",
            "del_s_i_y",
            "del_ref_s_i_y",
        ),
    ),
    Connection("EMS", "PW_ACT", ("lambda_y",)),
    Connection("EMS", "B_DEG", ("soc_l",)),
    Connection("B_DEG", "B_TMP", ("soh_l",)),
    Connection("EMS", "PV_DEG", ("p_pv_l",)),
    Connection("EMS", "B_REL", ("b_l",)),
    Connection("EMS", "W_REL", ("p_w_l",)),
    Connection("PV_DEG", "PV_REL", ("p_pv_deg_l",)),
    Connection("B_TMP", "LT_EMS", ("soh_all_l",)),
    Connection("W_REL", "LT_EMS", ("p_w_rel_l",)),
    Connection("PV_REL", "LT_EMS", ("p_pv_rel_l",)),
    Connection("B_REL", "LT_EMS", ("b_rel_l",)),
    Connection(
        "EMS",
        "LT_EMS",
        ("p_w_l", "p_pv_l", "price_l", "p_curt_l", "soc_l"),
    ),
    Connection("PW", "W_COST", ("p_w_y",)),
    Connection("PW_ACT", "W_COST", ("M_ACT",)),
    Connection("B_DEG", "B_COST", ("soh_l",)),
    Connection("PV", "SH_COST", ("A_PV",)),
    Connection("EMS", "FIN", ("price_l",)),
    Connection("LT_EMS", "FIN", ("p_hpp_deg_l", "pen_deg_l")),
    Connection("W_COST", "FIN", ("CAPEX_W", "OPEX_W")),
    Connection("PV_COST", "FIN", ("CAPEX_PV", "OPEX_PV")),
    Connection("B_COST", "FIN", ("CAPEX_B", "OPEX_B")),
    Connection(
        "SH_COST",
        "FIN",
        (
            "CAPEX_SH",
            "OPEX_SH",
            "CAPEX_SH_FIX",
            "CAPEX_SH_W",
            "CAPEX_SH_PV",
            ("CAPEX_SH", "CAPEX_EL"),
            ("OPEX_SH", "OPEX_EL"),
        ),
    ),
    # Deliberate lower-triangular feedback: NPV is the EGO sizing objective.
    Connection("FIN", "EGO", ("NPV",)),
)


# Outputs with no downstream consumer are still useful in the N2 tree.
TERMINAL_OUTPUTS = {
    "EMS": ("p_hpp_l", "pen_l"),
    "PW_ACT": ("p_w_act_y", "yaw_act_i_y", "del_act_s_i_y", "ldel_s_i", "ldel_ref_s_i"),
    "B_DEG": ("N_B",),
    "LT_EMS": ("p_curt_deg_l", "b_deg_l", "soc_deg_l", "CURT", "CURT_DEG"),
    "FIN": (
        "CAPEX",
        "OPEX",
        "REV",
        "NPV",
        "IRR",
        "NPV_CAPEX",
        "LCOE",
        "AEP",
        "PEN_LIFE",
        "BE_PPA",
        "capex_w_a",
        "capex_pv_a",
        "capex_b_a",
        "capex_sh_a",
        "capex_a",
        "opex_w_a",
        "opex_pv_a",
        "opex_b_a",
        "opex_sh_a",
        "opex_a",
    ),
}


SYSTEM_HYDESIGN_NAMES = {
    "EGO": "hydesign.Parallel_EGO.EfficientGlobalOptimizationDriver",
    "HKN": "HKN fixed data and hpp_model constructor arguments",
    "GNN": "design_friendly.utils.easy.easy_yaw_gnn",
    "EVAL": "hpp_pywake_gnn_loads.hpp_model.evaluate input/derived-value mapping",
    "ABL": "abl",
    "PW": "pywake_comp",
    "PV": "pvp",
    "EMS": "ems",
    "PW_ACT": "pywake_actual_comp",
    "B_DEG": "battery_degradation",
    "B_TMP": "battery_loss_in_capacity_due_to_temp",
    "PV_DEG": "pvp_with_degradation",
    "B_REL": "battery_with_reliability",
    "W_REL": "wpp_with_reliability",
    "PV_REL": "pvp_with_reliability",
    "LT_EMS": "ems_long_term_operation",
    "W_COST": "wpp_cost",
    "PV_COST": "pvp_cost",
    "B_COST": "battery_cost",
    "SH_COST": "shared_cost",
    "FIN": "finance",
}


# Exact spelling used in HyDesign (or the originating external API) for every
# abbreviated variable visible in either diagram.  The generated CSV is meant
# to be easy to review and edit when the notation is refined.
VARIABLE_HYDESIGN_NAMES = {
    "AEP": "mean_AEP",
    "ALT": "altitude",
    "AZI": "surface_azimuth",
    "A_PV": "Apvp",
    "A_PV_MW": "land_use_per_solar_MW",
    "A_W": "Awpp",
    "BATCH": "ems_batch_size / batch_size",
    "BE_PPA": "break_even_PPA_price",
    "B_E": "b_E",
    "B_E_H": "b_E_h",
    "B_P": "b_P",
    "CAPEX": "CAPEX",
    "CAPEX_B": "CAPEX_b",
    "CAPEX_EL": "CAPEX_el",
    "CAPEX_PV": "CAPEX_s",
    "CAPEX_REF": "CAPEX_ref",
    "CAPEX_SH": "CAPEX_sh",
    "CAPEX_SH_FIX": "CAPEX_sh_fixed",
    "CAPEX_SH_PV": "CAPEX_sh_s_land",
    "CAPEX_SH_W": "CAPEX_sh_w_land",
    "CAPEX_W": "CAPEX_w",
    "CLR": "clearance",
    "CURT": "total_curtailment",
    "CURT_DEG": "total_curtailment_with_deg",
    "C_BFLUCT": "cost_of_battery_P_fluct_in_peak_price_ratio",
    "C_BOS": "hpp_BOS_soft_cost",
    "C_B_BOP": "battery_BOP_installation_commissioning_cost",
    "C_B_CTRL": "battery_control_system_cost",
    "C_B_E": "battery_energy_cost",
    "C_B_ONM": "battery_energy_onm_cost",
    "C_B_P": "battery_power_cost",
    "C_GRID": "hpp_grid_connection_cost",
    "C_INV": "solar_inverter_cost",
    "C_LAND": "land_cost",
    "C_PV": "solar_PV_cost",
    "C_PV_FIX": "solar_fixed_onm_cost",
    "C_PV_INST": "solar_hardware_installation_cost",
    "C_WT": "wind_turbine_cost",
    "C_W_CIV": "wind_civil_works_cost",
    "C_W_FIX": "wind_fixed_onm_cost",
    "C_W_VAR": "wind_variable_onm_cost",
    "D": "d",
    "DECOM_PV": "decommissioning_cost_tot_s",
    "DECOM_W": "decommissioning_cost_tot_w",
    "DOD": "battery_depth_of_discharge",
    "D_REF": "d_ref",
    "EMS_TYPE": "ems_type",
    "ETA_B": "battery_charge_efficiency",
    "ETA_W": "wpp_efficiency",
    "FARM": "farm",
    "FARM_LOAD": "farm_load",
    "G": "G_MW",
    "H": "hh",
    "H_PK": "n_full_power_hours_expected_per_day_at_peak_price",
    "H_REF": "hh_ref",
    "INTERP_WD": "interpolate_wd",
    "IRR": "IRR",
    "LAT": "latitude",
    "LCOE": "LCOE",
    "LIFE": "life_y",
    "LOAD_CFG": "load_sensors / load_sensors_used / components",
    "LOAD_MIN": "load_min",
    "LON": "longitude",
    "MIN_LOH": "min_LoH",
    "M_ACT": "M_actual",
    "NPV": "NPV",
    "NPV_CAPEX": "NPV_over_CAPEX",
    "N_B": "n_batteries",
    "N_B_MAX": "max_num_batteries_allowed / num_batteries",
    "N_WT": "Nwt",
    "N_Y": "N_time",
    "OPEX": "OPEX",
    "OPEX_B": "OPEX_b",
    "OPEX_EL": "OPEX_el",
    "OPEX_PV": "OPEX_s",
    "OPEX_REF": "_OPEX",
    "OPEX_SH": "OPEX_sh",
    "OPEX_W": "OPEX_w",
    "PEN_LIFE": "penalty_lifetime",
    "P_PV": "solar_MW",
    "P_R": "p_rated",
    "P_R_REF": "p_rated_ref",
    "Q_PK": "peak_hr_quantile",
    "REF_INFL_YR": "ref_yr_inflation",
    "REV": "revenues",
    "RHO_W": "wind_MW_per_km2",
    "R_B_COST": "battery_price_reduction_per_year",
    "R_DC_AC": "DC_AC_ratio",
    "SP": "sp",
    "TAX": "tax_rate",
    "TI": "TI",
    "TILT": "surface_tilt",
    "TRACKING": "tracking",
    "V_CO2": "emissions_value",
    "WACC_B": "battery_WACC",
    "WACC_PV": "solar_WACC",
    "WACC_W": "wind_WACC",
    "WEATHER_FN": "input_ts_fn / weather_fn",
    "WEEKS": "weeks_per_season_per_year",
    "WT_GNN": "design_friendly wind-turbine model (wt)",
    "YAW_MODE": "yaw_mode",
    "b_deg_l": "b_t_with_deg",
    "b_l": "b_t",
    "b_rel_l": "b_t_rel",
    "capex_a": "CAPEX_vec",
    "capex_b_a": "CAPEX_b_vec",
    "capex_pv_a": "CAPEX_s_vec",
    "capex_sh_a": "CAPEX_sh_vec",
    "capex_w_a": "CAPEX_w_vec",
    "del_act_s_i_y": "DEL_actual",
    "del_ref_s_i_y": "DEL_ref",
    "del_s_i_y": "DEL",
    "dep_a": "depreciation",
    "dep_yr_a": "depreciation_yr",
    "infl_a": "inflation",
    "infl_yr_a": "inflation_yr",
    "lambda_y": "lambda_t",
    "ldel_ref_s_i": "LDEL_ref",
    "ldel_s_i": "LDEL",
    "m_exp_s": "wholer_exponents",
    "m_y": "M",
    "mdf_y": "marginal_displacement_factor_t",
    "opex_a": "OPEX_vec",
    "opex_b_a": "OPEX_b_vec",
    "opex_pv_a": "OPEX_s_vec",
    "opex_sh_a": "OPEX_sh_vec",
    "opex_w_a": "OPEX_w_vec",
    "p_curt_deg_l": "hpp_curt_t_with_deg",
    "p_curt_l": "hpp_curt_t",
    "p_hpp_deg_l": "hpp_t_with_deg",
    "p_hpp_l": "hpp_t",
    "p_pv_deg_l": "solar_t_ext_deg",
    "p_pv_l": "solar_t_ext",
    "p_pv_rel_l": "solar_t_rel",
    "p_pv_y": "solar_t",
    "p_w_act_y": "wind_t_actual",
    "p_w_l": "wind_t_ext",
    "p_w_ref_y": "wind_t_ref",
    "p_w_rel_l": "wind_t_rel",
    "p_w_y": "wind_t",
    "pen_deg_l": "penalty_t_with_deg",
    "pen_l": "penalty_t",
    "phase_capex_a": "phasing_CAPEX",
    "phase_yr_a": "phasing_yr",
    "price_l": "price_t_ext",
    "price_y": "price_t",
    "pv_deg_a": "pv_deg",
    "pv_deg_yr_a": "pv_deg_yr",
    "rel_b_l": "reliability_ts_battery",
    "rel_pv_l": "reliability_ts_pv",
    "rel_tr_l": "reliability_ts_trans",
    "rel_w_l": "reliability_ts_wind",
    "soc_deg_l": "b_E_SOC_t_with_deg",
    "soc_l": "b_E_SOC_t",
    "soh_all_l": "SoH_all",
    "soh_l": "SoH",
    "t_y": "time_stamp",
    "tilt_y": "tilt",
    "wd_g": "wd (Design-Friendly lookup grid)",
    "wd_y": "wd",
    "ws_g": "ws (Design-Friendly lookup grid)",
    "ws_y": "wst",
    "x_i": "x",
    "y_i": "y",
    "yaw_act_i_y": "yaw_actual",
    "yaw_i_g": "yaw",
    "yaw_i_y": "yaw_t",
}


def _shape(name: str) -> tuple[int, ...]:
    """Small representative shapes; only dimensionality matters to the viewer."""
    special = {
        "yaw_i_g": (2, 3, 4),
        "yaw_i_y": (2, 24),
        "yaw_act_i_y": (2, 24),
        "del_s_i_y": (4, 2, 24),
        "del_ref_s_i_y": (4, 2, 24),
        "del_act_s_i_y": (4, 2, 24),
        "ldel_s_i": (4, 2),
        "ldel_ref_s_i": (4, 2),
        "m_exp_s": (4,),
        "x_i": (2,),
        "y_i": (2,),
        "ws_g": (4,),
        "wd_g": (3,),
    }
    if name in special:
        return special[name]
    if name.endswith("_y"):
        return (24,)
    if name.endswith("_l"):
        return (48,)
    if name.endswith("_a"):
        return (3,)
    return ()


class _ViewerComponent(om.ExplicitComponent):
    """Interface-only component used solely by OpenMDAO's N2 writer."""

    def __init__(self, inputs: tuple[str, ...], outputs: tuple[str, ...]):
        self._viewer_inputs = inputs
        self._viewer_outputs = outputs
        super().__init__()

    def setup(self):
        for name in self._viewer_inputs:
            self.add_input(name, val=np.zeros(_shape(name)) if _shape(name) else 0.0)
        for name in self._viewer_outputs:
            self.add_output(name, val=np.zeros(_shape(name)) if _shape(name) else 0.0)

    def compute(self, inputs, outputs):
        # The diagram problem is never evaluated.
        pass


def _source_target(variable: str | tuple[str, str]) -> tuple[str, str]:
    if isinstance(variable, tuple):
        return variable
    return variable, variable


def _all_variable_aliases() -> set[str]:
    aliases = set()
    for connection in CONNECTIONS:
        for variable in connection.variables:
            aliases.update(_source_target(variable))
    for variables in TERMINAL_OUTPUTS.values():
        aliases.update(variables)
    return aliases


def write_name_map(output_file: Path) -> None:
    """Write the complete editable HyDesign-to-diagram naming table."""
    aliases = _all_variable_aliases()
    missing = aliases - VARIABLE_HYDESIGN_NAMES.keys()
    extra = VARIABLE_HYDESIGN_NAMES.keys() - aliases
    if missing or extra:
        raise ValueError(
            f"Variable-name map mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    with output_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("kind", "abbreviated_name", "hydesign_name"))
        for system in SYSTEMS:
            writer.writerow(("system", system.key, SYSTEM_HYDESIGN_NAMES[system.key]))
        for alias in sorted(aliases, key=str.casefold):
            writer.writerow(("variable", alias, VARIABLE_HYDESIGN_NAMES[alias]))


def _interfaces():
    inputs = {system.key: OrderedDict() for system in SYSTEMS}
    outputs = {system.key: OrderedDict() for system in SYSTEMS}
    for connection in CONNECTIONS:
        for variable in connection.variables:
            source_variable, target_variable = _source_target(variable)
            outputs[connection.source][source_variable] = None
            inputs[connection.target][target_variable] = None
    for system, variables in TERMINAL_OUTPUTS.items():
        for variable in variables:
            outputs[system][variable] = None
    return inputs, outputs


def build_n2_problem() -> om.Problem:
    """Build the interface-only OpenMDAO problem used by the N2 viewer."""
    inputs, outputs = _interfaces()
    problem = om.Problem(reports=None)
    for system in SYSTEMS:
        problem.model.add_subsystem(
            system.key,
            _ViewerComponent(tuple(inputs[system.key]), tuple(outputs[system.key])),
        )
    for connection in CONNECTIONS:
        for variable in connection.variables:
            source_variable, target_variable = _source_target(variable)
            problem.model.connect(
                f"{connection.source}.{source_variable}",
                f"{connection.target}.{target_variable}",
            )

    # Record the intended optimization semantics in the N2 driver metadata.
    problem.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", disp=False)
    for variable in (
        "CLR",
        "SP",
        "P_R",
        "N_WT",
        "RHO_W",
        "P_PV",
        "TILT",
        "AZI",
        "R_DC_AC",
        "B_P",
        "B_E_H",
        "C_BFLUCT",
    ):
        problem.model.add_design_var(f"EGO.{variable}")
    problem.model.add_objective("FIN.NPV", scaler=-1.0)
    problem.setup()
    return problem


def write_n2(output_file: Path) -> None:
    """Write a standalone interactive OpenMDAO N2 HTML file."""
    problem = build_n2_problem()
    om.n2(problem, outfile=str(output_file), show_browser=False)
    # OpenMDAO bundles upstream JavaScript/CSS with trailing spaces.  Normalize
    # the generated artifact so repository whitespace checks stay useful.
    html = output_file.read_text(encoding="utf-8")
    output_file.write_text(
        "\n".join(line.rstrip() for line in html.splitlines()) + "\n",
        encoding="utf-8",
    )


def _tex_var(name: str) -> str:
    """Convert compact ASCII N2 names to readable PyXDSM math."""
    replacements = {
        "lambda_y": r"\lambda_y",
        "yaw_i_g": r"\gamma^{opt}_{i,g}",
        "yaw_i_y": r"\gamma^{opt}_{i,y}",
        "yaw_act_i_y": r"\gamma^{act}_{i,y}",
        "ETA_W": r"\eta_W",
        "ETA_B": r"\eta_B",
        "V_CO2": r"V_{CO_2}",
        "M_ACT": r"M^{act}",
        "M_EXP_s": r"m_s",
    }
    if name in replacements:
        return replacements[name]
    parts = name.split("_")
    if len(parts) == 1:
        return name
    return rf"{parts[0]}_{{{' '.join(parts[1:])}}}"


def _edge_label(variables: tuple[str | tuple[str, str], ...], tex: bool = False) -> str:
    names = tuple(
        variable if isinstance(variable, str) else f"{variable[0]}>{variable[1]}"
        for variable in variables
    )
    if tex:
        return ", ".join(_tex_var(variable) for variable in names)
    return "\n".join(
        ", ".join(names[start : start + 3]) for start in range(0, len(names), 3)
    )


def build_pyxdsm() -> XDSM:
    """Build a PyXDSM definition from the same manifest as the N2."""
    diagram = XDSM(use_sfmath=True)
    for system in SYSTEMS:
        diagram.add_system(system.key, system.kind, rf"\text{{{system.label}}}")
    for connection in CONNECTIONS:
        diagram.connect(
            connection.source,
            connection.target,
            rf"\mathrm{{{_edge_label(connection.variables, tex=True)}}}",
        )
    for system, variables in TERMINAL_OUTPUTS.items():
        diagram.add_output(
            system,
            rf"\mathrm{{{_edge_label(variables, tex=True)}}}",
            side="right",
        )
    return diagram


def _trapezoid(ax, center, text, width=1.02, height=0.30, facecolor="#F2F2F2"):
    x, y = center
    line_count = text.count("\n") + 1
    height = max(height, 0.12 + 0.075 * line_count)
    inset = width * 0.10
    points = [
        (x - width / 2 + inset, y - height / 2),
        (x + width / 2, y - height / 2),
        (x + width / 2 - inset, y + height / 2),
        (x - width / 2, y + height / 2),
    ]
    ax.add_patch(
        Polygon(
            points,
            closed=True,
            facecolor=facecolor,
            edgecolor="black",
            linewidth=0.55,
            zorder=3,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=3.9, zorder=4)


def render_xdsm(output_base: Path) -> None:
    """Render a wide documentation-friendly XDSM SVG and PNG."""
    index = {system.key: i for i, system in enumerate(SYSTEMS)}
    count = len(SYSTEMS)
    x_scale = 1.22
    y_scale = 0.48
    fig, ax = plt.subplots(figsize=(30, 10), constrained_layout=True)
    ax.set_facecolor("white")

    def point(key):
        i = index[key]
        return i * x_scale, (count - 1 - i) * y_scale

    for connection in CONNECTIONS:
        sx, sy = point(connection.source)
        tx, ty = point(connection.target)
        ax.plot([sx, tx, tx], [sy, sy, ty], color="#D0D0D0", linewidth=2.6, zorder=1)

    colors = {FUNC: "#A3DA96", METAMODEL: "#F4D982", OPT: "#B3D5ED"}
    for system in SYSTEMS:
        x, y = point(system.key)
        color = "#EEEEEE" if system.key == "HKN" else colors[system.kind]
        ax.add_patch(
            Rectangle(
                (x - 0.48, y - 0.18),
                0.96,
                0.36,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )
        )
        ax.text(
            x,
            y,
            system.label,
            ha="center",
            va="center",
            fontsize=6.4,
            fontweight="semibold",
            zorder=6,
        )

    for connection in CONNECTIONS:
        _, sy = point(connection.source)
        tx, _ = point(connection.target)
        facecolor = "#F7E3E3" if connection.source == "FIN" else "#F2F2F2"
        _trapezoid(ax, (tx, sy), _edge_label(connection.variables), facecolor=facecolor)

    output_x = count * x_scale + 0.45
    for source, variables in TERMINAL_OUTPUTS.items():
        sx, sy = point(source)
        ax.plot([sx, output_x], [sy, sy], color="#D0D0D0", linewidth=2.6, zorder=1)
        _trapezoid(
            ax, (output_x, sy), _edge_label(variables), width=1.18, facecolor="white"
        )

    legend_y = -0.68
    legend = (
        ("Function", colors[FUNC]),
        ("Metamodel", colors[METAMODEL]),
        ("Optimization", colors[OPT]),
        ("Fixed/constructor data", "#EEEEEE"),
        ("NPV objective", "#F7E3E3"),
    )
    for i, (label, color) in enumerate(legend):
        x = i * 2.6
        ax.add_patch(
            Rectangle(
                (x - 0.24, legend_y - 0.13),
                0.40,
                0.26,
                facecolor=color,
                edgecolor="black",
                linewidth=0.6,
            )
        )
        ax.text(x + 0.25, legend_y, label, ha="left", va="center", fontsize=6.8)

    ax.text(
        (count - 1) * x_scale / 2,
        count * y_scale + 0.35,
        "HKN coupled HPP-WF assembly",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        (count - 1) * x_scale / 2,
        count * y_scale + 0.04,
        "Subscripts: y = representative-year hourly vector; l = lifetime hourly vector",
        ha="center",
        fontsize=8,
    )
    ax.set_xlim(-0.75, output_x + 0.85)
    ax.set_ylim(-0.95, count * y_scale + 0.63)
    ax.set_aspect("auto")
    ax.axis("off")

    png_path = output_base.with_suffix(".png")
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tex", action="store_true", help="also write PyXDSM .tex/.tikz"
    )
    parser.add_argument("--pdf", action="store_true", help="compile the PyXDSM PDF")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    output_dir = repo_root / "docs" / "_static"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = "xdsm_coupled_hpp_gnn_loads"
    render_xdsm(output_dir / output_name)
    write_n2(output_dir / "n2_coupled_hpp_gnn_loads.html")
    write_name_map(
        repo_root
        / "hydesign"
        / "examples"
        / "SUDOCO"
        / "coupled_hpp_diagram_name_map.csv"
    )

    if args.tex or args.pdf:
        build_pyxdsm().write(
            output_name,
            build=args.pdf,
            cleanup=True,
            quiet=True,
            outdir=str(output_dir),
        )


if __name__ == "__main__":
    main()
