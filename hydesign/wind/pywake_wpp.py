import os

import numpy as np
import tqdm
from costmodels.models.variable_opex import lifetime_aware_model
from surrogates_interface.surrogates import TensorFlowModel

try:
    from design_friendly.models import models_filepath
    from design_friendly.utils.easy import easy_yaw_gnn
except ImportError as error:
    models_filepath = None
    easy_yaw_gnn = None
    _design_friendly_import_error = error
else:
    _design_friendly_import_error = None

try:
    from wind_farm_loads.py_wake import predict_loads_rotor_average
except ImportError:
    try:
        from wind_farm_loads.tool_agnostic import predict_loads_rotor_average
    except ImportError as error:
        predict_loads_rotor_average = None
        _wind_farm_loads_import_error = error
    else:
        _wind_farm_loads_import_error = None
else:
    _wind_farm_loads_import_error = None

LOAD_PATHS = (
    os.path.join(models_filepath, "RA_IEA22_DTU")
    if models_filepath is not None
    else None
)


def _require_optional_dependencies():
    if _design_friendly_import_error is not None:
        raise ImportError(
            "PyWakeWPP requires the optional 'gnn-loads' dependencies. "
            "Install hydesign[gnn-loads] before using this component."
        ) from _design_friendly_import_error

    if _wind_farm_loads_import_error is not None:
        raise ImportError(
            "PyWakeWPP requires wind_farm_loads from the optional "
            "'gnn-loads' dependency set."
        ) from _wind_farm_loads_import_error


class PyWakeWPP:
    def __init__(
        self,
        n_wt,
        x,
        y,
        tilt,
        time_stamp,
        sim_pars,
        load_sensors,
        load_sensors_used,
        components,
        _LIFETIME,
        wpp_efficiency,
        farm,
        N_time,
        G_MW,
        _OPEX=1.42,
        CAPEX_ref=None,
        no_yaw=False,
    ):
        _require_optional_dependencies()
        if CAPEX_ref is None:
            CAPEX_ref = 4 * G_MW
        self.n_wt = n_wt
        self.x = x
        self.y = y
        self.tilt = tilt
        self.time_stamp = time_stamp
        self.sim_pars = sim_pars
        self.load_sensors = load_sensors
        self.load_sensors_used = load_sensors_used
        self.components = components
        self._LIFETIME = _LIFETIME
        self._OPEX = _OPEX
        self.CAPEX_ref = CAPEX_ref
        self.wpp_efficiency = wpp_efficiency
        self.farm = farm
        self.N_time = N_time
        self.G_MW = G_MW
        self.no_yaw = no_yaw

        surrogates = {}
        for ch in self.load_sensors_used:
            model_path = os.path.join(LOAD_PATHS, f"{ch}.keras")
            scaler_path = os.path.join(LOAD_PATHS, f"scaler_{ch}.h5")
            surrogates[ch] = TensorFlowModel.load_h5(
                model_path=model_path, extra_data_path=scaler_path
            )
        # wholer exp from surrogate metadata
        wholer_exponents = [
            surrogates[c].metadata["Wohler_exponent"].item() for c in load_sensors_used
        ]
        self.wholer_exponents = np.array(wholer_exponents)
        self.surrogates = surrogates

    def compute(self, wst, wd, **kwargs):  # yaw,
        n_wt = self.n_wt
        x = self.x
        y = self.y
        tilt = self.tilt
        time_stamp = self.time_stamp
        sim_pars = self.sim_pars
        components = self.components
        _LIFETIME = self._LIFETIME
        _OPEX = self._OPEX
        CAPEX_ref = self.CAPEX_ref
        wpp_efficiency = self.wpp_efficiency
        no_yaw = self.no_yaw
        surrogates = self.surrogates
        wholer_exponents = self.wholer_exponents
        load_sensors_used = self.load_sensors_used
        # load_sensors = self.load_sensors
        ws = wst
        n_t = len(ws)
        # https://docs.nrel.gov/docs/fy25osti/91775.pdf CAPEX = 5.411 dollar/MW = 4.65 euro/MW
        # https://guidetoanoffshorewindfarm.com/wind-farm-costs/ CAPEX = 3.47 pound/MW = 3.96 euro/MW
        sim_res_ref = self.farm(
            x=x,
            y=y,
            yaw=np.zeros((n_wt, n_t)),
            tilt=tilt,
            wd=wd,
            ws=ws,
            time=time_stamp,
            TI=sim_pars["TI"],
        )
        wind_t_ref = sim_res_ref.Power.sum("wt").to_numpy() * wpp_efficiency * 1e-6
        yaw_ref = np.zeros((n_wt, n_t))
        loads_rotor_ref = predict_loads_rotor_average(
            surrogates,
            sim_res_ref,  # ws (m/s) ti (%)
            yaw_ref,  # yaw (deg)
            100,  # power_demand (%)
            ti_in_percent=True,
            dtype=np.float32,
        ).to_numpy()  # (wt, time, channel)  # 3600
        loads_rotor_ref = loads_rotor_ref.transpose((2, 0, 1))  # (chanel, wt, time)
        loads_rotor_ref_used = np.asarray(
            [
                loads_rotor_ref[list(surrogates.keys()).index(s), :, :]
                for s in load_sensors_used
            ]
        )  # (wt, time, channel)

        if not no_yaw:
            yaw_t = easy_yaw_gnn(x, y, wd=wd, ws=ws, TI=sim_pars["TI"], time=True)
            sim_res = self.farm(
                x=x,
                y=y,
                yaw=yaw_t,
                tilt=tilt,
                wd=wd,
                ws=ws,
                time=time_stamp,
                TI=sim_pars["TI"],
            )
            wind_t = (
                sim_res.Power.sum("wt").to_numpy() * wpp_efficiency * 1e-6
            )  # W -> MW
            loads_rotor = predict_loads_rotor_average(
                surrogates,
                sim_res,  # ws (m/s) ti (%)
                sim_res.yaw.values,  # yaw (deg)
                100,  # power_demand (%)
                ti_in_percent=True,
                dtype=np.float32,
            ).to_numpy()  # 3600  # (wt, time, channel)
            loads_rotor = loads_rotor.transpose((2, 0, 1))  # (chanel, wt, time)
        else:
            yaw_t = yaw_ref
            loads_rotor = loads_rotor_ref  # (wt, time, channel)
            sim_res = sim_res_ref
            wind_t = wind_t_ref
        loads_rotor_used = np.asarray(
            [
                loads_rotor[list(surrogates.keys()).index(s), :, :]
                for s in load_sensors_used
            ]
        )

        DEL = loads_rotor_used  # .to_numpy()  # (wt, time, channel)
        DEL_ref = loads_rotor_ref_used  # .to_numpy()  # (wt, time, channel)

        LDELs = (
            DEL
            * (3600 * 24 * 365 * 20 / 10**7)
            ** wholer_exponents[:, np.newaxis, np.newaxis]
        )
        LDEL_refs = (
            DEL_ref
            * (3600 * 24 * 365 * 20 / 10**7)
            ** wholer_exponents[:, np.newaxis, np.newaxis]
        )

        M = np.zeros((self.N_time, n_wt))
        print("Computing M matrix for variable OPEX calculation...")
        gamma = (LDELs / LDEL_refs) ** wholer_exponents[:, np.newaxis, np.newaxis]

        gamma_ref = np.ones_like(gamma)
        for turbine in tqdm.tqdm(range(n_wt)):
            _gamma_controls = gamma[:, turbine, :]  # (n_sensors, N_time)
            _gamma_refs = gamma_ref[:, turbine, :]
            _CAPEX = CAPEX_ref / n_wt  # MEUR
            _, _, _, Total_ONM_cost_ref, Total_ONM_cost_control = lifetime_aware_model(
                _gamma_refs,
                _gamma_controls,
                components,
                _LIFETIME,
                _OPEX,
                _CAPEX,
            )
            # print('Total_ONM_cost_ref:', Total_ONM_cost_ref)
            M[:, turbine] = (
                (Total_ONM_cost_control - Total_ONM_cost_ref) * _CAPEX / (365 * 24)
            )
        M = (
            M.sum((1)) * 1e6
        )  # hourly ONM costs for all turbines for control based on ref (positive if control is more expensive) in EURO
        print("Done computing M matrix for variable OPEX calculation...")

        return [wind_t, wind_t_ref, yaw_t, M, DEL, DEL_ref]  # wholer_exponents,

    def compute_actual(
        self,
        wind_t,
        wind_t_ref,
        yaw_t,
        lambda_t,
        # wholer_exponents,
        DEL,
        DEL_ref,
        **kwargs,
    ):
        n_wt = self.n_wt
        CAPEX_ref = self.CAPEX_ref
        components = self.components
        wholer_exponents = self.wholer_exponents
        _LIFETIME = self._LIFETIME
        _OPEX = self._OPEX

        yaw_actual = yaw_t * lambda_t[np.newaxis, :]
        wind_t_actual = wind_t * lambda_t + wind_t_ref * (1 - lambda_t)
        DEL_actual = (
            DEL * lambda_t[np.newaxis, np.newaxis, :]
            + DEL_ref * (1 - lambda_t)[np.newaxis, np.newaxis, :]
        )
        LDEL = (
            (
                (DEL_actual / DEL_actual.mean(axis=2, keepdims=True))
                ** wholer_exponents[:, np.newaxis, np.newaxis]
            ).sum((2))
            * 3600
            * 24
            * 365
            * 20
            / 10**7
        ) ** (1 / wholer_exponents[:, np.newaxis]) * DEL_actual.mean(axis=2)
        LDEL_ref = (
            (
                (DEL_ref / DEL_ref.mean(axis=2, keepdims=True))
                ** wholer_exponents[:, np.newaxis, np.newaxis]
            ).sum((2))
            * 3600
            * 24
            * 365
            * 20
            / 10**7
        ) ** (1 / wholer_exponents[:, np.newaxis]) * DEL_ref.mean(axis=2)
        gamma = (LDEL / LDEL_ref) ** wholer_exponents[:, np.newaxis]
        gamma_ref = np.ones_like(gamma)
        M = np.zeros((n_wt))
        for turbine in tqdm.tqdm(range(n_wt)):
            _gamma_controls = gamma[:, turbine]  # (n_sensors)
            _gamma_refs = gamma_ref[:, turbine]
            _CAPEX = CAPEX_ref / n_wt  # MEUR
            _, Total_ONM_cost_ref, Total_ONM_cost_control, _, _ = lifetime_aware_model(
                _gamma_refs,
                _gamma_controls,
                components,
                _LIFETIME,
                _OPEX,
                _CAPEX,
            )
            M[turbine] = (Total_ONM_cost_control - Total_ONM_cost_ref) * _CAPEX
        M_actual = (
            M.sum() * 1e6
        )  # yearly ONM costs for all turbines for control based on ref (positive if control is more expensive) in EURO
        return [wind_t_actual, yaw_actual, DEL_actual, LDEL, LDEL_ref, M_actual]
