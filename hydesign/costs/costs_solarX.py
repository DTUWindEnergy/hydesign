import scipy as sp

from hydesign.openmdao_wrapper import ComponentWrapper


class sf_cost:
    """
    Solar Field (SF) Cost Model - Calculates the capital and operational expenses
    for a solar field based on the input parameters.
    """

    def __init__(self, heliostat_cost_per_m2, sf_opex_cost_per_m2):
        # super().__init__()
        # Set the cost parameters
        self.heliostat_cost_per_m2 = heliostat_cost_per_m2
        self.sf_opex_cost_per_m2 = sf_opex_cost_per_m2

        # def setup(self):
        # Define inputs and outputs for the component
        # inputs
        self.inputs = [
            (
                "sf_area",
                dict(desc="Installed capacity of the solar field", units="m**2"),
            )
        ]
        # outputs
        self.outputs = [
            ("CAPEX_sf", dict(desc="CAPEX of solar field (mirrors and controllers)")),
            ("OPEX_sf", dict(desc="OPEX of solar field (mirrors and controllers)")),
        ]

    def compute(self, **inputs):
        outputs = {}
        # Calculate capital and operational costs based on area of solar field
        sf_area = inputs["sf_area"]
        capex_sf = self.heliostat_cost_per_m2 * sf_area
        opex_sf = self.sf_opex_cost_per_m2 * sf_area
        outputs["CAPEX_sf"] = capex_sf
        outputs["OPEX_sf"] = opex_sf
        out_keys = ["CAPEX_sf", "OPEX_sf"]
        return [outputs[key] for key in out_keys]


class sf_cost_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = sf_cost(**insta_inp)
        super().__init__(
            inputs=model.inputs,
            outputs=model.outputs,
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


class cpv_cost:
    """
    Concentrated Photovoltaic (cpv) Cost Model - Calculates CAPEX and OPEX
    for cpv systems based on installation and equipment costs.
    """

    def __init__(
        self, cpv_cost_per_m2, inverter_cost_per_MW_DC, cpv_fixed_opex_cost_per_m2
    ):
        # super().__init__()
        # Set cost parameters for cpv system
        self.cpv_cost_per_m2 = cpv_cost_per_m2
        self.inverter_cost_per_MW_DC = inverter_cost_per_MW_DC
        self.cpv_fixed_opex_cost_per_m2 = cpv_fixed_opex_cost_per_m2

        # def setup(self):
        # Define inputs and outputs
        # inputs
        self.inputs = [
            (
                "cpv_inverter_mw",
                dict(desc="rated power of the cpv inverter", units="MW"),
            ),
            ("area_cpv_receiver_m2", dict(desc="Area of cpv receivers", units="m**2")),
        ]
        # outputs
        self.outputs = [
            ("CAPEX_cpv", dict(desc="CAPEX of cpv system")),
            ("OPEX_cpv", dict(desc="OPEX of cpv system")),
        ]

    def compute(self, **inputs):
        outputs = {}
        # Compute CAPEX and OPEX for cpv system
        cpv_inverter_mw = inputs["cpv_inverter_mw"]
        cpv_m2 = inputs["area_cpv_receiver_m2"]

        capex_cpv = self.cpv_cost_per_m2 * cpv_m2 + (
            self.inverter_cost_per_MW_DC * cpv_inverter_mw
        )
        opex_cpv = self.cpv_fixed_opex_cost_per_m2 * cpv_m2
        outputs["CAPEX_cpv"] = capex_cpv
        outputs["OPEX_cpv"] = opex_cpv
        out_keys = ["CAPEX_cpv", "OPEX_cpv"]
        return [outputs[key] for key in out_keys]


class cpv_cost_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = cpv_cost(**insta_inp)
        super().__init__(
            inputs=model.inputs,
            outputs=model.outputs,
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


class cst_cost:
    """
    Concentrated Solar Thermal (cst) Cost Model - Calculates the capital and
    operational expenses for cst systems based on collector, molten salt tank,
    and turbine-related costs.
    """

    def __init__(
        self,
        cst_th_collector_cost_per_m2,
        ms_installation_cost_per_m3,
        steam_turbine_cost_per_MW,
        heat_exchnager_cost_per_MW,  # MWt, kg/h steam
        fixed_opex_per_MW,
    ):
        # super().__init__()
        # Set cost parameters for cst system components
        self.cst_th_collector_cost_per_m2 = cst_th_collector_cost_per_m2
        self.ms_installation_cost_per_m3 = ms_installation_cost_per_m3
        self.steam_turbine_cost_per_MW = steam_turbine_cost_per_MW
        self.heat_exchnager_cost_per_MW = heat_exchnager_cost_per_MW
        self.fixed_opex_per_MW = fixed_opex_per_MW

        # def setup(self):
        # Define inputs and outputs for cst component
        # inputs
        self.inputs = [
            (
                "area_cst_receiver_m2",
                dict(desc="Area of heat receiver on the tower", units="m**2"),
            ),
            (
                "v_molten_salt_tank_m3",
                dict(desc="Volume of the molten salt storage", units="m**3"),
            ),
            ("p_rated_st", dict(desc="Steam turbine power capacity", units="MW")),
            (
                "heat_exchanger_capacity",
                dict(desc="Heat exchnager power capacity", units="MW"),
            ),
        ]

        # outputs
        self.outputs = [
            ("CAPEX_cst", dict(desc="CAPEX of cst system")),
            ("OPEX_cst", dict(desc="OPEX of cst system")),
        ]

    def compute(self, **inputs):
        outputs = {}
        # Calculate CAPEX and OPEX for cst system
        area_cst_receiver_m2 = inputs["area_cst_receiver_m2"]
        v_molten_salt_tank_m3 = inputs["v_molten_salt_tank_m3"]
        p_rated_st = inputs["p_rated_st"]
        heat_exchanger_capacity = inputs["heat_exchanger_capacity"]

        # CAPEX and OPEX calculations
        capex_receiver = self.cst_th_collector_cost_per_m2 * area_cst_receiver_m2
        capex_molten_salt = self.ms_installation_cost_per_m3 * v_molten_salt_tank_m3
        capex_heat_exchanger = heat_exchanger_capacity * self.heat_exchnager_cost_per_MW
        capex_turbine = p_rated_st * self.steam_turbine_cost_per_MW
        capex_cst = (
            capex_receiver + capex_molten_salt + capex_heat_exchanger + capex_turbine
        )

        opex_cst = self.fixed_opex_per_MW * p_rated_st

        # outputs
        outputs["CAPEX_cst"] = capex_cst
        outputs["OPEX_cst"] = opex_cst
        out_keys = ["CAPEX_cst", "OPEX_cst"]
        return [outputs[key] for key in out_keys]


class cst_cost_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = cst_cost(**insta_inp)
        super().__init__(
            inputs=model.inputs,
            outputs=model.outputs,
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


class H2Cost:
    """
    Hydrogen Production Cost Model - Calculates the capital and operational expenses
    for H2 production based on reactor, installation, and maintenance costs.
    """

    def __init__(
        self,
        reactor_cost_per_m2,  # Waiting for Luc input
        maximum_h2_production_reactor_kg_per_m2,
        el_heater_cost_kg_per_h,  # Waiting for Luc input
        pipe_pump_valves_cost_kg_per_h,  # Waiting for Luc input
        psa_cost_kg_per_h,
        carbon_capture_cost_kg_per_h,
        dni_installation_cost_kg_per_h,
        el_installation_cost_kg_per_h,
        maintenance_cost_kg_per_h,
        life_h,
        carbon_capture=False,
    ):

        # super().__init__()
        self.life_h = int(life_h)
        self.carbon_capture = carbon_capture
        self.maximum_h2_production_reactor_kg_per_m2 = (
            maximum_h2_production_reactor_kg_per_m2
        )
        # capex
        self.reactor_cost_per_m2 = reactor_cost_per_m2
        self.el_heater_cost_kg_per_h = (
            el_heater_cost_kg_per_h  # only for electrical reactors
        )
        self.pipe_pump_valves_cost_kg_per_h = pipe_pump_valves_cost_kg_per_h
        self.psa_cost_kg_per_h = psa_cost_kg_per_h
        self.carbon_capture_cost_kg_per_h = carbon_capture_cost_kg_per_h

        # installation
        self.dni_installation_cost_kg_per_h = dni_installation_cost_kg_per_h
        self.el_installation_cost_kg_per_h = el_installation_cost_kg_per_h

        # opex
        self.maintenance_cost_kg_per_h = maintenance_cost_kg_per_h

        # def setup(self):
        # Define inputs for the openmdao model
        # inputs
        self.inputs = [
            (
                "area_el_reactor_biogas_h2",
                dict(desc="Area of the biogas_h2 electrical reactor", units="m**2"),
            ),
            (
                "area_dni_reactor_biogas_h2",
                dict(desc="Area of the biogas_h2 dni reactor", units="m**2"),
            ),
            (
                "biogas_t_ext",
                dict(
                    desc="Biogas consumption time series",
                    units="kg/h",
                    shape=[self.life_h],
                ),
            ),
            (
                "water_t_ext",
                dict(
                    desc="Water consumption time series",
                    units="kg/h",
                    shape=[self.life_h],
                ),
            ),
            (
                "co2_t_ext",
                dict(
                    desc="CO2 consumption time series",
                    units="kg/h",
                    shape=[self.life_h],
                ),
            ),
            (
                "p_biogas_h2_t",
                dict(
                    desc="electricity consumption time series",
                    shape=[self.life_h],
                    units="MW",
                ),
            ),
            (
                "price_el_t_ext",
                dict(desc="electricity price time series", shape=[self.life_h]),
            ),
            (
                "price_biogas_t_ext",
                dict(desc="electricity price time series", shape=[self.life_h]),
            ),
            (
                "price_water_t_ext",
                dict(desc="electricity price time series", shape=[self.life_h]),
            ),
            (
                "price_co2_t_ext",
                dict(desc="electricity price time series", shape=[self.life_h]),
            ),
        ]

        # outputs
        self.outputs = [
            ("CAPEX_h2", dict(desc="CAPEX of H2 Production")),
            ("OPEX_h2", dict(desc="OPEX of H2 Production")),
            ("OPEX_el", dict(desc="OPEX costs for electricity")),
        ]

    def compute(self, **inputs):
        # Calculate CAPEX based on capacity and component costs
        # load data
        outputs = {}
        area_el_reactor_biogas_h2 = inputs["area_el_reactor_biogas_h2"]
        area_dni_reactor_biogas_h2 = inputs["area_dni_reactor_biogas_h2"]
        water_t_ext = inputs["water_t_ext"]
        biogas_t_ext = inputs["biogas_t_ext"]
        co2_t_ext = inputs["co2_t_ext"]
        p_biogas_h2_t = inputs["p_biogas_h2_t"]
        price_el_t = inputs["price_el_t_ext"]
        price_biogas_t = inputs["price_biogas_t_ext"]
        price_water_t = inputs["price_water_t_ext"]
        price_co2_t = inputs["price_co2_t_ext"]
        carbon_capture = (
            self.carbon_capture
        )  # indicator of the presence of carbon capture

        el_h2_kg_h = (
            area_el_reactor_biogas_h2 * self.maximum_h2_production_reactor_kg_per_m2
        )
        dni_h2_kg_h = (
            area_dni_reactor_biogas_h2 * self.maximum_h2_production_reactor_kg_per_m2
        )

        # Total area and capacity for H2 reactors
        overall_h2_receiver_area = (
            area_el_reactor_biogas_h2 + area_dni_reactor_biogas_h2
        )
        overall_h2_kg_per_h = el_h2_kg_h + dni_h2_kg_h

        # Reactor and component costs
        reactor_cost = self.reactor_cost_per_m2 * overall_h2_receiver_area
        pipe_pump_valves_cost = (
            self.pipe_pump_valves_cost_kg_per_h * overall_h2_kg_per_h
        )
        psa_cost = self.psa_cost_kg_per_h * overall_h2_kg_per_h

        # cost for capturing the carbon
        if carbon_capture:
            carbon_capture_cost = (
                self.carbon_capture_cost_kg_per_h * overall_h2_kg_per_h
            )
        else:
            carbon_capture_cost = 0

        # Electrical heater and installation costs
        el_heater_cost = el_h2_kg_h * self.el_heater_cost_kg_per_h
        installation_cost = (
            el_h2_kg_h * self.el_installation_cost_kg_per_h
            + dni_h2_kg_h * self.dni_installation_cost_kg_per_h
        )

        # Total CAPEX
        capex = (
            reactor_cost
            + pipe_pump_valves_cost
            + psa_cost
            + carbon_capture_cost
            + el_heater_cost
            + installation_cost
        )
        outputs["CAPEX_h2"] = capex

        # OPEX calculations for maintenance, consumed water, biogas, CO2
        maintenance_cost = overall_h2_kg_per_h * self.maintenance_cost_kg_per_h
        water_cost = sum(water_t_ext * price_water_t)
        biogas_cost = sum(biogas_t_ext * price_biogas_t)
        co2_cost = sum(co2_t_ext * price_co2_t)
        outputs["OPEX_h2"] = water_cost + co2_cost + biogas_cost + maintenance_cost

        # OPEX calculations for consumed electricity
        outputs["OPEX_el"] = sum(p_biogas_h2_t * price_el_t)
        out_keys = ["CAPEX_h2", "OPEX_h2", "OPEX_el"]
        return [outputs[key] for key in out_keys]


class H2Cost_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = H2Cost(**insta_inp)
        super().__init__(
            inputs=model.inputs,
            outputs=model.outputs,
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


class shared_cost:
    """
    Shared Cost Model - Calculates costs for electrical infrastructure, land rental, and tower.
    """

    def __init__(
        self,
        grid_connection_cost_per_mw,
        grid_h2_connection_cost_per_kg_h,
        grid_thermal_connection_cost_per_mwt,
        land_cost_m2,
        BOS_soft_cost,
        tower_cost_per_m,
    ):
        # super().__init__()
        # Set cost parameters
        self.BOS_soft_cost = BOS_soft_cost
        self.grid_connection_cost_per_mw = grid_connection_cost_per_mw
        self.grid_h2_connection_cost_per_kg_h = grid_h2_connection_cost_per_kg_h
        self.grid_thermal_connection_cost_per_mwt = grid_thermal_connection_cost_per_mwt
        self.land_cost_m2 = land_cost_m2
        self.tower_cost_per_m = tower_cost_per_m

        # def setup(self):
        # Define inputs and outputs for shared costs
        # inputs
        self.inputs = [
            ("grid_el_capacity", dict(desc="Grid electrical capacity", units="MW")),
            (
                "grid_heat_capacity",
                dict(desc="Grid Heat connection capacity", units="MW"),
            ),
            ("grid_h2_capacity", dict(desc="Grid Hydrogen capacity", units="kg/h")),
            ("sf_area", dict(desc="Land use area of SolarX", units="m**2")),
            ("tower_height", dict(desc="Total height of the tower", units="m")),
        ]
        # outputs
        self.outputs = [
            ("CAPEX_sh", dict(desc="Shared CAPEX costs")),
            ("OPEX_sh", dict(desc="Shared OPEX costs")),
        ]

    def compute(self, **inputs):
        outputs = {}
        # Calculate shared CAPEX and OPEX costs
        grid_el_capacity = inputs["grid_el_capacity"]
        grid_heat_capacity = inputs["grid_heat_capacity"]
        grid_h2_capacity = inputs["grid_h2_capacity"]
        sf_area = inputs["sf_area"]
        tower_height = inputs["tower_height"]

        # Land and grid connection costs
        land_cost_m2 = self.land_cost_m2
        BOS_soft_cost = self.BOS_soft_cost
        grid_connection_cost_per_mw = self.grid_connection_cost_per_mw

        land_rent = land_cost_m2 * sf_area
        CAPEX_tower = self.tower_cost_per_m * tower_height

        outputs["CAPEX_sh"] = (
            BOS_soft_cost * sf_area
            + grid_connection_cost_per_mw * grid_el_capacity
            + grid_heat_capacity * self.grid_thermal_connection_cost_per_mwt
            + grid_h2_capacity * self.grid_h2_connection_cost_per_kg_h
            + land_rent
            + CAPEX_tower
        )
        outputs["OPEX_sh"] = 0
        out_keys = ["CAPEX_sh", "OPEX_sh"]
        return [outputs[key] for key in out_keys]


class shared_cost_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = shared_cost(**insta_inp)
        super().__init__(
            inputs=model.inputs,
            outputs=model.outputs,
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )
