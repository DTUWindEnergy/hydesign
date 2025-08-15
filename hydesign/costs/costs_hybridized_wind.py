import openmdao.api as om

from hydesign.openmdao_wrapper import ComponentWrapper


class shared_cost:
    """Electrical infrastructure and land rent cost model"""

    def __init__(self, hpp_BOS_soft_cost, hpp_grid_connection_cost, land_cost):
        """Initialization of the shared costs model

        Parameters
        ----------
        hpp_BOS_soft_cost : Balancing of system cost [Euro/MW]
        hpp_grid_connection_cost : Grid connection cost [Euro/MW]
        land_cost : Land rent cost [Euro/km**2]
        """
        super().__init__()
        self.hpp_BOS_soft_cost = hpp_BOS_soft_cost
        self.hpp_grid_connection_cost = hpp_grid_connection_cost
        self.land_cost = land_cost

        self.inputs = [
            ("G_MW", dict(desc="Grid capacity", units="MW")),
            ("p_rated", dict(desc="Power rated,", units="MW")),
            (
                "Nwt",
                dict(
                    desc="Number of wind turbines",
                ),
            ),
            ("solar_MW", dict(desc="Solar capacity", units="MW")),
            ("Awpp", dict(desc="Land use area of WPP", units="km**2")),
            ("Apvp", dict(desc="Land use area of SP", units="km**2")),
        ]
        self.outputs = [
            (
                "CAPEX_sh_w",
                {
                    "desc": "CAPEX electrical infrastructure/ land rent for the wind stand-alone"
                },
            ),
            (
                "CAPEX_sh_s",
                {"desc": "CAPEX electrical infrastructure/ land rent for the added PV"},
            ),
            ("OPEX_sh", {"desc": "OPEX electrical infrastructure/ land rent"}),
        ]

    def compute(self, **inputs):
        """Computing the CAPEX and OPEX of the shared land and infrastructure.

        Parameters
        ----------
        G_MW : Grid capacity [MW]
        Awpp : Land use area of the wind power plant [km**2]
        Apvp : Land use area of the solar power plant [km**2]

        Returns
        -------
        CAPEX_sh_w : CAPEX electrical infrastructure/ land rent for the wind stand-alone[Eur]
        CAPEX_sh_s : CAPEX electrical infrastructure/ land rent for the added pv [Eur]
        OPEX_sh : OPEX electrical infrastructure/ land rent [Eur/year]
        """
        outputs = {}
        Nwt = inputs["Nwt"]
        p_rated = inputs["p_rated"]
        # solar_MW = inputs['solar_MW']
        # G_MW = inputs['G_MW']
        Awpp = inputs["Awpp"]
        Apvp = inputs["Apvp"]
        land_cost = self.land_cost
        hpp_BOS_soft_cost = self.hpp_BOS_soft_cost
        hpp_grid_connection_cost = self.hpp_grid_connection_cost

        # if (Awpp >= Apvp):
        #    land_rent = land_cost * Awpp
        # else:

        land_rent_wind = land_cost * Awpp
        land_rent_pv = land_cost * Apvp

        outputs["CAPEX_sh_w"] = (
            hpp_BOS_soft_cost + hpp_grid_connection_cost
        ) * p_rated * Nwt + land_rent_wind  # MODIFICA!

        if Apvp > Awpp:
            outputs["CAPEX_sh_s"] = (
                land_rent_pv - land_rent_wind
            )  # (hpp_BOS_soft_cost  + hpp_grid_connection_cost) * (G_MW-solar_MW)  # We don't include the land ofthe PV, because they can be occupy the same space of the wt
        else:
            outputs["CAPEX_sh_s"] = 0

        outputs["OPEX_sh"] = 0
        out_keys = ["CAPEX_sh_w", "CAPEX_sh_s", "OPEX_sh"]
        return [outputs[key] for key in out_keys]


class shared_cost_comp(ComponentWrapper):
    def __init__(self, hpp_BOS_soft_cost, hpp_grid_connection_cost, land_cost):
        model = shared_cost(hpp_BOS_soft_cost, hpp_grid_connection_cost, land_cost)
        super().__init__(
            inputs=model.inputs,
            outputs=model.outputs,
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


class decommissioning_cost:
    """Decommissioning cost model"""

    def __init__(
        self,
        decommissioning_cost_w,
        decommissioning_cost_s,
    ):
        """Initialization of the decommissioning costs model

        Parameters
        ----------
        decommissioning_cost_w : Decommissioning cost of the wind turbines [Euro/turbine]
        decommissioning_cost_s : Decommissioning cost of the PV [Euro/MW]

        """
        # super().__init__()
        self.decommissioning_cost_w = decommissioning_cost_w
        self.decommissioning_cost_s = decommissioning_cost_s

        # def setup(self):
        self.inputs = [
            ("CAPEX_w", dict(desc="CAPEX wpp")),
            ("solar_MW", dict(desc="Solar capacity", units="MW")),
        ]
        self.outputs = [
            (
                "decommissioning_cost_tot_w",
                dict(desc="Decommissioning cost of the entire wind plant"),
            ),
            (
                "decommissioning_cost_tot_s",
                dict(desc="Decommissioning cost of the entire PV plant"),
            ),
        ]

    def compute(self, **inputs):
        """Computing the decommissioning costs of the entire wind plant and PV plant.

        Parameters
        ----------
        Nwt : Number of wind turbines
        solar_MW : AC nominal capacity of the PV plant [MW]

        Returns
        -------
        decommissioning_cost_tot_w : Decommissioning cost of the entire wind plant [Eur]
        decommissioning_cost_tot_s : Decommissioning cost of the entire PV plant [Eur]
        """
        outputs = {}
        CAPEX_w = inputs["CAPEX_w"]
        solar_MW = inputs["solar_MW"]

        decommissioning_cost_w = self.decommissioning_cost_w
        decommissioning_cost_s = self.decommissioning_cost_s

        outputs["decommissioning_cost_tot_w"] = decommissioning_cost_w * CAPEX_w
        outputs["decommissioning_cost_tot_s"] = decommissioning_cost_s * solar_MW
        out_keys = ["decommissioning_cost_tot_w", "decommissioning_cost_tot_s"]
        return [outputs[key] for key in out_keys]


class decommissioning_cost_comp(ComponentWrapper):
    def __init__(self, decommissioning_cost_w, decommissioning_cost_s):
        model = decommissioning_cost(decommissioning_cost_w, decommissioning_cost_s)
        super().__init__(
            inputs=model.inputs,
            outputs=model.outputs,
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )
