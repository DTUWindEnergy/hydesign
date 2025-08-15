import openmdao.api as om

from hydesign.openmdao_wrapper import ComponentWrapper


class shared_cost:
    """Pure Python Electrical infrastructure and land rent cost model"""

    def __init__(self, hpp_BOS_soft_cost, hpp_grid_connection_cost, land_cost):
        """Initialization of the shared costs model

        Parameters
        ----------
        hpp_BOS_soft_cost : Balancing of system cost [Euro/MW]
        hpp_grid_connection_cost : Grid connection cost [Euro/MW]
        land_cost : Land rent cost [Euro/km**2]
        """
        self.hpp_BOS_soft_cost = hpp_BOS_soft_cost
        self.hpp_grid_connection_cost = hpp_grid_connection_cost
        self.land_cost = land_cost

    def compute(self, solar_MW, Awpp, Apvp, **kwargs):
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
        land_cost = self.land_cost
        hpp_BOS_soft_cost = self.hpp_BOS_soft_cost
        hpp_grid_connection_cost = self.hpp_grid_connection_cost
        land_rent_wind = land_cost * Awpp
        land_rent_pv = land_cost * Apvp

        outputs = {}
        outputs["CAPEX_sh_s"] = (
            hpp_BOS_soft_cost + hpp_grid_connection_cost
        ) * solar_MW + land_rent_pv  # MODIFICA!

        if Awpp > Apvp:
            outputs["CAPEX_sh_w"] = (
                land_rent_wind - land_rent_pv
            )  # (hpp_BOS_soft_cost  + hpp_grid_connection_cost) * (G_MW-solar_MW)  # We don't include the land ofthe PV, because they can be occupy the same space of the wt
        else:
            outputs["CAPEX_sh_w"] = 0

        outputs["OPEX_sh"] = 0
        return outputs["CAPEX_sh_s"], outputs["CAPEX_sh_w"], outputs["OPEX_sh"]


class shared_cost_comp(ComponentWrapper):
    def __init__(self, hpp_BOS_soft_cost, hpp_grid_connection_cost, land_cost):
        model = shared_cost(hpp_BOS_soft_cost, hpp_grid_connection_cost, land_cost)
        super().__init__(
            inputs=[
                (
                    "solar_MW",
                    {"units": "MW", "desc": "AC nominal capacity of the PV plant"},
                ),
                (
                    "Awpp",
                    {"units": "km**2", "desc": "Land use area of the wind power plant"},
                ),
                (
                    "Apvp",
                    {
                        "units": "km**2",
                        "desc": "Land use area of the solar power plant",
                    },
                ),
            ],
            outputs=[
                (
                    "CAPEX_sh_s",
                    {
                        "desc": "CAPEX electrical infrastructure/ land rent for the added pv"
                    },
                ),
                (
                    "CAPEX_sh_w",
                    {
                        "desc": "CAPEX electrical infrastructure/ land rent for the wind stand-alone"
                    },
                ),
                ("OPEX_sh", {"desc": "OPEX electrical infrastructure/ land rent"}),
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


class decommissioning_cost:
    """Pure Python Decommissioning cost model"""

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
        self.decommissioning_cost_w = decommissioning_cost_w
        self.decommissioning_cost_s = decommissioning_cost_s

    def compute(self, CAPEX_w, solar_MW, **kwargs):
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

        decommissioning_cost_w = self.decommissioning_cost_w
        decommissioning_cost_s = self.decommissioning_cost_s
        outputs = {}
        outputs["decommissioning_cost_tot_w"] = decommissioning_cost_w * CAPEX_w
        outputs["decommissioning_cost_tot_s"] = decommissioning_cost_s * solar_MW
        return (
            outputs["decommissioning_cost_tot_w"],
            outputs["decommissioning_cost_tot_s"],
        )


class decommissioning_cost_comp(ComponentWrapper):
    def __init__(self, decommissioning_cost_w, decommissioning_cost_s):
        model = decommissioning_cost(decommissioning_cost_w, decommissioning_cost_s)
        super().__init__(
            inputs=[
                ("CAPEX_w", {"desc": "CAPEX of the wind plant"}),
                (
                    "solar_MW",
                    {"units": "MW", "desc": "AC nominal capacity of the PV plant"},
                ),
            ],
            outputs=[
                (
                    "decommissioning_cost_tot_w",
                    {"desc": "Decommissioning cost of the entire wind plant"},
                ),
                (
                    "decommissioning_cost_tot_s",
                    {"desc": "Decommissioning cost of the entire PV plant"},
                ),
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )
