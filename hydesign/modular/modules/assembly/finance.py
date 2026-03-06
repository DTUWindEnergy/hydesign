from hydesign.modular.modules.assembly.contracts import (
    AssemblyComponentSpec,
    AssemblyModuleSpec,
)

FINANCE_ASSEMBLY_MODULE = AssemblyModuleSpec(
    name="finance",
    description="Cost and finance contribution to the WSB assembly.",
    source_of_truth=(
        "hydesign/assembly/hpp_assembly.py:582-685",
        "hydesign/assembly/hpp_assembly.py:769-784",
        "hydesign/assembly/hpp_assembly.py:827-859",
    ),
    depends_on=("base", "wind", "solar", "battery", "operations"),
    import_lines=(
        "from .costs import battery_cost_comp as battery_cost",
        "from .costs import pvp_cost_comp as pvp_cost",
        "from .costs import shared_cost_comp as shared_cost",
        "from .costs import wpp_cost_comp as wpp_cost",
        "from .finance import finance_comp as finance",
    ),
    component_specs=(
        AssemblyComponentSpec(
            subsystem_name="wpp_cost",
            constructor_code=(
                "wpp_cost(\n"
                "    wind_turbine_cost=sim_pars['wind_turbine_cost'],\n"
                "    wind_civil_works_cost=sim_pars['wind_civil_works_cost'],\n"
                "    wind_fixed_onm_cost=sim_pars['wind_fixed_onm_cost'],\n"
                "    wind_variable_onm_cost=sim_pars['wind_variable_onm_cost'],\n"
                "    d_ref=sim_pars['d_ref'],\n"
                "    hh_ref=sim_pars['hh_ref'],\n"
                "    p_rated_ref=sim_pars['p_rated_ref'],\n"
                "    N_time=N_time,\n"
                ")"
            ),
        ),
        AssemblyComponentSpec(
            subsystem_name="pvp_cost",
            constructor_code=(
                "pvp_cost(\n"
                "    solar_PV_cost=sim_pars['solar_PV_cost'],\n"
                "    solar_hardware_installation_cost=sim_pars['solar_hardware_installation_cost'],\n"
                "    solar_inverter_cost=sim_pars['solar_inverter_cost'],\n"
                "    solar_fixed_onm_cost=sim_pars['solar_fixed_onm_cost'],\n"
                ")"
            ),
        ),
        AssemblyComponentSpec(
            subsystem_name="battery_cost",
            constructor_code=(
                "battery_cost(\n"
                "    battery_energy_cost=sim_pars['battery_energy_cost'],\n"
                "    battery_power_cost=sim_pars['battery_power_cost'],\n"
                "    battery_BOP_installation_commissioning_cost=sim_pars['battery_BOP_installation_commissioning_cost'],\n"
                "    battery_control_system_cost=sim_pars['battery_control_system_cost'],\n"
                "    battery_energy_onm_cost=sim_pars['battery_energy_onm_cost'],\n"
                "    life_y=life_y,\n"
                "    battery_price_reduction_per_year=battery_price_reduction_per_year,\n"
                ")"
            ),
        ),
        AssemblyComponentSpec(
            subsystem_name="shared_cost",
            constructor_code=(
                "shared_cost(\n"
                "    hpp_BOS_soft_cost=sim_pars['hpp_BOS_soft_cost'],\n"
                "    hpp_grid_connection_cost=sim_pars['hpp_grid_connection_cost'],\n"
                "    land_cost=sim_pars['land_cost'],\n"
                ")"
            ),
        ),
        AssemblyComponentSpec(
            subsystem_name="finance",
            constructor_code=(
                "finance(\n"
                "    N_time=N_time,\n"
                "    depreciation_yr=sim_pars['depreciation_yr'],\n"
                "    depreciation=sim_pars['depreciation'],\n"
                "    inflation_yr=sim_pars['inflation_yr'],\n"
                "    inflation=sim_pars['inflation'],\n"
                "    ref_yr_inflation=sim_pars['ref_yr_inflation'],\n"
                "    phasing_yr=sim_pars['phasing_yr'],\n"
                "    phasing_CAPEX=sim_pars['phasing_CAPEX'],\n"
                "    life_y=life_y,\n"
                ")"
            ),
            key_map_code="{'CAPEX_el': 'CAPEX_sh', 'OPEX_el': 'OPEX_sh', 'penalty_t': 'penalty_t_with_deg'}",
        ),
    ),
    init_set_val_lines=(
        'prob.set_val("wind_WACC", sim_pars["wind_WACC"])',
        'prob.set_val("solar_WACC", sim_pars["solar_WACC"])',
        'prob.set_val("battery_WACC", sim_pars["battery_WACC"])',
        'prob.set_val("tax_rate", sim_pars["tax_rate"])',
        'prob.set_val("land_use_per_solar_MW", sim_pars["land_use_per_solar_MW"])',
    ),
    output_labels=(
        "NPV_over_CAPEX",
        "NPV [MEuro]",
        "IRR",
        "LCOE [Euro/MWh]",
        "Revenues [MEuro]",
        "CAPEX [MEuro]",
        "OPEX [MEuro]",
        "Wind CAPEX [MEuro]",
        "Wind OPEX [MEuro]",
        "PV CAPEX [MEuro]",
        "PV OPEX [MEuro]",
        "Batt CAPEX [MEuro]",
        "Batt OPEX [MEuro]",
        "Shared CAPEX [MEuro]",
        "Shared OPEX [MEuro]",
        "penalty lifetime [MEuro]",
        "Break-even PPA price [Euro/MWh]",
    ),
    output_terms=(
        "prob['NPV_over_CAPEX']",
        "prob['NPV'] / 1e6",
        "prob['IRR']",
        "prob['LCOE']",
        "prob['revenues'] / 1e6",
        "prob['CAPEX'] / 1e6",
        "prob['OPEX'] / 1e6",
        "prob.get_val('finance.CAPEX_w') / 1e6",
        "prob.get_val('finance.OPEX_w') / 1e6",
        "prob.get_val('finance.CAPEX_s') / 1e6",
        "prob.get_val('finance.OPEX_s') / 1e6",
        "prob.get_val('finance.CAPEX_b') / 1e6",
        "prob.get_val('finance.OPEX_b') / 1e6",
        "prob.get_val('finance.CAPEX_el') / 1e6",
        "prob.get_val('finance.OPEX_el') / 1e6",
        "prob['penalty_lifetime'] / 1e6",
        "prob['break_even_PPA_price']",
    ),
)
