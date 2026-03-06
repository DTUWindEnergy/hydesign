from hydesign.modular.modules.assembly.contracts import (
    AssemblyComponentSpec,
    AssemblyModuleSpec,
)

BASE_ASSEMBLY_MODULE = AssemblyModuleSpec(
    name="base",
    description=(
        "Shared WSB assembly infrastructure: weather preprocessing, baseline EMS, "
        "common OpenMDAO parameter wiring, and system-level reporting terms."
    ),
    source_of_truth=(
        "hydesign/assembly/hpp_assembly.py:376-621",
        "hydesign/assembly/hpp_assembly.py:727-887",
    ),
    import_lines=(
        "from .weather import ABL_comp as ABL",
        "from .ems import ems_comp as ems_static",
    ),
    init_setup_lines=(
        "ems_args = {",
        '    "N_time": N_time,',
        '    "weeks_per_season_per_year": weeks_per_season_per_year,',
        '    "life_y": life_y,',
        '    "ems_type": ems_type,',
        "}",
        "# Generated WSB assembly composes the local EMS artifact explicitly.",
        "ems = ems_static",
    ),
    component_specs=(
        AssemblyComponentSpec(
            subsystem_name="abl",
            constructor_code="ABL(weather_fn=input_ts_fn, N_time=N_time)",
        ),
        AssemblyComponentSpec(
            subsystem_name="ems",
            constructor_code="ems(**ems_args)",
        ),
    ),
    init_set_val_lines=(
        'prob.set_val("price_t", price)',
        'prob.set_val("G_MW", sim_pars["G_MW"])',
        'prob.set_val("battery_depth_of_discharge", sim_pars["battery_depth_of_discharge"])',
        'prob.set_val("battery_charge_efficiency", sim_pars["battery_charge_efficiency"])',
        'prob.set_val("peak_hr_quantile", sim_pars["peak_hr_quantile"])',
        "prob.set_val(",
        '    "n_full_power_hours_expected_per_day_at_peak_price",',
        '    sim_pars["n_full_power_hours_expected_per_day_at_peak_price"],',
        ")",
    ),
    evaluate_post_run_lines=(
        "AEP = ((prob['wind_t'].mean() + prob['solar_t'].mean()) * 1e-3 * 24 * 365)",
        "AEP_deg = (",
        "    (prob['wind_t_ext_deg'].mean() + prob['solar_t_ext_deg'].mean())",
        "    * 1e-3",
        "    * 24",
        "    * 365",
        ")",
    ),
    output_labels=(
        "Mean Annual Electricity Sold [GWh]",
        "GUF",
        "grid [MW]",
        "Total curtailment [GWh]",
        "Total curtailment with deg [GWh]",
        "Plant area [km2]",
        "AEP [GWh]",
        "AEP with degradation [GWh]",
    ),
    output_terms=(
        "prob['mean_AEP'] / 1e3",
        "prob['mean_AEP'] / (self.sim_pars['G_MW'] * 365 * 24)",
        "self.sim_pars['G_MW']",
        "prob['total_curtailment'] / 1e3",
        "prob['total_curtailment_with_deg'] / 1e3",
        "max(Awpp, prob.get_val('shared_cost.Apvp'))",
        "AEP",
        "AEP_deg",
    ),
)
