# basic libraries
import pyomo.environ as pyo
from docplex.mp.model import Model
from ortools.linear_solver import pywraplp

from .ems_core_battery import EMSCoreBattery
from .ems_core_combined import EMSCoreCombined
from .ems_core_defaults import EMSCoreDefaults
from .ems_core_solar import EMSCoreSolar
from .ems_core_wind import EMSCoreWind


def ems_core(**kwargs):
    technology = kwargs.get("technology", {"wind", "solar", "battery"})
    ems_solver = kwargs.get("ems_solver", "cplex")

    #
    out_ems = {}
    ems_core_combined = EMSCoreCombined(**kwargs)

    batches = ems_core_combined.batches

    ems_solver = "cplex"  # "cplex"

    if ems_solver == "cplex":
        for ib, batch in enumerate(batches):
            ## cplex share
            mdl = Model(name="EMS")
            mdl.context.cplex_parameters.threads = 1
            mdl.context.cplex_parameters.emphasis.mip = 1

            #
            ems_core_combined.solver_init(batch, **kwargs)
            ems_core_combined.cplex_variables(mdl)
            ems_core_combined.cplex_opjective(mdl)
            ems_core_combined.cplex_constraints(mdl)
            sol = mdl.solve(log_output=False)
            ems_core_combined.cplex_outputs(sol)
        out_ems = ems_core_combined.get_final_outputs()

    elif ems_solver == "ortools":
        for ib, batch in enumerate(batches):
            # Create linear solver (GLOP)
            solver = pywraplp.Solver.CreateSolver("GLOP")
            if solver is None:
                raise RuntimeError(
                    "OR-Tools GLOP solver not available. Install ortools or choose another solver backend."
                )

            ems_core_defaults.solver_init(batch, **kwargs)
            if "wind" in technology:
                ems_core_wind.solver_init(batch, **kwargs)

            if "solar" in technology:
                ems_core_solar.solver_init(batch, **kwargs)

            if "battery" in technology:
                ems_core_battery.solver_init(batch, **kwargs)

                # ----------
            # vars and constraints

            ems_core_defaults.ortools_variables(solver)
            if "wind" in technology:
                ems_core_wind.ortools_variables(solver)

            if "solar" in technology:
                ems_core_solar.ortools_variables(solver)

            if "battery" in technology:
                ems_core_battery.ortools_variables(solver)

            ems_core_defaults.ortools_objective(solver)
            if "wind" in technology:
                ems_core_wind.ortools_objective(solver)

            if "solar" in technology:
                ems_core_solar.ortools_objective(solver)

            if "battery" in technology:
                ems_core_battery.ortools_objective(solver)

            ems_core_defaults.ortools_constraints(solver)
            if "wind" in technology:
                ems_core_wind.ortools_constraints(solver)

            if "solar" in technology:
                ems_core_solar.ortools_constraints(solver)

            if "battery" in technology:
                ems_core_battery.ortools_constraints(solver)

            # ------------
            # Solve
            result_status = solver.Solve()
            if result_status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                raise RuntimeError(
                    f"Solver failed or returned infeasible (status {result_status})."
                )

            ems_core_defaults.ortools_outputs()
            if "wind" in technology:
                ems_core_wind.ortools_outputs()
            if "solar" in technology:
                ems_core_solar.ortools_outputs()
            if "battery" in technology:
                ems_core_battery.ortools_outputs()

        # update ems outputs
        out_ems = ems_core_defaults.get_final_outputs()
        out_ems.update(ems_core_wind.get_final_outputs())
        out_ems.update(ems_core_solar.get_final_outputs())
        out_ems.update(ems_core_battery.get_final_outputs())

    elif ems_solver == "pyomo":
        for ib, batch in enumerate(batches):
            ## cplex share
            mdl = pyo.ConcreteModel()

            # init
            ems_core_defaults.solver_init(batch, **kwargs)
            if "wind" in technology:
                ems_core_wind.solver_init(batch, **kwargs)
            if "solar" in technology:
                ems_core_solar.solver_init(batch, **kwargs)
            if "battery" in technology:
                ems_core_battery.solver_init(batch, **kwargs)

            # variables
            ems_core_defaults.pyomo_variables(mdl, **kwargs)
            if "wind" in technology:
                ems_core_wind.pyomo_variables(mdl)
            if "solar" in technology:
                ems_core_solar.pyomo_variables(mdl)
            if "battery" in technology:
                ems_core_battery.pyomo_variables(mdl)

            # Constraints
            ems_core_defaults.pyomo_constraints(mdl, **kwargs)
            if "wind" in technology:
                ems_core_wind.pyomo_constraints(mdl)
            if "solar" in technology:
                ems_core_solar.pyomo_constraints(mdl)
            if "battery" in technology:
                ems_core_battery.pyomo_constraints(mdl)

            # objective
            ems_core_defaults.pyomo_opjective(mdl, **kwargs)
            if "wind" in technology:
                ems_core_wind.pyomo_opjective(mdl)
            if "solar" in technology:
                ems_core_solar.pyomo_opjective(mdl)
            if "battery" in technology:
                ems_core_battery.pyomo_opjective(mdl)

            # Solve
            # solve with CPLEX via Pyomo

            solver = pyo.SolverFactory("cplex")  # glpk
            # solver.options["threads"] = 6
            result = solver.solve(mdl, tee=False)  # tee=True if you want solver log

            ems_core_defaults.pyomo_outputs(mdl)
            if "wind" in technology:
                ems_core_wind.pyomo_outputs(mdl)
            if "solar" in technology:
                ems_core_solar.pyomo_outputs(mdl)
            if "battery" in technology:
                ems_core_battery.pyomo_outputs(mdl)

        # update ems outputs
        out_ems = ems_core_defaults.get_final_outputs()
        out_ems.update(ems_core_wind.get_final_outputs())
        out_ems.update(ems_core_solar.get_final_outputs())
        out_ems.update(ems_core_battery.get_final_outputs())

    # final process on outputs
    ems_core_combined.final_process(**out_ems)
    # if "wind" in technology:
    #     ems_core_wind.final_process(**out_ems)

    # if "solar" in technology:
    #     ems_core_solar.final_process(**out_ems)

    # if "battery" in technology:
    #     ems_core_battery.final_process(**out_ems)

    return out_ems
