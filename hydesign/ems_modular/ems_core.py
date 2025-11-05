# basic libraries
import pyomo.environ as pyo
from docplex.mp.model import Model
from ortools.linear_solver import pywraplp

from .ems_core_combined import EMSCoreCombined


def ems_core(**kwargs):
    ems_solver = kwargs["ems_solver"]
    #
    out_ems = {}
    ems_core_combined = EMSCoreCombined(**kwargs)

    batches = ems_core_combined.batches

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
        ems_core_combined.update_final_outputs(out_ems)

    elif ems_solver == "ortools":
        for ib, batch in enumerate(batches):
            # Create linear solver (GLOP)
            solver = pywraplp.Solver.CreateSolver("GLOP")
            if solver is None:
                raise RuntimeError(
                    "OR-Tools GLOP solver not available. Install ortools or choose another solver backend."
                )

            #
            ems_core_combined.solver_init(batch, **kwargs)
            ems_core_combined.ortools_variables(solver)
            ems_core_combined.ortools_objective(solver)
            ems_core_combined.ortools_constraints(solver)
            # Solve
            result_status = solver.Solve()
            if result_status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                raise RuntimeError(
                    f"Solver failed or returned infeasible (status {result_status})."
                )
            ems_core_combined.ortools_outputs()
        ems_core_combined.update_final_outputs(out_ems)
        # ------------

    elif ems_solver == "pyomo":
        for ib, batch in enumerate(batches):
            ## cplex share
            mdl = pyo.ConcreteModel()

            ems_core_combined.solver_init(batch, **kwargs)
            ems_core_combined.pyomo_variables(mdl)
            ems_core_combined.pyomo_constraints(mdl)
            ems_core_combined.pyomo_opjective(mdl)
            # Solve
            # solve with CPLEX via Pyomo
            solver = pyo.SolverFactory("cplex")  # glpk
            result = solver.solve(mdl, tee=False)  # tee=True if you want solver log
            ems_core_combined.pyomo_outputs(mdl)
        ems_core_combined.update_final_outputs(out_ems)

    # final process on outputs
    ems_core_combined.final_process(**out_ems)

    return out_ems
