# basic libraries
from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd
import pyomo.environ as pyo
from docplex.mp.constants import ComparisonType

# import xarray as xr
from docplex.mp.model import Model


class EMSCoreABC(ABC):
    """
    Abstract interface for EMS core classes.
    Concrete implementations must provide:
      - inputs
      - inputs_batch
      - outputs_batch
      - final_process
    """

    def __init__(self, **kwargs):
        pass

    def solve(self, **kwargs):
        ems_solver = self.ems_solver
        for ib, batch in enumerate(self.batches_all):
            self.batch()
            if ems_solver == "cplex":
                self.cplex()
            elif ems_solver == "pyomo":
                self.pyomo()
            elif ems_solver == "ortools":
                self.ortools()
            out_ems

    def inputs(self, **kwargs) -> Dict[str, Any]:
        """Prepare and return the initial inputs / allocated arrays and meta values."""
        return {}

    def cplex(self, mdl, **kwargs) -> Dict[str, Any]:
        """Merge batch outputs back into global arrays stored in kwargs and return kwargs."""
        raise NotImplementedError

    # def pyomo(self, batch: Iterable[int], **kwargs) -> Dict[str, Any]:
    #     """Merge batch outputs back into global arrays stored in kwargs and return kwargs."""
    #     raise NotImplementedError

    # def ortools(self, batch: Iterable[int], **kwargs) -> Dict[str, Any]:
    #     """Merge batch outputs back into global arrays stored in kwargs and return kwargs."""
    #     raise NotImplementedError
    def cplex_variables(self, mdl):
        pass

    def cplex_constraints(self, mdl):
        pass

    def cplex_opjective(self, mdl):
        pass

    def cplex_outputs(self, sol):
        pass

    def ortools_variables(self, solver):
        pass

    def ortools_constraints(self, solver):
        pass

    def ortools_objective(self, solver):
        pass

    def ortools_outputs(self):
        pass

    def pyomo_variables(self, mdl):
        pass

    def pyomo_constraints(self, mdl):
        pass

    def pyomo_opjective(self, mdl):
        pass

    def pyomo_outputs(self, mdl):
        pass

    def final_process(self, **kwargs) -> Dict[str, Any]:
        """
        Final processing step.
        """
        return kwargs

    def update_final_outputs(self):
        return {}

    def update_constraint_rhs(self, mdl, constraint_name, added_value):
        constr = mdl.get_constraint_by_name(constraint_name)
        if constr is None:
            raise ValueError(f"Constraint '{constraint_name}' not found in the model.")

        lhs_expr = constr.lhs
        rhs_value = constr.rhs
        ct_type = constr.sense

        # Remove old constraint
        mdl.remove_constraint(constr)

        # Add new constraint with updated RHS
        if ct_type == ComparisonType.EQ:
            mdl.add_constraint(
                lhs_expr == rhs_value + added_value, ctname=constraint_name
            )
        elif ct_type == ComparisonType.LE:
            mdl.add_constraint(
                lhs_expr <= rhs_value + added_value, ctname=constraint_name
            )
        elif ct_type == ComparisonType.GE:
            mdl.add_constraint(
                lhs_expr >= rhs_value + added_value, ctname=constraint_name
            )
        else:
            raise ValueError(f"Unknown constraint sense: {ct_type}")

    def add_to_objective(self, mdl, addition_expr):
        """
        Add a new expression to the current model objective.

        Parameters
        ----------
        mdl : docplex.mp.model.Model
            The CPLEX model.
        addition_expr : docplex expression
            Expression to add to the existing objective.

        Returns
        -------
        None (updates model in-place)
        """
        # Get the existing objective expression and sense
        old_obj = mdl.objective_expr
        sense = mdl.objective_sense

        # Remove old objective
        mdl.remove_objective()

        # Add new objective: old + addition
        if sense.name.lower() == "maximize":
            mdl.maximize(old_obj + addition_expr)
        else:
            mdl.minimize(old_obj + addition_expr)

    def add_to_objective_pyomo(self, mdl, addition_expr, obj_name="OBJ"):
        """
        Add `addition_expr` to model objective.
        - tries model.OBJ first (matches your code),
        - falls back to the first active scalar Objective,
        - if none exists, creates model.OBJ with the given expr.
        """
        # prefer explicit name (e.g. mdl.OBJ)
        target = getattr(mdl, obj_name, None)

        # update objective expression (preserves sense)
        target.set_value(target.expr + addition_expr)

    # update function (very small, similar to your original)
    def update_constraint_rhs_pyomo(self, mdl, constraint_name, added_value, idx=None):
        comp = getattr(mdl, constraint_name, None)
        if comp is None:
            raise ValueError(f"Constraint '{constraint_name}' not found.")
        # target single ConstraintData if idx provided
        target = comp[idx] if (idx is not None and comp.is_indexed()) else comp
        c = target  # ConstraintData
        lb, ub, body = c.lower, c.upper, c.body
        if (lb is not None) and (ub is not None) and float(lb) == float(ub):
            c.set_value(body == (ub + added_value))
        elif ub is not None:
            c.set_value(body <= (ub + added_value))
        elif lb is not None:
            c.set_value((lb + added_value) <= body)
        else:
            raise ValueError("No bounds to update.")

    def update_constraint_rhs_ortools(
        self, solver, constraint_name, addition_expr: float, coeff: float = 1
    ):
        """
        Add a numeric 'addition_expr' to the RHS of the named equality constraint.
        This function shifts the constraint's bounds (lb and ub) by addition_expr.
        """
        target = None
        for c in solver.constraints():
            if c.name() == constraint_name:
                target = c
                break
        if target is None:
            raise KeyError("constraint not found")

        target.SetCoefficient(addition_expr, -coeff)

    from ortools.linear_solver import pywraplp


def add_piecewise_relation_ortools(solver, P_var, H_var, breakpoints, name_prefix="pw"):
    """
    Adds linear constraints to solver that enforce:
        H_var == piecewise_linear(P_var)
    where breakpoints is a list of (P_i, H_i) sorted by P_i (increasing).
    Uses a one-hot segment selection + interpolation variable per segment.

    Returns:
      dict with keys 'b_vars' (list of binary vars) and 'w_vars' (list of interpolation vars).
    """
    # sanity checks
    if len(breakpoints) < 2:
        raise ValueError(
            "Need at least two breakpoints to define a piecewise linear function."
        )

    # ensure sorted by P
    bp = sorted(breakpoints, key=lambda x: x[0])
    P_points = [float(p) for p, h in bp]
    H_points = [float(h) for p, h in bp]
    S = len(bp) - 1  # number of segments

    b_vars = []
    w_vars = []

    # create variables per segment
    for s in range(S):
        b = solver.IntVar(0, 1, f"{name_prefix}_b_{s}")  # binary
        # interpolation variable w_s: continuous in [0,1] but we'll also bound by b
        # pywraplp uses NumVar for continuous
        w = solver.NumVar(0.0, 1.0, f"{name_prefix}_w_{s}")
        b_vars.append(b)
        w_vars.append(w)

        # link w_s to b_s: w_s <= b_s
        # This enforces w_s == 0 when b_s == 0. When b_s == 1, w_s in [0,1].
        solver.Add(w <= b)

    # exactly one segment active
    solver.Add(sum(b_vars) == 1)

    # Bound P_var to lie inside full breakpoint range (optional but recommended)
    P_min = P_points[0]
    P_max = P_points[-1]
    # solver.Add(P_var >= P_min)
    # solver.Add(P_var <= P_max)

    # Build linear expressions for P and H
    # P_expr  = sum_s (P_s * b_s + (P_{s+1} - P_s) * w_s)
    # H_expr  = sum_s (H_s * b_s + (H_{s+1} - H_s) * w_s)
    P_expr = None
    H_expr = None
    for s in range(S):
        Ps = P_points[s]
        Ps1 = P_points[s + 1]
        Hs = H_points[s]
        Hs1 = H_points[s + 1]
        deltaP = Ps1 - Ps
        deltaH = Hs1 - Hs

        # term for P: Ps * b_s + deltaP * w_s
        termP = solver.NumVar(
            -solver.infinity(), solver.infinity(), f"{name_prefix}_termP_{s}"
        )
        solver.Add(termP == Ps * b_vars[s] + deltaP * w_vars[s])

        termH = solver.NumVar(
            -solver.infinity(), solver.infinity(), f"{name_prefix}_termH_{s}"
        )
        solver.Add(termH == Hs * b_vars[s] + deltaH * w_vars[s])

        if P_expr is None:
            P_expr = termP
            H_expr = termH
        else:
            P_expr = P_expr + termP
            H_expr = H_expr + termH

    # Now enforce equality with the model variables
    solver.Add(P_var == P_expr)
    solver.Add(H_var == H_expr)

    return {"b_vars": b_vars, "w_vars": w_vars, "P_range": (P_min, P_max)}


def split_in_batch(array, N):
    batch = []
    counter = 0
    while counter * N < len(array):
        if (counter + 1) * N > len(array):
            end = len(array)
        else:
            end = (counter + 1) * N
        batch += [array[counter * N : end]]
        counter += 1
    return batch


# helper to safely read an indexed Pyomo Var into a pandas Series
def series_from_indexed_var(var, idx_map):
    """
    var: Pyomo Var indexed by integers 0..T-1
    idx_map: list-like of labels for the pandas index (e.g. time_batch)
    returns: pd.Series indexed by idx_map with numeric values
    """
    values = []
    for i in range(len(idx_map)):
        try:
            values.append(pyo.value(var[i]))
        except KeyError:
            # missing index in var
            values.append(0.0)
        except Exception:
            # value not available (uninitialized) -> preserve failure behavior or set nan/0
            values.append(0.0)
    return pd.Series(values, index=idx_map)
