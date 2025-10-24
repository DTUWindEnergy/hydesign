# basic libraries
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List

import numpy as np

# from numpy import newaxis as na
import pandas as pd
import pyomo.environ as pyo
from docplex.mp.constants import ComparisonType

# import xarray as xr
from docplex.mp.model import Model


class EMSCoreInterface(ABC):
    """
    Abstract interface for EMS core classes.
    Concrete implementations must provide:
      - inputs
      - inputs_batch
      - outputs_batch
      - final_process
    """

    def __init__(self, **kwargs):
        self.wind_ts = kwargs["wind_ts"]

        self.solar_ts = kwargs["solar_ts"]

        self.hpp_grid_connection = kwargs.get("hpp_grid_connection", 0)
        self.price_ts = kwargs["price_ts"]
        self.peak_hr_quantile = kwargs.get("peak_hr_quantile", 0.9)
        self.n_full_power_hours_expected_per_day_at_peak_price = kwargs.get(
            "n_full_power_hours_expected_per_day_at_peak_price", 3
        )
        self.batch_size = kwargs.get("batch_size", 110)

        n_ts = len(self.price_ts)
        self.n_ts = n_ts

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

    def get_final_outputs(self):
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
