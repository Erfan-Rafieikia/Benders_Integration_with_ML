from gurobipy import GRB, Model, quicksum
from data import UFLData, HUBData, CMNDData, MCFLData, SSLPData  # Placeholder types
import numpy as np


def _set_params(model: Model) -> None:
    """Set the parameters for the Gurobi solver to suppress console output."""
    model.Params.OutputFlag = 0

def solve_dual_subproblem(problem_type: str, data, subproblem_idx: int, first_stage_values: np.ndarray):
    """
    Solve the dual of the subproblem for a given subproblem index with the given values for the first-stage variable (e.g., open/close facilities in UFL)

    Args:
        problem_type (str): The type of problem (e.g., 'UFL')
        data: Problem data object (e.g., UFLData) with fields: d (demands), c (costs), F (facility indices)
        subproblem_idx (int): Index of subproblem (e.g., index of customer in UFL since each customer is a subproblem)
        first_stage_values (np.ndarray): 1D array of first-stage variable values (e.g., binary open/close status of facilities)

    Returns:
        tuple: (objective_value, lambda_value, mu_values_dict)
    """

    if problem_type.upper() == "UFL":
        i = subproblem_idx
        d_i = data.d[i]  # demand of customer i

        with Model("UFL_Dual_Subproblem") as mod:
            _set_params(mod)  # Optional: set Gurobi parameters

            # Dual variables
            lambda_i = mod.addVar(name="lambda")
            mu = mod.addVars(data.F, lb=0, name="mu")

            # Objective: maximize lambda_i - sum_j mu_ij * y_j
            mod.setObjective(
                lambda_i - quicksum(mu[j] * first_stage_values[j] for j in data.F),
                GRB.MAXIMIZE
            )

            # Constraints: lambda_i - mu_ij <= c_ij * d_i
            mod.addConstrs(
                (lambda_i - mu[j] <= data.c[i, j] * d_i for j in data.F),
                name="DualConstraint"
            )

            mod.optimize()

            if mod.status != GRB.OPTIMAL:
                raise RuntimeError(f"Dual subproblem not solved to optimality. Status: {mod.status}")

            lambda_val = lambda_i.X
            mu_vals = {j: mu[j].X for j in data.F}
            obj_val = mod.ObjVal

            return obj_val, lambda_val, mu_vals

    elif problem_type.upper() == "HUB":
        pass  # TODO: Implement dual subproblem for HUB

    elif problem_type.upper() == "CMND":
        pass  # TODO: Implement dual subproblem for CMND

    elif problem_type.upper() == "MCFL":
        pass  # TODO: Implement dual subproblem for MCFL

    elif problem_type.upper() == "SSLP":
        pass  # TODO: Implement dual subproblem for SSLP

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
