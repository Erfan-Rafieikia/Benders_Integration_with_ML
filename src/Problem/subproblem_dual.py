from gurobipy import GRB, Model, quicksum
from data import UFLData, HUBData, CMNDData, MCFLData, SSLPData  # Placeholder types


def _set_params(model: Model) -> None:
    """Set the parameters for the Gurobi solver to suppress console output."""
    model.Params.OutputFlag = 0


def solve_dual_subproblem(problem_type: str, data, subproblem_idx: int, first_stage_values: dict):
    """
    Solve the dual of the subproblem for a given scenario index and first-stage variable values.

    Args:
        problem_type (str): The type of problem (e.g., 'UFL')
        dat: Problem data class (UFLData, HUBData, etc.)
        subproblem_idx (int): Index of the subproblem (e.g., customer index for UFL)
        first_stage_values (dict): Dictionary mapping first-stage variable index to 0/1 (e.g., facility open decisions)

    Returns:
        tuple: (objective_value, lambda_value, mu_values)
    """

    if problem_type.upper() == "UFL":
        i = subproblem_idx
        d_i = data.d[i]  # demand for subproblem i

        with Model("UFL_Dual_Subproblem") as mod:
            _set_params(mod)

            lambda_i = mod.addVar(name="lambda")
            mu = mod.addVars(data.F, lb=0, name="mu")

            # Objective: maximize lambda_i - sum_j mu_ij * y_j
            mod.setObjective(lambda_i - quicksum(mu[j] * first_stage_values[j] for j in data.F), GRB.MAXIMIZE)

            # Constraints: lambda_i - mu_ij <= c_ij * d_i
            mod.addConstrs(
                (lambda_i - mu[j] <= data.c[i, j] * d_i for j in data.F),
                name="DualConstraint"
            )

            mod.optimize()

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