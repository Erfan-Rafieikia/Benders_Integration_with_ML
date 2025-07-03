import numpy as np
from scipy.spatial.distance import cdist
import gurobipy as gp
from gurobipy import GRB

def select_subproblems_based_on_features(feature_vectors: dict, method: str = "p-median", p: int = 5):
    """
    Selects a subset of subproblems (scenarios) based on feature vectors using the specified selection method.

    Args:
        feature_vectors (dict): A dictionary {subproblem_id: feature_vector}
        method (str): Method to use for selection. Supported: "p-median", ...
        p (int): Number of subproblems to select. If None, defaults to ceil(20% of total)

    Returns:
        selected_subproblems (list): List of selected subproblem IDs
        assignments (dict): Mapping from each subproblem to its closest selected one
    """
    if not feature_vectors:
        raise ValueError("Feature vector dictionary is empty.")

    scenario_ids = list(feature_vectors.keys())
    num_scenarios = len(scenario_ids)
    if p is None:
        p = max(1, int(0.2 * num_scenarios))

    if method == "p-median":
        return solve_p_median(feature_vectors, p)

    elif method == "other-method":
        # TODO: Implement other scenario selection logic
        pass

    else:
        raise NotImplementedError(f"Selection method '{method}' is not implemented.")

def solve_p_median(feature_vectors, p):
    scenario_ids = list(feature_vectors.keys())
    num_scenarios = len(scenario_ids)

    if p <= 0 or p > num_scenarios:
        raise ValueError(f"Invalid value for p: {p}. Must be between 1 and {num_scenarios}.")

    feature_matrix = np.array([feature_vectors[s] for s in scenario_ids])
    distance_matrix = cdist(feature_matrix, feature_matrix, metric="euclidean")

    model = gp.Model("p-Median")
    model.Params.OutputFlag = 0

    x = model.addVars(num_scenarios, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_scenarios, num_scenarios, vtype=GRB.BINARY, name="y")

    model.setObjective(
        gp.quicksum(distance_matrix[i, j] * y[i, j] for i in range(num_scenarios) for j in range(num_scenarios)),
        GRB.MINIMIZE
    )

    model.addConstr(gp.quicksum(x[i] for i in range(num_scenarios)) == p, name="Select_p_Medians")

    for i in range(num_scenarios):
        model.addConstr(gp.quicksum(y[i, j] for j in range(num_scenarios)) == 1, name=f"Assign_{i}")
        for j in range(num_scenarios):
            model.addConstr(y[i, j] <= x[j], name=f"AssignOnlyIfSelected_{i}_{j}")

    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"Warning: Gurobi status {model.status}. Returning empty result.")
        return [], {}

    selected = [scenario_ids[i] for i in range(num_scenarios) if x[i].X > 0.5]
    assignments = {
        scenario_ids[i]: scenario_ids[j]
        for i in range(num_scenarios)
        for j in range(num_scenarios)
        if y[i, j].X > 0.5
    }
    return selected, assignments
