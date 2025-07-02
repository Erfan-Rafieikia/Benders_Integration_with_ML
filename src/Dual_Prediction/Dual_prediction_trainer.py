from gurobipy import Model, GRB, quicksum
import numpy as np


def train_feasible_predictor(problem_type, data, solved_duals, feature_vectors):
    """
    Trains a linear prediction model for each dual variable using features, constrained to satisfy
    feasibility constraints of the corresponding dual subproblem.

    Args:
        problem_type (str): Problem class (e.g., 'UFL')
        data: Instance data (e.g., UFLData)
        solved_duals: dict[sub_idx] -> {'lambda': val, 'mu': {j: val}}
        feature_vectors: dict[sub_idx] -> np.array of features

    Returns:
        dict[str, np.ndarray] or None: Trained parameter vectors for each dual variable,
                                      or None if model is infeasible.
    """
    if problem_type.upper() == "UFL":
        num_features = len(next(iter(feature_vectors.values())))
        model = Model("UFL_Train")
        model.Params.OutputFlag = 0

        # Parameters to learn
        w_lambda = model.addVars(num_features, name="w_lambda")
        w_mu = model.addVars(len(data.F), num_features, lb=0, name="w_mu")

        for i in solved_duals:
            phi = feature_vectors[i]  # Feature vector for subproblem i
            d_i = data.d[i]           # Demand

            # Predict lambda_i and mu_j for all j
            lambda_pred = quicksum(w_lambda[f] * phi[f] for f in range(num_features))
            mu_pred = {j: quicksum(w_mu[j, f] * phi[f] for f in range(num_features)) for j in data.F}

            # Dual feasibility constraint: lambda_i - mu_j <= c_ij * d_i for all j
            for j in data.F:
                model.addConstr(lambda_pred - mu_pred[j] <= data.c[i, j] * d_i,
                                name=f"dual_feasibility_{i}_{j}")

            # TODO: Add training loss constraints here (e.g., ||predicted - actual|| <= slack)

        model.optimize()

        if model.status != GRB.OPTIMAL:
            print("[Training] No feasible solution found for dual prediction model.")
            return None

        # Extract parameters
        w_lambda_vec = np.array([w_lambda[f].X for f in range(num_features)])
        w_mu_mat = {
            j: np.array([w_mu[j, f].X for f in range(num_features)]) for j in data.F
        }

        return {"lambda": w_lambda_vec, "mu": w_mu_mat}

    else:
        raise NotImplementedError(f"Training for {problem_type} not yet implemented.")
