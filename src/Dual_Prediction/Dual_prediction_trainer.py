from gurobipy import Model, GRB, quicksum
import numpy as np


from gurobipy import Model, GRB, quicksum
import numpy as np

def train_feasible_predictor(problem_type, data, solved_duals, feature_vectors):
    """
    Trains a linear prediction model for dual variables using feature vectors.
    The structure of dual variables and feasibility constraints depends on the problem type.

    Args:
        problem_type (str): Problem class (e.g., "UFL", "HUB", etc.)
        data: Problem data (e.g., UFLData)
        solved_duals: dict[sub_idx] -> dict of dual variables (e.g., {"lambda": val, "mu": {j: val}})
        feature_vectors: dict[sub_idx] -> np.array of features

    Returns:
        dict or None: Trained model parameters (e.g., {"lambda": w_lambda_vec, "mu": {j: w_mu_vec}})
                      or None if no feasible solution is found
    """

    problem_type = problem_type.upper()

    if problem_type == "UFL":
        # Sanity check
        for i, dual in solved_duals.items():
            assert "lambda" in dual and "mu" in dual, f"Missing keys in solved_duals[{i}]"
            assert isinstance(dual["mu"], dict)

        num_features = len(next(iter(feature_vectors.values())))
        model = Model("UFL_Train")
        model.Params.OutputFlag = 0

        # Define learnable parameters
        w_lambda = model.addVars(num_features, name="w_lambda")
        w_mu = model.addVars(len(data.F), num_features, lb=0, name="w_mu")

        for i in solved_duals:
            phi = feature_vectors[i]
            d_i = data.d[i]

            lambda_pred = quicksum(w_lambda[f] * phi[f] for f in range(num_features))
            mu_pred = {
                j: quicksum(w_mu[j, f] * phi[f] for f in range(num_features)) for j in data.F
            }

            # Dual feasibility: λ - μ_j ≤ c_ij * d_i
            for j in data.F:
                model.addConstr(lambda_pred - mu_pred[j] <= data.c[i, j] * d_i,
                                name=f"dual_feasibility_{i}_{j}")

        model.optimize()

        if model.status != GRB.OPTIMAL:
            print("[Training] No feasible solution found for UFL dual prediction model.")
            return None

        # Extract learned parameters
        w_lambda_vec = np.array([w_lambda[f].X for f in range(num_features)])
        w_mu_mat = {
            j: np.array([w_mu[j, f].X for f in range(num_features)]) for j in data.F
        }

        return {"lambda": w_lambda_vec, "mu": w_mu_mat}

    elif problem_type == "HUB":
        # TODO: Define structure of solved_duals and constraints for HUB
        raise NotImplementedError("Dual training not yet implemented for HUB.")

    elif problem_type == "CMND":
        # TODO: Define structure of solved_duals and constraints for CMND
        raise NotImplementedError("Dual training not yet implemented for CMND.")

    elif problem_type == "MCFL":
        # TODO: Define structure of solved_duals and constraints for MCFL
        raise NotImplementedError("Dual training not yet implemented for MCFL.")

    elif problem_type == "SSLP":
        # TODO: Define structure of solved_duals and constraints for SSLP
        raise NotImplementedError("Dual training not yet implemented for SSLP.")

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

