from gurobipy import GRB, quicksum
from Problem.subproblem_dual import solve_dual_subproblem
from Dual_Prediction.Dual_prediction_trainer import train_feasible_predictor
import numpy as np

class Callback:
    def __init__(self, problem_type, data, first_stage_values, theta_vars,
                 selected_subproblems, feature_vectors,
                 prediction_method="REG",
                 n_neighbors=5,
                 use_prediction=True,
                 fallback_method='solve'):
        self.problem_type = problem_type
        self.data = data
        self.first_stage_values = first_stage_values
        self.theta = theta_vars #auxiliary variable for optimality cuts
        self.selected_subproblems = set(selected_subproblems) # Set of selected subproblems for which we solve the dual
        self.feature_vectors = feature_vectors # Feature vectors for each subproblem used in REG prediction
        self.prediction_method = prediction_method
        self.n_neighbors = n_neighbors
        self.use_prediction = use_prediction
        self.fallback_method = fallback_method.lower()
        self.trained_models = {}

        # Counters
        self.num_cuts_mip_selected = {s: 0 for s in data.S}
        self.num_cuts_rel_selected = {s: 0 for s in data.S}
        self.num_cuts_mip_ml = {s: 0 for s in data.S}
        self.num_cuts_rel_ml = {s: 0 for s in data.S}
        self.num_cuts_mip_unselected = {s: 0 for s in data.S}
        self.num_cuts_rel_unselected = {s: 0 for s in data.S}
        self.num_fallback_used = 0

        # Initialize KNN neighbors if using KNN prediction or fallback to avoid doing this on each iteration
        if self.use_prediction and (self.prediction_method.upper() == "KNN" or self.fallback_method.upper() == "KNN"):
            self.knn_neighbors = self.compute_knn_neighbors()


    def __call__(self, mod, where):
        if where != GRB.Callback.MIPSOL:
            return

        y_sol = mod.cbGetSolution(self.first_stage_values)
        theta_val = mod.cbGetSolution(self.theta)

        dual_cache = {}
        cuts_added = False

        # Step 1: Solve selected subproblems
        for s in self.selected_subproblems:
            result = solve_dual_subproblem(self.problem_type, self.data, s, y_sol)
            duals = self.structure_duals(result)
            dual_cache[s] = duals
            obj_val = self.compute_dual_obj(duals, s, y_sol)

            if obj_val > theta_val[s]:
                self.add_optimality_cut(mod, duals, s)
                self.num_cuts_mip_selected[s] += 1
                cuts_added = True

        unselected = set(self.data.S) - self.selected_subproblems

        # Step 2: Predict duals or fallback
        if self.use_prediction:
            if self.prediction_method.upper() == "REG":
                self.trained_models = train_feasible_predictor(self.problem_type, self.data, dual_cache, self.feature_vectors)

                if self.trained_models is None:
                    self.num_fallback_used += 1
                    self._handle_fallback(mod, unselected, y_sol, theta_val,dual_cache)
                else:
                    predicted = self.predict_duals(unselected)
                    for s in unselected:
                        duals = predicted[s]
                        pred_obj = self.compute_dual_obj(duals, s, y_sol)
                        if pred_obj > theta_val[s]:
                            self.add_optimality_cut(mod, duals, s)
                            self.num_cuts_mip_ml[s] += 1
                            cuts_added = True

            elif self.prediction_method.upper() == "KNN":
                predicted = self.knn_predict(unselected, dual_cache)
                for s in unselected:
                    duals = predicted[s]
                    pred_obj = self.compute_dual_obj(duals, s, y_sol)
                    if pred_obj > theta_val[s]:
                        self.add_optimality_cut(mod, duals, s)
                        self.num_cuts_mip_ml[s] += 1
                        cuts_added = True
        else:
            self._handle_fallback(mod, unselected, y_sol, theta_val,dual_cache)

    def _handle_fallback(self, mod, subproblem_set, y_sol, theta_val,dual_cache):
        if self.fallback_method == "solve":
            for s in subproblem_set:
                result = solve_dual_subproblem(self.problem_type, self.data, s, y_sol)
                duals = self.structure_duals(result)
                obj_val = self.compute_dual_obj(duals, s, y_sol)
                if obj_val > theta_val[s]:
                    self.add_optimality_cut(mod, duals, s)
                    self.num_cuts_mip_unselected[s] += 1

        elif self.fallback_method == "knn":
            predicted = self.knn_predict(subproblem_set, dual_cache)
            for s in subproblem_set:
                duals = predicted[s]
                pred_obj = self.compute_dual_obj(duals, s, y_sol)
                if pred_obj > theta_val[s]:
                    self.add_optimality_cut(mod, duals, s)
                    self.num_cuts_mip_ml[s] += 1

        else:
            raise ValueError(f"Unsupported fallback_method: {self.fallback_method}")
    
    def predict_duals(self, subproblems):
        if not self.trained_models:
            raise ValueError("Prediction model is not trained.")

        predicted = {}

        if self.problem_type.upper() == "UFL":
            for s in subproblems:
                phi = self.feature_vectors[s]
                lam_pred = float(np.dot(self.trained_models["lambda"], phi))
                mu_pred = {j: float(np.dot(self.trained_models["mu"][j], phi)) for j in self.data.F}
                predicted[s] = {"lambda": lam_pred, "mu": mu_pred}

        elif self.problem_type.upper() == "HUB":
            pass  # TODO: Add HUB prediction logic

        elif self.problem_type.upper() == "CMND":
            pass  # TODO: Add CMND prediction logic

        elif self.problem_type.upper() == "MCFL":
            pass  # TODO: Add MCFL prediction logic

        elif self.problem_type.upper() == "SSLP":
            pass  # TODO: Add SSLP prediction logic

        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")

        return predicted


    def compute_dual_obj(self, duals, sub_idx, y_sol):
        if self.problem_type.upper() == "UFL":
            lam = duals["lambda"]
            mu = duals["mu"]
            return lam - sum(mu[j] * y_sol[j] for j in self.data.F)

        elif self.problem_type.upper() == "HUB":
            pass  # TODO: Add cut for HUB
        elif self.problem_type.upper() == "CMND":
            pass  # TODO: Add cut for CMND
        elif self.problem_type.upper() == "MCFL":
            pass  # TODO: Add cut for MCFL
        elif self.problem_type.upper() == "SSLP":
            pass  # TODO: Add cut for SSLP
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")

    def add_optimality_cut(self, mod, duals, sub_idx):
        if self.problem_type.upper() == "UFL":
            lam = duals["lambda"]
            mu = duals["mu"]
            rhs = lam - quicksum(mu[j] * self.first_stage_values[j] for j in self.data.F)
            mod.cbLazy(self.theta[sub_idx] >= rhs)

        elif self.problem_type.upper() == "HUB":
            pass  # TODO: Add cut for HUB
        elif self.problem_type.upper() == "CMND":
            pass  # TODO: Add cut for CMND
        elif self.problem_type.upper() == "MCFL":
            pass  # TODO: Add cut for MCFL
        elif self.problem_type.upper() == "SSLP":
            pass  # TODO: Add cut for SSLP
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")
        

    def structure_duals(self, result):
        if self.problem_type.upper() == "UFL":
            obj_val, lambda_val, mu_vals = result
            return {"lambda": lambda_val, "mu": mu_vals}
        elif self.problem_type.upper() == "HUB":
            pass  # TODO: Define structure for HUB
        elif self.problem_type.upper() == "CMND":
            pass  # TODO: Define structure for CMND
        elif self.problem_type.upper() == "MCFL":
            pass  # TODO: Define structure for MCFL
        elif self.problem_type.upper() == "SSLP":
            pass  # TODO: Define structure for SSLP
        else:
            raise ValueError(f"Unsupported problem type: {self.problem_type}")
        


    def compute_knn_neighbors(self):
        unselected = set(self.data.S) - self.selected_subproblems
        selected = list(self.selected_subproblems)
        selected_features = {s: self.feature_vectors[s] for s in selected}

        knn_map = {}

        for u in unselected:
            phi_u = self.feature_vectors[u]
            distances = [
                (s, np.linalg.norm(phi_u - selected_features[s]))
                for s in selected
            ]
            distances.sort(key=lambda x: x[1])
            knn_map[u] = [s for s, _ in distances[:self.n_neighbors]]

        return knn_map
    

    def knn_predict(self, subproblems, dual_cache):
        predicted = {}

        if self.problem_type.upper() == "UFL":
            for s in subproblems:
                neighbors = self.knn_neighbors[s]

                # Extract lambda and mu from already-solved subproblems
                lam_vals = [dual_cache[n]["lambda"] for n in neighbors]
                mu_vals_list = [dual_cache[n]["mu"] for n in neighbors]

                # Average lambda
                lam_pred = np.mean(lam_vals)

                # Average mu component-wise
                mu_pred = {
                    j: np.mean([mu[j] for mu in mu_vals_list]) for j in self.data.F
                }

                predicted[s] = {"lambda": lam_pred, "mu": mu_pred}

        else:
            raise NotImplementedError(f"KNN prediction not implemented for {self.problem_type}")

        return predicted
