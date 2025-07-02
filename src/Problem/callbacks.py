from gurobipy import GRB, quicksum
from Problem.subproblem_dual import solve_dual_subproblem
from Dual_Prediction.Dual_prediction_trainer import train_feasible_predictor
from config import *
import numpy as np

class Callback:
    def __init__(self, problem_type, data, first_stage_values, theta_vars, selected_subproblems, feature_vectors,
                 prediction_method=PREDICTION_METHOD, n_neighbors=N_NEIGHBORS, use_prediction=USE_PREDICTION):
        self.problem_type = problem_type
        self.data = data
        self.first_stage_values = first_stage_values
        self.theta = theta_vars
        self.selected_subproblems = set(selected_subproblems)
        self.feature_vectors = feature_vectors
        self.prediction_method = prediction_method
        self.n_neighbors = n_neighbors
        self.use_prediction = use_prediction

        self.trained_models = {}  # Keyed by dual variable name (e.g., "lambda", "mu")
        self.num_cuts_mip_selected = {s: 0 for s in data.S}
        self.num_cuts_rel_selected = {s: 0 for s in data.S}
        self.num_cuts_mip_ml = {s: 0 for s in data.S}
        self.num_cuts_rel_ml = {s: 0 for s in data.S}
        self.num_cuts_mip_unselected = {s: 0 for s in data.S}
        self.num_cuts_rel_unselected = {s: 0 for s in data.S}

    def __call__(self, mod, where):
        if where == GRB.Callback.MIPSOL:
            first_stage_values = mod.cbGetSolution(self.first_stage_values)
            theta_val = mod.cbGetSolution(self.theta)
            dual_cache = {}
            cuts_added_selected = False
            cuts_added_ml = False
            cuts_added_unselected = False

            # Step 1: Solve selected subproblems
            for s in self.selected_subproblems:
                obj, duals = solve_dual_subproblem(self.problem_type, self.data, s, first_stage_values)
                dual_cache[s] = duals
                if obj > theta_val[s]:
                    self.add_optimality_cut(mod, duals, s, first_stage_values)
                    self.num_cuts_mip_selected[s] += 1
                    cuts_added_selected = True

            # Step 2: Train prediction model and apply if enabled
            if self.use_prediction:
                self.trained_models = train_feasible_predictor(self.problem_type, self.data, dual_cache, self.feature_vectors)

                unselected = set(self.data.S) - self.selected_subproblems

                if self.trained_models is not None:
                    predicted = self.predict_duals(unselected)
                    for s in unselected:
                        duals = predicted[s]
                        pred_obj = self.compute_dual_obj(duals, s, first_stage_values)
                        if pred_obj > theta_val[s]:
                            self.add_optimality_cut(mod, duals, s, first_stage_values)
                            self.num_cuts_mip_ml[s] += 1
                            cuts_added_ml = True
                else:
                    # fallback: directly solve subproblems if training failed
                    for s in unselected:
                        obj, duals = solve_dual_subproblem(self.problem_type, self.data, s, first_stage_values)
                        if obj > theta_val[s]:
                            self.add_optimality_cut(mod, duals, s, first_stage_values)
                            self.num_cuts_mip_unselected[s] += 1
                            cuts_added_unselected = True
            else:
                for s in set(self.data.S) - self.selected_subproblems:
                    obj, duals = solve_dual_subproblem(self.problem_type, self.data, s, first_stage_values)
                    if obj > theta_val[s]:
                        self.add_optimality_cut(mod, duals, s, first_stage_values)
                        self.num_cuts_mip_unselected[s] += 1
                        cuts_added_unselected = True

            # Optional: fallback if no cuts were added
            if SOLVE_UNSELECTED_IF_NO_CUTS and self.use_prediction and not cuts_added_selected and not cuts_added_ml:
                for s in set(self.data.S) - self.selected_subproblems:
                    obj, duals = solve_dual_subproblem(self.problem_type, self.data, s, first_stage_values)
                    if obj > theta_val[s]:
                        self.add_optimality_cut(mod, duals, s, first_stage_values)
                        self.num_cuts_mip_unselected[s] += 1
                        break

    def predict_duals(self, unselected_subproblems):
        if not self.trained_models:
            raise ValueError("Dual models not trained.")

        predicted = {}
        for s in unselected_subproblems:
            X = np.array([self.feature_vectors[s]])
            predicted[s] = {
                key: (np.dot(weight_vec, X.flatten()) if key == "lambda" else {j: np.dot(w_j, X.flatten()) for j, w_j in weight_vec.items()})
                for key, weight_vec in self.trained_models.items()
            }
        return predicted

    def compute_dual_obj(self, duals, sub_idx, first_stage_values):
        if self.problem_type.upper() == "UFL":
            mu = duals["mu"]
            lam = duals["lambda"]
            return lam - sum(mu[j] * first_stage_values[j] for j in self.data.F)
        else:
            pass  # TODO: Add other cases

    def add_optimality_cut(self, mod, duals, sub_idx, first_stage_values):
        if self.problem_type.upper() == "UFL":
            mu = duals["mu"]
            lam = duals["lambda"]
            rhs = lam - quicksum(mu[j] * self.first_stage_values[j] for j in self.data.F)
            mod.cbLazy(self.theta[sub_idx] >= rhs)
        else:
            pass  # TODO: Add other cases
