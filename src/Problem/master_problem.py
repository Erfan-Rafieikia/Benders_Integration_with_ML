from dataclasses import dataclass
from .callbacks import Callback
from .data import UFLData, HUBData, CMNDData, MCFLData, SSLPData
from gurobipy import GRB, Model, quicksum
import numpy as np

WRITE_MP_LP = False  # Set to True if you want to write the master problem LP file

@dataclass
class Solution:
    objective_value: float
    locations: list
    solution_time: float
    num_cuts_mip_selected: dict
    num_cuts_rel_selected: dict
    num_cuts_mip_ml: dict
    num_cuts_rel_ml: dict
    num_cuts_mip_unselected: dict
    num_cuts_rel_unselected: dict
    num_bnb_nodes: int = 0

def _set_params(mod: Model): # sets Gurobi solver parameters for the optimization model. 
    mod.Params.LazyConstraints = 1 # Enable lazy constraints. These constraints are not added to the model up front, but are instead added dynamically during the branch-and-bound process (typically via a callback).
    # mod.Params.TimeLimit = 60.0

def solve_master_problem(problem_type, data, selected_subproblems, feature_vectors,
                         n_neighbors,
                         prediction_method='KNN',
                         use_prediction=True):
    if problem_type.upper() == "UFL":
        with Model("UFL_Master") as mod:
            _set_params(mod)

            first_stage_values = mod.addVars(data.F, vtype=GRB.BINARY, name="y")
            theta_vars = mod.addVars(data.S, name="theta")

            total_cost = quicksum(data.f[j] * first_stage_values[j] for j in data.F) + \
                         quicksum(theta_vars[s] for s in data.S)
            mod.setObjective(total_cost, GRB.MINIMIZE)

            #mod.addConstr(
            #    quicksum(data.u[j] * first_stage_values[j] for j in data.F) >= data.total_demand,
            #    name="Feasibility"
            #)  # This constraint ensures that the total demand served by the facilities meets the overall demand. This is not required in uncapcitated UFL which we study given that only in that case, it can decomposed by customers

            callback = Callback(problem_type=problem_type,
                                data=data,
                                first_stage_values=first_stage_values,
                                theta_vars=theta_vars,
                                selected_subproblems=selected_subproblems,
                                feature_vectors=feature_vectors,
                                prediction_method=prediction_method,
                                n_neighbors=n_neighbors,
                                use_prediction=use_prediction)

            if WRITE_MP_LP: # Write the master problem to a .lp file if True
                mod.write(f"{mod.ModelName}.lp")

            mod.optimize(callback)

            obj = mod.ObjVal
            sol_time = round(mod.Runtime, 2)
            y_values = mod.getAttr("x", first_stage_values)

            return Solution(
                objective_value=obj,
                locations=y_values,
                solution_time=sol_time,
                num_cuts_mip_selected=callback.num_cuts_mip_selected,
                num_cuts_rel_selected=callback.num_cuts_rel_selected,
                num_cuts_mip_ml=callback.num_cuts_mip_ml,
                num_cuts_rel_ml=callback.num_cuts_rel_ml,
                num_cuts_mip_unselected=callback.num_cuts_mip_unselected,
                num_cuts_rel_unselected=callback.num_cuts_rel_unselected,
                num_bnb_nodes=int(mod.NodeCount)
            )

    elif problem_type.upper() == "HUB":
        pass  # TODO

    elif problem_type.upper() == "CMND":
        pass  # TODO

    elif problem_type.upper() == "MCFL":
        pass  # TODO

    elif problem_type.upper() == "SSLP":
        pass  # TODO

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
