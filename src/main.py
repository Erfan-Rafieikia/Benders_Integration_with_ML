import os
import time
import json
import numpy as np
import pandas as pd
from .Problem.data import read_problem_data
from .Problem.master_problem import solve_master_problem
from .Problem.master_problem import Solution


from .config import (
    DATA_ROOT,
    RESULTS_ROOT,
    problem_classes_to_instances,
    prediction_method,
    use_prediction,
    n_neighbors,
    n_trials,
    FEATURE_PARAMS,
    n_selected_subproblems
)

solving_logs = []

# ========== STEP 1+2: Solve with embedded learning in Benders ==========
for problem_class, instance_list in problem_classes_to_instances.items():
    for instance_filename in instance_list:
        instance_path = os.path.join(DATA_ROOT, problem_class, instance_filename)
        print(f"\n🚀 Solving [{problem_class}] {instance_filename}...")

        # Load problem data
        print("Loading problem data for instance:", instance_filename)
        t0 = time.time()
        data = read_problem_data(problem_class, instance_path)
        t_data = time.time() - t0
        print(f"Data for instacen {instance_filename} loaded in {t_data:.2f} seconds.")

        # Solve using Benders with learning built-in
        print("solving instance ", instance_filename)
        t0 = time.time()
        solution = solve_master_problem(
            problem_type=problem_class,
            data=data,
            selected_subproblems=[],  # Empty initially; handled inside callback
            feature_vectors={},       # Empty initially; handled inside callback
            prediction_method=prediction_method,
            n_neighbors=n_neighbors,
            use_prediction=use_prediction,
            n_trials=n_trials,
            feature_params=FEATURE_PARAMS,
            n_selected_subproblems=n_selected_subproblems
        )
        t_sol = time.time() - t0
        print(f"Instance {instance_filename} solved in {t_sol:.2f} seconds.")

        # Logging
        if solution is not None:
            solving_logs.append({
                "problem_class": problem_class,
                "instance_filename": instance_filename,
                "objective_value": round(solution.objective_value, 4),
                "solution_time_sec": solution.solution_time,
                "feature_gen_time_sec": None,
                "subproblem_selection_time_sec": None,
                "bnb_nodes": solution.num_bnb_nodes,
                "cuts_mip_selected": sum(solution.num_cuts_mip_selected.values()),
                "cuts_rel_selected": sum(solution.num_cuts_rel_selected.values()),
                "cuts_mip_ml": sum(solution.num_cuts_mip_ml.values()),
                "cuts_rel_ml": sum(solution.num_cuts_rel_ml.values()),
                "cuts_mip_unselected": sum(solution.num_cuts_mip_unselected.values()),
                "cuts_rel_unselected": sum(solution.num_cuts_rel_unselected.values())
            })
        else:
            solving_logs.append({
                "problem_class": problem_class,
                "instance_filename": instance_filename,
                "objective_value": None,
                "solution_time_sec": None,
                "feature_gen_time_sec": None,
                "subproblem_selection_time_sec": None,
                "bnb_nodes": None,
                "cuts_mip_selected": None,
                "cuts_rel_selected": None,
                "cuts_mip_ml": None,
                "cuts_rel_ml": None,
                "cuts_mip_unselected": None,
                "cuts_rel_unselected": None
            })

# Save logs
solving_log_path = os.path.join(RESULTS_ROOT, "solution_log.xlsx")
pd.DataFrame(solving_logs).to_excel(solving_log_path, index=False)
print(f"\nSTEP 2 done: Final results saved to {solving_log_path}")
