import os
import time
import json
import numpy as np
import pandas as pd
from .Problem.data import read_problem_data
from .feature_engineering.generate_first_stage_trial import generate_scenario_features
from .subproblem_selection.subproblem_selection import select_subproblems_based_on_features
from .Problem.master_problem import solve_master_problem
from .Problem.master_problem import Solution


from .config import (
    DATA_ROOT, FEATURES_ROOT, RESULTS_ROOT,
    FEATURE_PARAMS, problem_classes_to_instances,
    n_trials, binary_fraction,
    subproblem_selection_method, n_selected_subproblems,prediction_method, n_neighbors, use_prediction
)

# ========== Prepare Output Directories ==========
os.makedirs(FEATURES_ROOT, exist_ok=True)
os.makedirs(RESULTS_ROOT, exist_ok=True)
json_log_dir = os.path.join(FEATURES_ROOT, "detailed_logs")
os.makedirs(json_log_dir, exist_ok=True)

# ========== Logs ==========
feature_logs = []
solving_logs = []

# ========== STEP 1: Feature Generation + Subproblem Selection ==========
for problem_class, instance_list in problem_classes_to_instances.items():
    for instance_filename in instance_list:
        instance_path = os.path.join(DATA_ROOT, problem_class, instance_filename)
        print(f"\n🚀 STEP 1: Generating features for [{problem_class}] {instance_filename}...")

        # Read data
        data = read_problem_data(problem_class, instance_path)

        # Feature generation
        t0 = time.time()
        result = generate_scenario_features(
            data=data,
            problem_type=problem_class,
            n_trials=n_trials,
            binary_fraction=binary_fraction,
            **FEATURE_PARAMS
        )
        t_feat = time.time() - t0

        features = result["features"]
        x_trials = result["x_trials"]
        params = result["params"]

        print("Feature generation for instance [{}] and problem class [{}] completed in {:.2f} seconds.".format(
            instance_filename, problem_class, t_feat))

        # Subproblem selection
        t1 = time.time()
        selected, assignment = select_subproblems_based_on_features(
            feature_vectors=features,
            method=subproblem_selection_method,
            p=n_selected_subproblems
        )
        if (selected is None) or (not hasattr(selected, '__iter__')):
            selected = []
        if (assignment is None) or (not hasattr(assignment, '__iter__')):
            assignment = []
        t_sel = time.time() - t1

        print("Subproblem selection for instance [{}] and problem class [{}] completed in {:.2f} seconds.".format(
            instance_filename, problem_class, t_sel))

        # Save everything to a JSON file
        json_filename = f"{problem_class}_{instance_filename}_log.json"
        json_path = os.path.join(json_log_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump({
                "problem_class": problem_class,
                "instance_filename": instance_filename,
                "x_trials": x_trials.tolist(),
                "features": {str(k): v.tolist() for k, v in features.items()},
                "feature_generation_params": params,
                "feature_gen_time_sec": round(t_feat, 2),
                "subproblem_selection_method": subproblem_selection_method,
                "n_selected_subproblems": n_selected_subproblems,
                "subproblem_selection_time_sec": round(t_sel, 2),
                "selected_subproblems": selected,
                "subproblem_assignment": assignment
            }, f, indent=2)

        # Log feature summary
        feature_logs.append({
            "problem_class": problem_class,
            "instance_filename": instance_filename,
            "feature_gen_time_sec": round(t_feat, 2),
            "subproblem_selection_time_sec": round(t_sel, 2),
            "n_trials": n_trials,
            "binary_fraction": binary_fraction,
            **params,
            "n_selected_subproblems": n_selected_subproblems,
            "selected_subproblems": selected
        })

# Save summary log for STEP 1
feature_log_path = os.path.join(RESULTS_ROOT, "feature_subproblem_log.xlsx")
pd.DataFrame(feature_logs).to_excel(feature_log_path, index=False)
print(f"\n STEP 1 done: Feature logs saved to {feature_log_path}")
print(f" JSON logs saved to: {json_log_dir}")

# ========== STEP 2: Solving ==========
for filename in os.listdir(json_log_dir):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(json_log_dir, filename)
    with open(json_path, "r") as f:
        log = json.load(f)

    problem_class = log["problem_class"]
    instance_filename = log["instance_filename"]
    instance_path = os.path.join(DATA_ROOT, problem_class, instance_filename)

    print(f"\n STEP 2: Solving [{problem_class}] {instance_filename}...")

    #x_trials = np.array(log["x_trials"])
    features = {int(k): np.array(v) for k, v in log["features"].items()}
    selected = log["selected_subproblems"]
    #assignment = log["subproblem_assignment"]

    # Load problem data again
    data = read_problem_data(problem_class, instance_path)

    # Solve
    t0 = time.time()
    solution = solve_master_problem(
        problem_type=problem_class,
        data=data,
        selected_subproblems=selected,
        feature_vectors=features,
        prediction_method=prediction_method,
        n_neighbors=n_neighbors,
        use_prediction=use_prediction
)
    t_sol = time.time() - t0

    if solution is not None:
        solving_logs.append({
            "problem_class": problem_class,
            "instance_filename": instance_filename,
            "objective_value": round(solution.objective_value, 4),
            "solution_time_sec": solution.solution_time,
            "feature_gen_time_sec": log["feature_gen_time_sec"],
            "subproblem_selection_time_sec": log["subproblem_selection_time_sec"],
            "bnb_nodes": solution.num_bnb_nodes,

            # Cuts - MIP Selected
            "cuts_mip_selected": sum(solution.num_cuts_mip_selected.values()),
            # Cuts - Relaxed Selected
            "cuts_rel_selected": sum(solution.num_cuts_rel_selected.values()),
            # Cuts - MIP ML
            "cuts_mip_ml": sum(solution.num_cuts_mip_ml.values()),
            # Cuts - Relaxed ML
            "cuts_rel_ml": sum(solution.num_cuts_rel_ml.values()),
            # Cuts - MIP Unselected
            "cuts_mip_unselected": sum(solution.num_cuts_mip_unselected.values()),
            # Cuts - Relaxed Unselected
            "cuts_rel_unselected": sum(solution.num_cuts_rel_unselected.values())
        })
    else:
        solving_logs.append({
            "problem_class": problem_class,
            "instance_filename": instance_filename,
            "objective_value": None,
            "solution_time_sec": None,
            "feature_gen_time_sec": log["feature_gen_time_sec"],
            "subproblem_selection_time_sec": log["subproblem_selection_time_sec"],
            "bnb_nodes": None,
            "cuts_mip_selected": None,
            "cuts_rel_selected": None,
            "cuts_mip_ml": None,
            "cuts_rel_ml": None,
            "cuts_mip_unselected": None,
            "cuts_rel_unselected": None
        })


# Save final solving log
solving_log_path = os.path.join(RESULTS_ROOT, "solution_log.xlsx")
pd.DataFrame(solving_logs).to_excel(solving_log_path, index=False)
print(f"\n STEP 2 done: Final results saved to {solving_log_path}")
