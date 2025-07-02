import os
import pickle
import time
from feature_engineering.generate_first_stage_trial import generate_scenario_features
from Problem.master_problem import solve_master_problem
from Problem.data import read_problem_data

"""
This script executes a three-step workflow:

1. **Feature Generation**:
   For specified problem classes and selected instance names,
   it generates scenario-based feature vectors using dual sensitivity analysis and
   random walk embeddings. These features are saved to disk for later use. Execution time
   for feature generation is recorded for performance analysis.

2. **Subproblem Selection** (placeholder):
   Uses feature vectors to select a subset of subproblems/scenarios to train the dual prediction model.
   This selection logic will be implemented via a separate function later.

3. **Solving with Variant Benders Decomposition**:
   Loads the features and selected subproblems to solve each instance using
   Benders decomposition enhanced with ML-based dual predictions.
"""

# Configuration parameters
DATA_ROOT = r"G:\My Drive\Programming\Research Projects\Benders_Integration_with_ML\DATA"
FEATURES_ROOT = r"G:\My Drive\Programming\Research Projects\Benders_Integration_with_ML\FEATURES"

# Specify problem classes and their respective instance files
PROBLEM_INSTANCES = {
    "UFL": ["MO1", "MO2"],
    "HUB": [],
    "CMND": [],
    "MCFL": [],
    "SSLP": []
}

# Feature generation parameters
n_trials = 30
binary_fraction = 0.3
seed = 42
n_walks = 20
walk_length = 10
feature_dim = 8
window = 5
min_count = 1
sg = 1

# Prediction method parameters
prediction_method = 'KNN'
n_neighbors = 5
use_prediction = True

# Ensure features directory exists
os.makedirs(FEATURES_ROOT, exist_ok=True) #create the FEATURES_ROOT directory if it does not exist. Parent directories will be created if they do not exist. If the directory already exists, it will not raise an error.

# Step 1: Feature Generation
for problem_class, instance_list in PROBLEM_INSTANCES.items(): # Iterate over each problem class and its instances
    for instance_filename in instance_list: # Iterate over each instance filename in the list. instance_list is a list of instance filenames for the current problem class
        instance_path = os.path.join(DATA_ROOT, problem_class, instance_filename) # Construct the full path to the instance file
        if not os.path.exists(instance_path):
            print(f"Instance {instance_filename} for {problem_class} not found, skipping.")
            continue # If the instance file does not exist, skip to the next iteration

        # Read data
        data = read_problem_data(problem_class, instance_path)# Read the problem data from the specified instance file using the read_problem_data function defined in data.py using the created path instance_path. Takes the problem class as argument since reading files depedens on the problem class. Returns an instance of the problem class with the read data.

        # Generate scenario features
        print(f"Generating features for {problem_class} instance {instance_filename}")
        start_time = time.time() # Start timer for feature generation for the current instance
        feature_vectors = generate_scenario_features(
            data, problem_class, n_trials, binary_fraction, seed,
            n_walks, walk_length, feature_dim, window, min_count, sg
        ) 
        feature_gen_time = time.time() - start_time # Calculate the time taken for feature generation for the current instance
        print(f"Feature generation took {feature_gen_time:.2f} seconds")

        # Save features
        feature_file_path = os.path.join(FEATURES_ROOT, f"features_{problem_class}_{instance_filename}.pkl")
        with open(feature_file_path, "wb") as f:
            pickle.dump({"features": feature_vectors, "generation_time": feature_gen_time}, f)

# Step 2: Placeholder for Subproblem Selection (to be implemented)
def select_subproblems(feature_vectors, data):
    """
    Placeholder for scenario selection logic.
    Should return a list of selected scenario indices.
    """
    selected_fraction = 0.2
    num_selected = max(1, int(len(data.S) * selected_fraction))
    return list(data.S[:num_selected])  # Dummy selection: first few scenarios

# Step 3: Solve Instances
for problem_class, instance_list in PROBLEM_INSTANCES.items():
    for instance_filename in instance_list:
        instance_path = os.path.join(DATA_ROOT, problem_class, 'O', instance_filename)
        feature_file_path = os.path.join(FEATURES_ROOT, f"features_{problem_class}_{instance_filename}.pkl")

        if not os.path.exists(instance_path):
            print(f"Instance file {instance_filename} not found, skipping.")
            continue
        if not os.path.exists(feature_file_path):
            print(f"Features file for {instance_filename} not found, skipping solving.")
            continue

        # Load problem data and features
        data = read_problem_data(problem_class, instance_path)
        with open(feature_file_path, "rb") as f:
            loaded_data = pickle.load(f)
            feature_vectors = loaded_data["features"]
            feature_gen_time = loaded_data["generation_time"]

        # Step 2: Select subproblems
        selected_subproblems = select_subproblems(feature_vectors, data)

        # Solve with Benders
        print(f"Solving {problem_class} instance {instance_filename}")
        solution = solve_master_problem(
            problem_class,
            data,
            selected_subproblems,
            feature_vectors,
            prediction_method=prediction_method,
            n_neighbors=n_neighbors,
            use_prediction=use_prediction
        )

        print(f"Instance: {instance_filename}, Objective: {solution.objective_value}, "
              f"Solution Time: {solution.solution_time}s, Feature Generation Time: {feature_gen_time:.2f}s")
