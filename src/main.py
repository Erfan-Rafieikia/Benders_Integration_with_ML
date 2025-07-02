import os
import pickle
import time
from data import read_problem_data
from feature_engineering.generate_first_stage_trial import generate_scenario_features
from Problem.master_problem import solve_master_problem


"""
This script executes a two-step workflow:

1. **Feature Generation**:
   For each problem class (UFL, HUB, CMND, MCFL, SSLP) and corresponding instances,
   it generates scenario-based feature vectors using dual sensitivity analysis and
   random walk embeddings. These features are saved to disk for later use. Execution time
   for feature generation is recorded for performance analysis.

2. **Solving with Variant Benders Decomposition**:
   It then loads the previously generated features and solves each instance using
   a variant of Benders decomposition enhanced with machine learning-based predictions.

This script uses configured parameters such as number of trials, embedding dimensions,
and prediction methods to control the feature generation and solution processes.
"""

# Configuration parameters
DATA_ROOT = r"G:\My Drive\Programming\Research Projects\Benders_Integration_with_ML\DATA"
FEATURES_ROOT = r"G:\My Drive\Programming\Research Projects\Benders_Integration_with_ML\FEATURES"
PROBLEM_CLASSES = ['UFL'] # List of problem classes to process (e.g., ['UFL', 'HUB', 'CMND', 'MCFL', 'SSLP'])

# Feature generation parameters
n_trials = 30                 # Number of first-stage decision trials
binary_fraction = 0.3         # Fraction of first-stage variables treated as binary
seed = 42                     # Random seed for reproducibility
n_walks = 20                  # Number of random walks for embeddings
walk_length = 10              # Length of each random walk
feature_dim = 8               # Dimension of embedding features
window = 5                    # Window size for Word2Vec
min_count = 1                 # Minimum count for Word2Vec
sg = 1                        # Skip-gram (1) or CBOW (0) for Word2Vec

# Prediction method parameters
prediction_method = 'KNN'     # Method used for prediction (e.g., 'KNN')
n_neighbors = 5               # Number of neighbors for KNN prediction
use_prediction = True         # Whether to use ML predictions in Benders decomposition
selected_subproblems_fraction = 0.2  # Fraction of subproblems initially selected for training

# Ensure features directory exists
os.makedirs(FEATURES_ROOT, exist_ok=True)

# Step 1: Feature Generation
for problem_class in PROBLEM_CLASSES:
    problem_data_path = os.path.join(DATA_ROOT, problem_class, 'O')
    if not os.path.exists(problem_data_path):
        print(f"Data path for problem class {problem_class} does not exist, skipping.") 
        continue

    # Iterate over each instance file
    for instance_filename in os.listdir(problem_data_path):
        instance_path = os.path.join(problem_data_path, instance_filename)

        # Read data from file based on problem class
        data = read_problem_data(problem_class, instance_path)

        # Generate scenario features using dual sensitivity and random walk embeddings
        print(f"Generating features for {problem_class} instance {instance_filename}")
        start_time = time.time()
        feature_vectors = generate_scenario_features(
            data, problem_class, n_trials, binary_fraction, seed,
            n_walks, walk_length, feature_dim, window, min_count, sg
        )
        feature_gen_time = time.time() - start_time
        print(f"Feature generation took {feature_gen_time:.2f} seconds")

        # Save generated feature vectors and generation time
        feature_file_path = os.path.join(FEATURES_ROOT, f"features_{problem_class}_{instance_filename}.pkl")
        with open(feature_file_path, "wb") as f:
            pickle.dump({"features": feature_vectors, "generation_time": feature_gen_time}, f)

# Step 2: Solving with Variant Benders Decomposition
for problem_class in PROBLEM_CLASSES:
    problem_data_path = os.path.join(DATA_ROOT, problem_class, 'O')
    if not os.path.exists(problem_data_path):
        continue

    # Iterate over each instance file again to solve using Benders decomposition
    for instance_filename in os.listdir(problem_data_path):
        instance_path = os.path.join(problem_data_path, instance_filename)

        # Load problem data
        data = read_problem_data(problem_class, instance_path)

        # Load previously generated features and feature generation time
        feature_file_path = os.path.join(FEATURES_ROOT, f"features_{problem_class}_{instance_filename}.pkl")
        if not os.path.exists(feature_file_path):
            print(f"Features file not found for {instance_filename}, skipping solving.")
            continue

        with open(feature_file_path, "rb") as f:
            loaded_data = pickle.load(f)
            feature_vectors = loaded_data["features"]
            feature_gen_time = loaded_data["generation_time"]

        # Select initial subset of subproblems for model training
        num_selected = max(1, int(len(data.S) * selected_subproblems_fraction))
        selected_subproblems = list(data.S[:num_selected])

        # Solve the instance using Benders decomposition variant
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

        # Output solution details and feature generation time
        print(f"Instance: {instance_filename}, Objective: {solution.objective_value}, Solution Time: {solution.solution_time}s, Feature Generation Time: {feature_gen_time:.2f}s")