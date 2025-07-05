import os

# ========== Root Directories ==========
DATA_ROOT = r"G:\My Drive\Programming\Research Projects\Benders_Integration_with_ML\DATA"
FEATURES_ROOT = r"G:\My Drive\Programming\Research Projects\Benders_Integration_with_ML\FEATURES"
RESULTS_ROOT = r"G:\My Drive\Programming\Research Projects\Benders_Integration_with_ML\RESULTS"

# ========== Problem Classes and Instances ==========
problem_classes_to_instances = {
    "UFL": ["MO0", "MO1"],
    "HUB": [],
    "CMND": [],
    "MCFL": [],
    "SSLP": []
}

# ========== Feature Generation Parameters ==========
n_trials = 5              # Number of first-stage trial vectors
binary_fraction = 0      # Fraction of binary trials

FEATURE_PARAMS = {
    "n_walks": 5,         # Number of random walks
    "walk_length": 5,     # Length of each random walk
    "feature_dim": 3,      # Embedding dimension (Word2Vec)
    "window": 5,           # Window size for Word2Vec
    "min_count": 1,        # Minimum word frequency
    "sg": 1                # Use Skip-Gram (1) or CBOW (0)
}

# ========== Subproblem Selection Parameters ==========
subproblem_selection_method = "p-median"  # Method: "p-median" (default)
n_selected_subproblems = 5                # Number of subproblems to sample (p)

# ========== Solver Parameters (Optional for Expansion) ==========
prediction_method = 'REG'
n_neighbors = 5 #Number of neighbors for KNN if KNN method use for ML-augmented BD
use_prediction = True # If True, implemets Ml-augmented BD, if False, implements classical BD
fallback_method = "solve" # "solve"  or "KNN" . iF "solve", fallback to solving the subproblem, if "KNN", fallback to KNN prediction when REG prediction fails
