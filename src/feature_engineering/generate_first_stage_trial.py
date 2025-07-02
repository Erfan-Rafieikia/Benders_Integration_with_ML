import numpy as np
from itertools import product
from gensim.models import Word2Vec
from ..Problem.subproblem_dual import solve_dual_subproblem
import numpy as np

def generate_first_stage_trials(n_trials, data, binary_fraction, seed, problem_type):
    """
    Generate feasible first-stage solutions satisfying problem-specific constraints. For UFL, this means generating binary vectors (which facilities are opened) that satisfy the total demand constraint.

    Args:
        n_trials (int): Number of first-stage samples to generate. We create (n_trials × size of first stage variable ) values 
        data: Data object containing problem parameters and index sets.
        binary_fraction (float): Fraction of trials that are fully binary. This comes handy where we allow geenration of cuts at fractional solution as well in B&C procedure.
        seed (int): Random seed for reproducibility.
        problem_type (str): One of {"UFL", "HUB", "CMND", "MCFL", "SSLP"}

    Returns:
        np.ndarray: Matrix of feasible first-stage solutions (n_trials × |F|).
    """
    np.random.seed(seed)
    x_trials = [] # List to store generated first-stage solutions

    n_binary_trials = int(binary_fraction * n_trials) # Number of trials that will be binary vectors
    n_continuous_trials = n_trials - n_binary_trials # Number of trials that will be continuous vectors in [0, 1]
    total_trials = 0 # Total number of trials generated so far

    # Determine first-stage variable indices
    if problem_type.upper() == "UFL":
        F = data.F  # First-stage variable indices for UFL are the facilities
        num_vars = len(F) # Number of first-stage variables (facilities)

    else:
        raise NotImplementedError(f"Problem type {problem_type} not yet implemented.")

    while len(x_trials) < n_trials:
        if total_trials < n_binary_trials:
            # Binary vector
            x = np.random.choice([0, 1], size=num_vars)
        else:
            # Continuous values in [0, 1]
            x = np.random.uniform(0, 1, size=num_vars)

        # Feasibility check
        if problem_type.upper() == "UFL":
            pass # For UFL, we don't havce any constraints on the first-stage variables, so all generated vectors are feasible.
            #total_capacity = sum(data.capacities[j] * x[j] for j in range(num_vars))
            #if total_capacity >= data.total_demand - 1e-6:
            #   x_trials.append(x)

        # Add more problem classes as needed
        total_trials += 1  # Avoid infinite loop in case of tight feasibility

    return np.array(x_trials) # Matrix of shape (n_trials, size of first-stage decision vector) where size of first-stage decision vector is |data.F| is the number of facilities in UFL.


def get_dual_vectors(x_trials: np.ndarray, data, problem_type: str):
    """
    For each trial vector (first-stage decision values), solves all the subproblems (customers in UFL) to get the optimal dual vector(lambda and mu values in UFL) for all subproblems (customers in UFL).
    Then return output that includes all optimal dual vectors for all trials and all subproblems.
    This is used to compute the average dual distances between all pairs of scenarios/subproblems (customers iN UFL). This step is not done inside this function. 
    Args:
        x_trials (np.ndarray): Matrix (n_trials × |size of first-satge vector|) where |size of first-satge vector| for UFL is the number of facilities and equal to |data.F|
        data: data object for storing all the data needed to describe an a problem instance 
        problem_type (str): "UFL", "HUB", "CMND", "MCFL", "SSLP".

    Returns:
        Should be a data type that includes trial_idx, dual subproblem index (customer index in UFL), and optimal dual vector for that dual subproblem (customer) under that trial vector which is optimal lambda and mu values in UFL.
        Dict[(trial_idx, customer_idx)] → dual vector (lambda followed by mu_j's in UFL)
    """
    duals = {}

    if problem_type.upper() == "UFL":
        for trial_idx, x in enumerate(x_trials):
            for subproblem_dual_idx in data.S:
                _, lambda_val, mu_dict = solve_dual_subproblem(
                    problem_type=problem_type,
                    data=data,
                    subproblem_idx=subproblem_dual_idx,
                    first_stage_values=x
                )
                mu_vec = np.array([mu_dict[j] for j in data.F])
                dual_vec = np.concatenate(([lambda_val], mu_vec))  # Include lambda at the front
                duals[(trial_idx, subproblem_dual_idx)] = dual_vec

    else:
        raise NotImplementedError(f"{problem_type} not yet implemented")

    return duals



def dual_distance(dual_1, dual_2):
    """
    Normalized Euclidean distance between two dual vectors.
    """
    norm = np.linalg.norm(dual_1) + np.linalg.norm(dual_2)
    return np.linalg.norm(dual_1 - dual_2) / norm if norm > 0 else 0


def compute_scenario_weights(duals, x_trials, M):
    """
    Compute average dual distances between all pairs of scenarios.
    """
    weights = {}
    for m1, m2 in product(M, M):
        if m1 == m2:
            continue
        total = 0
        for i in range(len(x_trials)):
            d1 = duals.get((i, m1))
            d2 = duals.get((i, m2))
            if d1 is not None and d2 is not None:
                total += dual_distance(d1, d2)
        weights[(m1, m2)] = total / len(x_trials)
    return weights


def generate_random_walks(weights, M, n_walks, walk_length, seed):
    """
    Simulate biased random walks over the scenario similarity graph.
    """
    np.random.seed(seed)
    walks = []
    for _ in range(n_walks):
        for start in M:
            walk = [start]
            current = start
            for _ in range(walk_length):
                neighbors = [t for (s, t) in weights if s == current]
                if not neighbors:
                    break
                probs = np.array([weights[(current, t)] for t in neighbors])
                probs = probs / probs.sum()
                current = np.random.choice(neighbors, p=probs)
                walk.append(current)
            walks.append([str(s) for s in walk])
    return walks


def learn_embeddings(walks, feature_dim, window, min_count, sg):
    """
    Train Word2Vec model over random walks and extract embeddings.
    """
    model = Word2Vec(
        sentences=walks,
        vector_size=feature_dim,
        window=window,
        min_count=min_count,
        sg=sg
    )
    return {int(k): model.wv[k] for k in model.wv.index_to_key}


def generate_scenario_features(
    data, problem_type,
    n_trials=30, binary_fraction=0.3, seed=42,
    n_walks=20, walk_length=10,
    feature_dim=8, window=5, min_count=1, sg=1
):
    """
    Full pipeline: Generate first-stage samples, solve duals, build graph, embed scenarios.
    """
    x_trials = generate_first_stage_trials(n_trials, data, binary_fraction, seed, problem_type)
    duals = get_dual_vectors(x_trials, data, problem_type)
    weights = compute_scenario_weights(duals, x_trials, list(data.M))
    walks = generate_random_walks(weights, list(data.M), n_walks, walk_length, seed)
    feature_vectors = learn_embeddings(walks, feature_dim, window, min_count, sg)
    return feature_vectors



