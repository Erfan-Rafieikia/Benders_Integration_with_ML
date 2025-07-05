import numpy as np
from itertools import product
from gensim.models import Word2Vec
from ..Problem.subproblem_dual import solve_dual_subproblem
import numpy as np
import time 

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
            x_trials.append(x) # For UFL, we don't havce any constraints on the first-stage variables, so all generated vectors are feasible.
            #total_capacity = sum(data.capacities[j] * x[j] for j in range(num_vars))
            #if total_capacity >= data.total_demand - 1e-6:
            #   x_trials.append(x)

        # Add more problem classes as needed
        total_trials += 1  # Avoid infinite loop in case of tight feasibility
        print(f"Generated {len(x_trials)} feasible first-stage solutions so far out of {n_trials} trials.")

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
                duals[(trial_idx, subproblem_dual_idx)] = dual_vec #Mapping from (trial index, subproblem index) to dual vector. So, it inludes optimal dual variables for each trial and each subproblem (customer in UFL).

    else:
        raise NotImplementedError(f"{problem_type} not yet implemented")

    return duals



def compute_scenario_weights(duals):
    """
    Computes the average normalized Euclidean distance between all pairs of scenarios (subproblems)
    based on their dual variable vectors across all sampled trials.

    Each dual vector corresponds to a specific trial and scenario combination:
        duals[(trial_idx, scenario_idx)] → np.ndarray (dual vector, e.g., [lambda, mu_1, mu_2, ..., mu_J] in case of UFL)

    The result is a symmetric similarity matrix (dict), where:
        weights[(s1, s2)] = average distance between subproblems s1 and s2 over all trials.

    Scenarios with similar dual behavior will have smaller distances.
    This is useful for building a scenario similarity graph or clustering.

    Args:
        duals (dict): Mapping from (trial index, scenario index) to dual vector (NumPy array).

    Returns:
        dict: Mapping from (scenario1, scenario2) to average normalized distance.
    """
    if duals is None:
        raise ValueError("Input 'duals' is None. Cannot compute scenario weights.")
        return {}

    weights = {}  # Final dictionary of pairwise distances between scenarios

    # Get all unique scenario IDs from duals, e.g., {0, 1, 2}
    scenario_ids = sorted({scenario for (_, scenario) in duals}) 

    # Get all unique trial indices, e.g., {0, 1, ..., n_trials - 1}
    trial_ids = sorted({trial for (trial, _) in duals})

    # Loop over unique pairs of scenarios (avoid repeats)
    for i, s1 in enumerate(scenario_ids): 
        for s2 in scenario_ids[i + 1:]:
            total = 0   # Sum of distances over all trials
            count = 0   # Count of valid trial comparisons

            # Loop over all trials to compare s1 and s2
            for t in trial_ids:
                d1 = duals.get((t, s1))  # Dual vector for scenario s1 at trial t
                d2 = duals.get((t, s2))  # Dual vector for scenario s2 at trial t

                if d1 is not None and d2 is not None:
                    # Calculate normalized Euclidean distance between d1 and d2
                    # Example: d1 = [1, 2], d2 = [2, 4] → norm = 5.38, dist = 0.37
                    norm = np.linalg.norm(d1) + np.linalg.norm(d2)
                    dist = np.linalg.norm(d1 - d2) / norm if norm > 0 else 0

                    total += dist  # Accumulate distance
                    count += 1     # Count valid comparisons

            # Compute average distance for this scenario pair
            avg_dist = total / count if count > 0 else 0

            # Store both (s1, s2) and (s2, s1) to make the graph symmetric
            weights[(s1, s2)] = avg_dist
            weights[(s2, s1)] = avg_dist

    return weights  # Dictionary of average pairwise distances


def generate_random_walks(weights, n_walks, walk_length, seed):
    """
    Simulate biased random walks over a scenario similarity graph defined by `weights`.

    Each walk starts from a node (scenario), and moves to neighbors with probabilities 
    proportional to edge weights (i.e., similarity/distance).

    Args:
        weights (dict): Dictionary of form {(s1, s2): weight} representing pairwise distances/similarities.
        n_walks (int): Number of walks to generate *per node*.
        walk_length (int): Number of steps in each walk.
        seed (int): Random seed for reproducibility.

    Returns:
        list of list[str]: Each inner list is a walk (sequence of scenario indices as strings).
    """
    np.random.seed(seed)
    walks = []

    # Extract all nodes (scenarios) from weight keys
    nodes = sorted({s for s, _ in weights})

    for _ in range(n_walks):
        for start in nodes:
            walk = [start]
            current = start
            for _ in range(walk_length):
                # Get neighbors of the current node
                neighbors = [t for (s, t) in weights if s == current]
                if not neighbors:
                    break
                # Get transition probabilities based on weights
                probs = np.array([weights[(current, t)] for t in neighbors])
                probs = probs / probs.sum()
                # Choose next node based on biased probability
                current = np.random.choice(neighbors, p=probs)
                walk.append(current)
            walks.append([str(s) for s in walk])  # Convert all nodes to strings for Word2Vec
    return walks



def learn_embeddings(walks, feature_dim, window, min_count, sg):
    """
    Train a Word2Vec model on random walks over the scenario graph,
    learning continuous vector embeddings for each scenario.

    Args:
        walks (list[list[str]]): Random walks, each a list of scenario IDs as strings.
        feature_dim (int): Dimensionality of the output embeddings.
        window (int): Context window size for Word2Vec.
        min_count (int): Minimum number of occurrences for a node to be embedded.
        sg (int): Skip-gram (1) vs CBOW (0) model.

    Returns:
        dict[int, np.ndarray]: Mapping from scenario index to embedding vector.
    """
    model = Word2Vec(
        sentences=walks,
        vector_size=feature_dim,
        window=window,
        min_count=min_count,
        sg=sg
    )
    return {int(k): model.wv[k] for k in model.wv.index_to_key if k is not None}


def generate_scenario_features(
    data, problem_type,
    n_trials=30, binary_fraction=0.3, seed=42,
    n_walks=20, walk_length=10,
    feature_dim=8, window=5, min_count=1, sg=1
):
    """
    End-to-end pipeline to generate scenario (subproblem) embeddings.

    Steps:
        1. Generate first-stage decision vectors (x_trials).
        2. Solve all subproblem duals for each trial → get dual vectors.
        3. Build a scenario similarity graph using average dual distances.
        4. Generate random walks over this graph using weighted transitions.
        5. Train a Word2Vec model over the walks to embed scenarios.

    Returns:
        dict[int, np.ndarray]: Mapping from scenario index to embedding vector.
    """
    start_time = time.time()
    print("Starting feature generation for problem type [{}]...".format(problem_type))
    # Step 1: Generate first-stage trials
    x_trials = generate_first_stage_trials(n_trials, data, binary_fraction, seed, problem_type)
    feature_gen_time = time.time() - start_time
    print("Generation of first-stage trials for problem type [{}] completed in {:.2f} seconds.".format(problem_type, feature_gen_time))
    start_time = time.time()
    # Step 2: Get dual vectors for each trial
    duals = get_dual_vectors(x_trials, data, problem_type)
    duals_time = time.time() - start_time
    print("Dual vectors for problem type [{}] generated in {:.2f} seconds.".format(problem_type, duals_time))   
    # Step 3: Compute scenario weights based on dual vectors
    start_time = time.time()
    weights = compute_scenario_weights(duals)
    weights_time = time.time() - start_time
    print("Scenario weights for problem type [{}] computed in {:.2f} seconds.".format(problem_type, weights_time))
    # Step 4: Generate random walks over the scenario graph
    start_time = time.time()
    walks = generate_random_walks(weights, n_walks, walk_length, seed)
    walks_time = time.time() - start_time
    print("Random walks for problem type [{}] generated in {:.2f} seconds.".format(problem_type, walks_time))
    # Step 5: Learn embeddings from the random walks
    start_time = time.time()
    feature_vectors = learn_embeddings(walks, feature_dim, window, min_count, sg)
    embedding_time = time.time() - start_time
    print("Embeddings for problem type [{}] learned in {:.2f} seconds.".format(problem_type, embedding_time))
    return {
        "features": feature_vectors,
        "x_trials": x_trials,
        "params": {
            "n_trials": n_trials,
            "binary_fraction": binary_fraction,
            "seed": seed,
            "n_walks": n_walks,
            "walk_length": walk_length,
            "feature_dim": feature_dim,
            "window": window,
            "min_count": min_count,
            "sg": sg
        }
    }



