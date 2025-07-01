import numpy as np
from itertools import product
from gensim.models import Word2Vec
from subproblem_dual import solve_dual_subproblem_general


def generate_first_stage_trials(n_trials, data, binary_fraction, seed, problem_type):
    """
    Generate feasible first-stage solutions x satisfying problem-specific constraints.

    Args:
        n_trials (int): Number of first-stage samples to generate.
        data: Data object containing problem parameters.
        binary_fraction (float): Fraction of variables to treat as binary (rest in [0, 1]).
        seed (int): Random seed.
        problem_type (str): One of {"UFL", "HUB", "CMND", "MCFL", "SSLP"}

    Returns:
        np.ndarray: Matrix of feasible first-stage solutions (n_trials × |J|).
    """
    np.random.seed(seed)
    x_trials = []
    n_binary = int(binary_fraction * len(data.J))
    indices = list(range(len(data.J)))

    while len(x_trials) < n_trials:
        np.random.shuffle(indices)
        bin_idx = indices[:n_binary]
        cont_idx = indices[n_binary:]

        x = np.zeros(len(data.J))
        x[bin_idx] = np.random.choice([0, 1], size=n_binary)
        x[cont_idx] = np.random.uniform(0, 1, size=len(cont_idx))

        # Feasibility check based on problem type
        if problem_type.upper() == "UFL":
            # Total capacity from selected facilities must meet or exceed total demand
            total_capacity = sum(data.capacities[j] * x[j] for j in data.J)
            if total_capacity >= data.total_demand - 1e-6:
                x_trials.append(x)

        elif problem_type.upper() == "HUB":
            # TODO: Add feasibility condition for HUB problem
            pass

        elif problem_type.upper() == "CMND":
            # TODO: Add feasibility condition for CMND problem
            pass

        elif problem_type.upper() == "MCFL":
            # TODO: Add feasibility condition for MCFL problem
            pass

        elif problem_type.upper() == "SSLP":
            # TODO: Add feasibility condition for SSLP problem
            pass

    return np.array(x_trials)


def get_dual_vectors(x_trials, data, problem_type):
    """
    Solve dual subproblems for each x and each scenario.

    Returns:
        duals: Dict[(trial_idx, scenario_idx)] → dual_vector (NumPy array)
    """
    duals = {}
    for i, x in enumerate(x_trials):
        x_dict = {j: x[j] for j in data.J}
        for m in data.M:
            _, dual_dict = solve_dual_subproblem_general(problem_type, x_dict, m, data)
            # For UFL, only include lambda vector for similarity
            dual_vec = np.array([dual_dict[i_] for i_ in data.I])
            duals[(i, m)] = dual_vec
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