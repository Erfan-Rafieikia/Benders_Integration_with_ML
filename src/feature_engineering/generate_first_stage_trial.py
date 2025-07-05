import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict

def get_dual_vectors_from_trials(trial_duals):
    """
    Convert flat trial_duals dict (trial_idx, scenario_idx) → dual vector
    into a structure suitable for weight computation.
    """
    return trial_duals


def compute_scenario_weights(duals):
    weights = {}
    scenario_ids = sorted({scenario for (_, scenario) in duals})
    trial_ids = sorted({trial for (trial, _) in duals})

    for i, s1 in enumerate(scenario_ids): 
        for s2 in scenario_ids[i + 1:]:
            total, count = 0, 0
            for t in trial_ids:
                d1 = duals.get((t, s1))
                d2 = duals.get((t, s2))
                if d1 is not None and d2 is not None:
                    norm = np.linalg.norm(d1) + np.linalg.norm(d2)
                    dist = np.linalg.norm(d1 - d2) / norm if norm > 0 else 0
                    total += dist
                    count += 1
            avg_dist = total / count if count > 0 else 0
            weights[(s1, s2)] = avg_dist
            weights[(s2, s1)] = avg_dist
    return weights


def generate_random_walks(weights, n_walks, walk_length, seed):
    np.random.seed(seed)
    walks = []
    nodes = sorted({s for s, _ in weights})

    for _ in range(n_walks):
        for start in nodes:
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
    model = Word2Vec(
        sentences=walks,
        vector_size=feature_dim,
        window=window,
        min_count=min_count,
        sg=sg
    )
    return {int(k): model.wv[k] for k in model.wv.index_to_key if k is not None}


def compute_scenario_features_from_duals(
    trial_duals,
    feature_params
):
    """
    Generate scenario embeddings from dual vectors across multiple trials.
    Args:
        trial_duals: dict of {(trial_idx, scenario_idx) → dual vector}
        feature_params: dict with keys:
            - n_walks
            - walk_length
            - seed
            - feature_dim
            - window
            - min_count
            - sg
    Returns:
        dict[scenario_idx] → feature vector
    """
    duals = get_dual_vectors_from_trials(trial_duals)
    weights = compute_scenario_weights(duals)
    walks = generate_random_walks(weights, **{k: feature_params[k] for k in ["n_walks", "walk_length", "seed"]})
    return learn_embeddings(walks, **{k: feature_params[k] for k in ["feature_dim", "window", "min_count", "sg"]})
