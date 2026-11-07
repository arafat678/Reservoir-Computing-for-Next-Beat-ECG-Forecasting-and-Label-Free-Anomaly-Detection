"""
genetic_algorithm.py
====================
Genetic Algorithm (GA) neuroevolution for reservoir hyperparameters.
Tunes parameters on validation next-beat MSE without ever seeing beat labels.

Matches Section 5.5 and Table 4 & 5 of:
"Genetically Tuned Reservoir Computing for Next-Beat ECG Forecasting and Label-Free Anomaly Detection:
A Negative-Findings Benchmark Against Simple Sequence Baselines"
(Arafat Ali Mondal and Ankush Kumar)
"""

import numpy as np
from models import build_reservoir, fit_reservoir_readout, compute_reservoir_states, W_OUT

# Hyperparameter search bounds (Table 4)
BOUNDS = {
    "N":     (50, 300),      # Number of reservoir units
    "rho":   (0.1, 1.6),     # Spectral radius
    "kappa": (0.02, 0.30),   # Recurrent sparsity
    "s":     (0.005, 2.0),   # Input scaling (log-sampled)
    "alpha": (0.02, 1.0),    # Leak rate
    "lam":   (1e-8, 1.0),    # Ridge regularisation parameter (log-sampled)
    "gamma": (0.0, 2.0),     # Dendritic coupling
    "beta":  (0.1, 4.0)      # Gating steepness
}

LOG_SCALE_GENES = {"s", "lam"}
GENES_ESN = ["N", "rho", "kappa", "s", "alpha", "lam"]
GENES_DEN = ["N", "rho", "kappa", "s", "alpha", "lam", "gamma", "beta"]

# Exact Archived Genomes from Table 5 (for instant 100% paper reproduction)
ARCHIVED_GENOMES = {
    "esn_ga": {
        42:  {"N": 273, "rho": 0.936, "kappa": 0.109, "s": 0.182, "alpha": 0.866, "lam": 2.2e-7},
        123: {"N": 228, "rho": 0.725, "kappa": 0.042, "s": 0.219, "alpha": 0.663, "lam": 2.2e-7}
    },
    "dendritic_ga": {
        42:  {"N": 263, "rho": 0.842, "kappa": 0.112, "s": 0.276, "alpha": 0.645, "lam": 0.0316, "gamma": 0.545, "beta": 3.00},
        123: {"N": 266, "rho": 1.084, "kappa": 0.067, "s": 0.230, "alpha": 0.509, "lam": 1.7e-8,  "gamma": 0.403, "beta": 3.61}
    }
}


def sample_random_gene(param_name, rng):
    """Draws a random gene value uniformly or log-uniformly within bounds."""
    lo, hi = BOUNDS[param_name]
    if param_name == "N":
        return int(rng.integers(lo, hi + 1))
    if param_name in LOG_SCALE_GENES:
        return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))
    return float(rng.uniform(lo, hi))


def mutate_genome(genome, genes, rng):
    """Applies Gaussian mutation with probability 0.5 per parameter."""
    mutated = dict(genome)
    for k in genes:
        lo, hi = BOUNDS[k]
        if k == "N":
            if rng.random() < 0.25:
                mutated[k] = sample_random_gene(k, rng)
        elif rng.random() < 0.5:
            if k in LOG_SCALE_GENES:
                val = np.log10(mutated[k]) + rng.normal(0, 0.12 * (np.log10(hi) - np.log10(lo)))
                mutated[k] = float(10 ** np.clip(val, np.log10(lo), np.log10(hi)))
            else:
                mutated[k] = float(np.clip(mutated[k] + rng.normal(0, 0.12 * (hi - lo)), lo, hi))
    return mutated


def evaluate_validation_fitness(genome, dendritic, seed, data):
    """
    Evaluates next-beat forecasting MSE on the validation partition (records 122, 208, 220).
    Labels are never seen.
    """
    R = build_reservoir(genome, seed=seed, dendritic=dendritic)
    try:
        W_out = fit_reservoir_readout(R, data)
        val_mses = []
        for rid in data.split["val"]:
            U = data.U[rid]
            X = compute_reservoir_states(R, U)
            Xa = np.hstack([X[W_OUT:-1], np.ones((len(X) - W_OUT - 1, 1))])
            pred = Xa @ W_out
            target = U[W_OUT + 1:]
            val_mses.append(np.mean((pred - target) ** 2))
        fitness = float(np.mean(val_mses))
        return fitness if np.isfinite(fitness) else 1e3
    except Exception:
        return 1e3


def run_genetic_algorithm(dendritic, seed, data, population_size=30, generations=40, verbose=True):
    """
    Runs GA neuroevolution:
      - Population: 30
      - Generations: 40
      - Elitism: top 25% kept intact
      - Uniform crossover with probability 0.7
      - Gaussian mutation with probability 0.5
    """
    genes = GENES_DEN if dendritic else GENES_ESN
    rng = np.random.default_rng(seed + 1000)
    fitness_cache = {}

    def get_fitness(g):
        key = tuple(round(float(g[k]), 8) for k in genes)
        if key not in fitness_cache:
            fitness_cache[key] = (evaluate_validation_fitness(g, dendritic, seed, data), dict(g))
        return fitness_cache[key][0]

    population = [{k: sample_random_gene(k, rng) for k in genes} for _ in range(population_size)]
    history = []

    model_name = "Dendritic ESN" if dendritic else "Standard ESN"
    if verbose:
        print(f"Starting GA optimization for {model_name} (Seed {seed}, Pop {population_size}, Gen {generations})...")

    for gen in range(generations):
        fitnesses = np.array([get_fitness(g) for g in population])
        best_gen_fitness = float(fitnesses.min())
        history.append(best_gen_fitness)

        if verbose and ((gen + 1) % 10 == 0 or gen == 0 or gen == generations - 1):
            print(f"  Gen [{gen+1:02d}/{generations}] Best Val MSE: {best_gen_fitness:.5f}")

        if gen == generations - 1:
            break

        # Elitism: retain top 25%
        num_elites = max(2, population_size // 4)
        elite_indices = np.argsort(fitnesses)[:num_elites]
        elites = [population[i] for i in elite_indices]

        # Spawn next generation
        new_population = [dict(e) for e in elites]
        while len(new_population) < population_size:
            p1, p2 = rng.choice(len(elites), 2, replace=False)
            if rng.random() < 0.7:  # Uniform crossover
                child = {k: (elites[p1][k] if rng.random() < 0.5 else elites[p2][k]) for k in genes}
            else:
                child = dict(elites[p1])
            new_population.append(mutate_genome(child, genes, rng))

        population = new_population

    best_fitness, best_genome = min(fitness_cache.values(), key=lambda v: v[0])
    return best_genome, best_fitness, history

