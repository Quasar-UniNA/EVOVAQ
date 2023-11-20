import numpy as np
from typing import Union, Sequence


# ---------- SELECTION OPERATORS ------------
def sel_best(population: np.ndarray, fitness: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Select the first `k` best individuals (with the smallest objective values) in the population.

    Args:
        population: A population of individuals as array of real parameters with (`pop_size`, `n_params`)
                    shape.
        fitness: A set of fitness values associated to the population as array of real values with (`pop_size`,)
                shape.
        k: Number of individuals to be selected.

    Returns:
        Subsets of `k` best individuals and fitness values.
    """
    sorted_idx = np.argsort(fitness)
    return population[sorted_idx[:k]], fitness[sorted_idx[:k]]


def sel_permutation(population: np.ndarray, fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Select the mating pool by permutation of the population indices.

    Args:
        population: A population of individuals as array of real parameters with (`pop_size`, `n_params`)
                    shape.
        fitness: A set of fitness values associated to the population as array of real values with (`pop_size`,)
                shape.

    Returns:
        Mating pool with individuals and fitness values with new indices.
    """
    permutation_idx = np.random.permutation(len(population))
    return population[permutation_idx], fitness[permutation_idx]


def sel_tournament(population: np.ndarray, fitness: np.ndarray, k: int, tournsize: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Select the best individual among `tournsize` randomly chosen individuals, `k` times.

    Args:
        population: A population of individuals as array of real parameters with (`pop_size`, `n_params`)
                    shape.
        fitness: A set of fitness values associated to the population as array of real values with (`pop_size`,)
                shape.
        k: Number of individuals to be selected.
        tournsize:  Number of random individuals participating in each tournament.

    Returns:
        Subsets of `k` individuals and fitness values.
    """
    chosen = []
    for _ in range(k):
        aspirants = np.random.choice(len(population), size=tournsize)
        fit_aspirants = fitness[aspirants]
        winner = aspirants[np.argmin(fit_aspirants)]
        chosen.append(winner)
    return population[chosen], fitness[chosen]


def sel_random(population: np.ndarray, fitness: np.ndarray, k: int, replace: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Select `k` individuals randomly.

    Args:
        population: A population of individuals as array of real parameters with (`pop_size`, `n_params`)
                    shape.
        fitness: A set of fitness values associated to the population as array of real values with (`pop_size`,)
                shape.
        k: Number of individuals to be selected.
        replace: Whether the sample is with or without replacement. Default is True, meaning that an element can be
                 selected multiple times.

    Returns:
        Subsets of `k` individuals and fitness values randomly chosen.
    """
    random_idx = np.random.choice(len(population), size=k, replace=replace)
    return population[random_idx], fitness[random_idx]


# ---------- CROSSOVER OPERATORS ------------
def cx_blx_alpha(par1: np.ndarray, par2: np.ndarray, alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    BLX-alpha crossover.

    Args:
        par1: The first parent as array of real parameters with (`n_params`,) shape.
        par2: The second parent as array of real parameters with (`n_params`,) shape.
        alpha: Positive real hyperparameter.

    Returns:
        The two resulting children.
    """
    d = np.abs(par1 - par2)
    stacked = np.stack((par1, par2), axis=1)
    lower_bounds = np.min(stacked, axis=1) - alpha * d
    upper_bounds = np.max(stacked, axis=1) + alpha * d
    child1 = np.random.uniform(lower_bounds, upper_bounds, par1.shape)
    child2 = np.random.uniform(lower_bounds, upper_bounds, par2.shape)
    return child1, child2


def cx_one_point(par1: np.ndarray, par2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    One-point crossover.

    Args:
        par1: The first parent as array of real parameters with (`n_params`,) shape.
        par2: The second parent as array of real parameters with (`n_params`,) shape.

    Returns:
        The two resulting children.
    """
    point = np.random.randint(1, par1.size - 1)
    child1 = np.concatenate((par1[:point], par2[point:]))
    child2 = np.concatenate((par2[:point], par1[point:]))
    return child1, child2


def cx_two_point(par1: np.ndarray, par2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-point crossover.

    Args:
        par1: The first parent as array of real parameters with (`n_params`,) shape.
        par2: The second parent as array of real parameters with (`n_params`,) shape.

    Returns:
        The two resulting children.
    """
    point1, point2 = sorted(np.random.choice(par1.size, 2, replace=False))
    child1 = np.concatenate((par1[:point1], par2[point1:point2], par1[point2:]))
    child2 = np.concatenate((par2[:point1], par1[point1:point2], par2[point2:]))
    return child1, child2


def cx_uniform(par1: np.ndarray, par2: np.ndarray, cx_indpb: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniform crossover.

    Args:
        par1: The first parent as array of real parameters with (`n_params`,) shape.
        par2: The second parent as array of real parameters with (`n_params`,) shape.
        cx_indpb: Independent probability for each parameter to be exchanged.

    Returns:
        The two resulting children.
    """
    mask = np.random.choice([True, False], size=par1.size, p=[cx_indpb, 1 - cx_indpb])
    child1 = np.where(mask, par1, par2)
    child2 = np.where(mask, par2, par1)
    return child1, child2


# ---------- MUTATION OPERATORS ------------
def mut_gaussian(individual: np.ndarray, mu: Union[float, Sequence] = 0, sigma: Union[float, Sequence] = 1,
                 mut_indpb: float = 0.1) -> np.ndarray:
    """
    Gaussian mutation.

    Args:
        individual: Individual to be mutated as array of real parameters with (`n_params`,) shape.
        mu: Mean or sequence of means for the gaussian addition mutation.
        sigma: Standard deviation or sequence of standard deviations for the gaussian addition mutation.
        mut_indpb: Independent probability for each parameter to be mutated.

    Returns:
        Resulting mutated individual.
    """
    mask = np.random.choice([True, False], size=individual.size, p=[mut_indpb, 1 - mut_indpb])
    mutated_individual = individual + mask * np.random.normal(mu, sigma, size=individual.shape)
    return mutated_individual


def mut_flip_bit(individual: np.ndarray, mut_indpb: float = 0.1) -> np.ndarray:
    """
    Flip-bit mutation.

    Args:
        individual: Individual to be mutated as array of real parameters with (`n_params`,) shape.
        mut_indpb: Independent probability for each parameter to be mutated.

    Returns:
        Resulting mutated individual.
    """
    mutated_individual = individual.copy()
    mask = np.random.choice([True, False], size=individual.size, p=[mut_indpb, 1 - mut_indpb])
    mutated_individual[mask] = 1 - mutated_individual[mask]
    return mutated_individual


SELECTION_ARGS = {
    sel_tournament: ['tournsize']
}

CROSSOVER_ARGS = {
    cx_blx_alpha: ['alpha'],
    cx_uniform: ['cx_indpb']
}

MUTATION_ARGS = {
    mut_gaussian: ['mu', 'sigma', 'mut_indpb'],
    mut_flip_bit: ['mut_indpb']
}