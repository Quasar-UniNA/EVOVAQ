import numpy as np
from functools import partial
import random
from evovaq.tools.support import compute_statistics, print_info, Logbook, set_progress_bar, BestIndividualTracker, \
    FinalResult
from evovaq.tools.operators import sel_permutation, sel_best, CROSSOVER_ARGS
from evovaq.problem import Problem
from typing import Union, Callable


class CHC(object):
    """
    Cross generational elitist selection, Heterogeneous recombination, and Cataclysmic mutation (CHC) algorithm is a
    nontraditional genetic algorithm which combines a conservative selection strategy that always
    preserves the best individuals found so far with a radical (highly disruptive) crossover operator that produces
    offspring that are maximally different from both parents [1]. In detail, it is based on four main components:
    a elitist selection, a highly disruptive crossover, an incest prevention check to avoid the recombination of similar
    solutions, and a population reinitialization method when the population has converged.
    In `evovaq`, a real-coded CHC version based on [2-3] is implemented. However, the similarity is measured between
    fitness values and not between individuals, since different solutions can lead to equal or close cost values when
    training variational quantum circuits.

    References:
        [1] Larry J. Eshelman, "The CHC Adaptive Search Algorithm: How to Have Safe Search When Engaging in
        Nontraditional Genetic Recombination", Foundations of Genetic Algorithms, Elsevier, vol. 1, pp. 265-283, 1991.

        [2] O. Cordón, S. Damas, J. Santamaría, "Feature-based image registration by means of the CHC evolutionary algorithm",
        Image and Vision Computing, vol. 24, Issue 5, pp. 525-533, 2006.

        [3] Cuéllar, M. P., Gómez-Torrecillas, J., Lobillo, F. J., & Navarro, G., "Genetic algorithms with
        permutation-based representation for computing the distance of linear codes", Swarm and Evolutionary Computation,
        vol. 60, pp. 100797, 2021.

    Args:
        crossover: Crossover operator used to mate two individuals. The crossover function is defined as
                   ``cx_function(par1, par2, *args) -> (child1, child2)``, where ``par1`` and ``par2`` are two arrays of
                   real parameters with (`n_params`,) shape, ``args`` is a tuple of other fixed parameters needed to
                   specify the function, and the output is the resulting children with the same shape as the parents.
        distance: Distance metric used to compute the similarity between parents. In this implementation, similarity is
                  measured between two fitness values, so Euclidean distance can be considered.
        multiplier: Factor influencing the initial crossover threshold.
        dec_percentage:  Crossover threshold update rate.
        kwargs: Additional keyword arguments used to set hyperparameter values of crossover operator.
    """
    def __init__(self,
                 crossover: Callable, distance: Callable, multiplier: float = 1, dec_percentage: float = 0.1, **kwargs):
        self.distance = distance
        self.multiplier = multiplier
        self.dec_percentage = dec_percentage
        self.kwargs = kwargs

        if crossover in CROSSOVER_ARGS:
            args = {arg: self.kwargs.get(arg) for arg in CROSSOVER_ARGS[crossover] if arg in self.kwargs.keys()}
            self.crossover = partial(crossover, **args)
        else:
            self.crossover = crossover

    def incest_prevention_check(self, parents: np.ndarray, fitness: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Incest prevention check. Before mating, the similarity between the potential parents is calculated, and if this
        distance does not exceed the threshold `thr`, they are not mated.

        Args:
            parents: A set of parents as array of real parameters with (`pop_size`, `n_params`) shape.
            fitness: A set of fitness values associated to the parents as array of real values with (`pop_size`,)
                    shape.
            thr: Crossover threshold.

        Returns:
            Parents and corresponding fitness values allowed for mating.
        """
        allowed = np.zeros(len(parents), dtype=bool)
        for i in range(0, len(parents), 2):
            if self.distance(fitness[i], fitness[i + 1]) > thr:
                allowed[i] = allowed[i + 1] = True
            else:
                pass
        return parents[allowed], fitness[allowed]

    def initialize_cx_threshold(self, population: np.ndarray, fitness: np.ndarray) -> tuple[float, float]:
        """
        Initialize the crossover threshold and compute the decrement value.

        Args:
            population: A population of individuals as array of real parameters with (`pop_size`, `n_params`)
                        shape.
            fitness: A set of fitness values associated to the population as array of real values with (`pop_size`,)
                    shape.

        Returns:
            The initial crossover threshold and the decrement value.
        """
        avg_d = 0
        n = 0
        max_d = 0
        for i in range(len(population) - 1):
            for j in range(i + 1, len(population)):
                n += 1
                d = self.distance(fitness[i], fitness[j])
                avg_d += d
                if d > max_d:
                    max_d = d
        avg_d /= n
        return avg_d * self.multiplier, max_d * self.dec_percentage

    def optimize(
            self, problem: Problem, pop_size: int, initial_pop: Union[np.ndarray, None] = None,
            max_nfev: Union[int, None] = None, max_gen: int = 1000, n_run: int = 1, seed: Union[int, float, str, None] = None,
            verbose: bool = True) -> FinalResult:
        """
        Optimize the parameters of the problem to be solved.

        Args:
            problem: :class:`~.Problem` to be solved.
            pop_size: Population size.
            initial_pop: Initial population of possible solutions as array of real parameters with (`pop_size`, `n_params`)
                         shape. If None, the initial population is randomly generated from `param_bounds`.
            max_nfev: Maximum number of fitness evaluations used as stopping criterion. If None, the maximum number of
                      generations `max_gen` is considered as stopping criterion.
            max_gen: Maximum number of generations used as stopping criterion. If `max_nfev` is not None, this is
                     considered as stopping criterion.
            n_run: Independent execution number of the algorithm.
            seed: Initialize the random number generator. If None, the current time is used.
            verbose: If True, the statistics of fitness values is printed during the evolution.

        Returns:
            A :class:`~.FinalResult` containing the optimization result.
        """

        # Set the seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Create and initialize the population
        if initial_pop is None:
            population = problem.generate_random_pop(pop_size)
            fitness = np.array(list(map(problem.evaluate_fitness, population)))
            nfev = len(population)
            tot_nfev = len(population)
        else:
            population = initial_pop[:].copy()
            fitness = np.array(list(map(problem.evaluate_fitness, population)))
            nfev = len(population)
            tot_nfev = len(population)

        # Store the best solution ever found
        best_tracker = BestIndividualTracker()
        best_tracker.update(population, fitness)

        # Set the progress bar considering the stopping criterion
        pbar = set_progress_bar(max_gen, max_nfev)
        if max_nfev is not None:
            pbar.update(nfev)

        # Compute the statistics of the fitness values
        stats = compute_statistics(fitness)

        # Set the logbook
        lg = Logbook()
        lg.record(gen=0, nfev=nfev, **stats)

        # Print the evolution info
        if verbose:
            print_info(n_run=n_run, gen=0, nfev=nfev, **stats, header=True)

        # Initialize threshold
        thr, dec = self.initialize_cx_threshold(population, fitness)

        # Begin the generational process
        for gen in range(1, max_gen + 1):

            # Check the stopping criterion
            if max_nfev is not None and tot_nfev >= max_nfev:
                pbar.close()
                break

            # Parent selection
            parents, fit_parents = sel_permutation(population, fitness)

            # Incest prevention
            allowed_parents, fit_allowed_parents = self.incest_prevention_check(parents, fit_parents, thr)

            # Crossover and replacement
            if len(allowed_parents) > 0:
                # Clone the individuals to produce the offspring
                offspring = allowed_parents.copy()

                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    child1[:], child2[:] = self.crossover(child1, child2)

                # Check the bounds of the parameters
                for child in offspring:
                    child[:] = problem.check_bounds(child)

                # Evaluate the individuals of the offspring
                fit_offspring = np.array(list(map(problem.evaluate_fitness, offspring)))
                nfev = len(offspring)
                tot_nfev += len(offspring)

                # Elitism
                joint_pop = np.concatenate((population, offspring))
                joint_fitness = np.concatenate((fitness, fit_offspring))

                # Sort from the best (min fitness) to worst (max fitness) individual
                sorted_idx = np.argsort(joint_fitness)

                # Replacement
                population[:] = joint_pop[sorted_idx[:pop_size]]
                fitness[:] = joint_fitness[sorted_idx[:pop_size]]

            else:
                sorted_idx = None
                nfev = 0

            # Decrease the distance threshold
            if sorted_idx is None or np.all(sorted_idx[:pop_size] < pop_size):
                thr -= dec
            # Reinitialization process: if thr <= 0, all parents, even equal, combine and thus there is no exploration
                if thr <= 0:
                    best, best_fit = sel_best(population, fitness, 1)
                    rand_individuals = problem.generate_random_pop(pop_size - 1)
                    rand_fitness = np.array(list(map(problem.evaluate_fitness, rand_individuals)))
                    next_population = np.concatenate((best, rand_individuals))
                    next_fitness = np.concatenate((best_fit, rand_fitness))
                    nfev += len(rand_individuals)
                    tot_nfev += len(rand_individuals)
    
                    # Replace the population with a new population
                    population[:] = next_population
                    fitness[:] = next_fitness
    
                    # Re-compute the threshold and the decrement factor
                    thr, dec = self.initialize_cx_threshold(population, fitness)
    
            # Store the best solution ever found
            best_tracker.update(population, fitness)
    
            # Compute the statistics of the fitness values
            stats = compute_statistics(fitness)

            # Record info in the logbook
            lg.record(gen=gen, nfev=nfev, **stats)

            # Print the evolution info
            if verbose:
                print_info(n_run=n_run, gen=gen, nfev=nfev, **stats)

            # Update the progress bar
            if max_nfev is not None:
                pbar.update(nfev)
            else:
                pbar.update()

        res = FinalResult(x=best_tracker.get_best(), fun=best_tracker.get_best_fit(), nfev=tot_nfev, gen=gen,
                          log=lg.get_log())
        return res
