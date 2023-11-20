import numpy as np
from evovaq.tools.support import compute_statistics, print_info, Logbook, set_progress_bar, BestIndividualTracker, \
    FinalResult
from evovaq.tools.operators import sel_best
import random
from typing import Union
from evovaq.problem import Problem


class BBBC(object):
    """
    The Big Bang - Big Crunch (BBBC) method as developed by Erol and Eks in 2006 [1] consists of two alternating steps:
    1) a Big Bang phase, where the candidate solutions are randomly distributed over the search space; 2) a Big Crunch
    phase, where a contraction operation estimates a weighted average, denoted as Centre of Mass, of the randomly
    distributed candidate solutions. During the Big Bang phases, new candidate solutions are generated considering the
    Center of Mass and the best global solution, as introduced in [2].

    References:
        [1] O. K. Erol and I. Eksin, “A new optimization method: big bang–big crunch”, Advances in Engineering Software,
        vol. 37, no. 2, pp. 106– 111, 2006.

        [2] C. V. Camp, “Design of space trusses using big bang–big crunch optimization,” Journal of Structural
        Engineering, vol. 133, no. 7, pp. 999–1008, 2007.

    Args:
        elitism: If True, the best solution of current population is transferred directly into the next generation.
        alpha: Hyperparameter limiting the size of the search space.
        beta: Hyperparameter defined in range (0,1) controlling the influence of the best individual on the location of
              new candidate solutions.
    """

    def __init__(self, elitism: bool = True, alpha: float = 10.0, beta: float = 0.25):
        self.elitism = elitism
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def compute_centre_of_mass(population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Compute the centre of mass.

        Args:
            population: A population of possible solutions as array of real parameters with (`pop_size`, `n_params`)
                        shape.
            fitness: A set of fitness values associated to the population as array of real values with (`pop_size`,)
                     shape.

        Returns:
            The centre of mass as an array of real parameters with (`n_params`,) shape.
        """
        return np.sum(population / fitness[:, None], axis=0) / np.sum(1 / fitness)

    def generate_new_individual(
            self, problem: Problem, com: np.ndarray, best: np.ndarray, gen: int) -> np.ndarray:
        """
        Generate a new individual distributed between the center of mass and the best global solution.

        Args:
            problem: :class:`~.Problem` to be solved.
            com: The centre of mass as an array of real parameters with (`n_params`,) shape.
            best: The global best solution as an array of real parameters with (`n_params`,) shape.
            gen: Generation number.

        Returns:
            A new individual as an array of real parameters with (`n_params`,) shape.
        """
        r = np.random.normal(0, 1)
        diff = np.zeros(com.shape)
        if isinstance(problem.param_bounds, tuple):
            diff[:] = np.full(len(com), problem.param_bounds[1] - problem.param_bounds[0])
        elif isinstance(problem.param_bounds, list):
            diff[:] = np.array([b[1] - b[0] for b in problem.param_bounds])
        return self.beta * com + (1 - self.beta) * best + diff * r * self.alpha / gen

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

        # Big Bang phase: create and initialize the population
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

        # Begin the generational process
        for gen in range(1, max_gen + 1):

            # Check the stopping criterion
            if max_nfev is not None and tot_nfev >= max_nfev:
                pbar.close()
                break

            # Select the best individual of the current population
            best, best_fit = sel_best(population, fitness, 1)

            # Big Crunch phase: compute the Centre of Mass
            com = self.compute_centre_of_mass(population, fitness)

            # Clone the individuals to produce the offspring
            offspring = population.copy()

            # Big Bang phase: create new solutions around the Centre of Mass
            for child in offspring:
                child[:] = self.generate_new_individual(problem, com, best, gen)

            # Check the bounds of the parameters
            for child in offspring:
                child[:] = problem.check_bounds(child)

            # Evaluate the individuals with an invalid fitness
            fit_offspring = np.array(list(map(problem.evaluate_fitness, offspring)))
            nfev = len(offspring)
            tot_nfev += len(offspring)

            if self.elitism:
                joint_pop = np.concatenate((offspring, best))
                joint_fitness = np.concatenate((fit_offspring, best_fit))

                # Sort from the best (min fitness) to worst (max fitness) individual
                sorted_idx = np.argsort(joint_fitness)

                # Replacement
                population[:] = joint_pop[sorted_idx[:pop_size]]
                fitness[:] = joint_fitness[sorted_idx[:pop_size]]
            else:
                population[:] = offspring
                fitness[:] = fit_offspring

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
