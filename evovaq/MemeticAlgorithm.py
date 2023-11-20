import random
import numpy as np
from functools import partial
from evovaq.tools.support import compute_statistics, print_info, Logbook, set_progress_bar, BestIndividualTracker, \
    FinalResult
from evovaq.tools.operators import sel_best, sel_random
from typing import Callable, Union
from evovaq.problem import Problem


class MA(object):
    """
    Memetic Algorithm (MA) [1] is an evolutionary approach that merge population- and local-based methods to improve
    exploration and exploitation capabilities in visiting the problem search space.
    In `evovaq`, MA workflow described in [2] is implemented.

    References:
        [1] P. Moscato, et al., "On evolution, search, optimization, genetic algorithms and martial arts: towards memetic
        algorithms", Caltech concurrent computation program, C3P Report, vol. 826, pp. 37, 1989.

        [2] Giovanni Acampora, Angela Chiatto, and Autilia Vitiello, "Training circuit-based quantum classifiers through
        memetic algorithms", Pattern Recognition Letters, Elsevier, 2023.

    Args:
        global_search: Population-based method used to evolve the population of individuals. The global search function
                       is defined as ``global_search(prob, pop, fitness, *args) -> (offspring, fit_offspring, nfev)``, where
                       ``prob`` is :class:`~.Problem` to be solved; ``pop`` and ``fitness`` are two arrays of real parameters
                       with (`pop_size`,`n_params`) and (`n_params`,) shape, respectively; ``args`` is a tuple of other
                       fixed parameters needed to specify the function; and the output is the resulting offspring and
                       fitness values  with the same shape as ``pop`` and ``fitness``, and number of fitness evaluations
                       completed during the evolution. Here, it is possible to use :class:`~.GA` or :class:`~.DE`
                       as a population-based method via :meth:`~DE.evolve_population` method.
        local_search: Local search method used to improve a subset of individuals. The local search function is defined
                      as ``local_search(prob, ind, fitness, *args) -> (ind, fitness, nfev)``, where ``prob`` is
                      :class:`~.Problem` to be solved; ``ind``  is an array of real parameters with (`n_params`,);
                      ``fitness`` is a float value; ``args`` is a tuple of other fixed parameters needed to specify the
                      function; and the output is the improved individual and fitness value with the same shape as
                      ``ind`` and ``fitness``, and number of fitness evaluations completed during the local research.
                      Here, it is possible to use :class:`~.HC` as a local search method via :meth:`~HC.stochastic_var`
                      method.
        sel_for_refinement: Selection operator used to choose the individuals undergoing local refinement.
        frequency: Individual learning frequency defined in the range (0 , 1) influencing the number of individuals that
                   is undergone to local refinement.
        intensity: Individual learning intensity representing the maximum computational budget allowable for individual
                   learning to expend on improving a single solution. Here, it corresponds to the maximum number
                   of iterations to be performed during the local search.
        elitism: If True, the best solution of current population is transferred directly into the next generation.
    """
    def __init__(
            self, global_search: Callable, local_search: Callable, sel_for_refinement: Callable, frequency: float,
            intensity: int, elitism: bool = True):
        self.global_search = global_search
        self.local_search = local_search
        self.sel_for_refinement = sel_for_refinement
        self.frequency = frequency
        self.intensity = intensity
        self.elitism = elitism

        if self.sel_for_refinement == sel_random:
            self.sel_for_refinement = partial(sel_random, replace=False)

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

        # Begin the generational process
        for gen in range(1, max_gen + 1):

            # Check the stopping criterion
            if max_nfev is not None and tot_nfev >= max_nfev:
                pbar.close()
                break

            best, best_fit = sel_best(population, fitness, 1)

            # Global search method used to evolve the population of possible solutions
            offspring, fit_offspring, glob_nfev = self.global_search(problem, population, fitness)
            tot_nfev += glob_nfev
            nfev = glob_nfev

            # Local search method used to improve a subset of individuals
            omega_idx, fit_omega = self.sel_for_refinement(np.arange(len(offspring)), fit_offspring,
                                                           int(self.frequency * pop_size))

            for ind_idx in omega_idx:
                for _ in range(self.intensity):
                    offspring[ind_idx][:], fit_offspring[ind_idx], loc_nfev = self.local_search(problem,
                                                                                                offspring[ind_idx],
                                                                                                fit_offspring[ind_idx])
                    tot_nfev += loc_nfev
                    nfev += loc_nfev

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
