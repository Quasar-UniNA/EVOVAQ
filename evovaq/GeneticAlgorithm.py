import random
import numpy as np
from functools import partial
from evovaq.tools.support import compute_statistics, print_info, Logbook, set_progress_bar, BestIndividualTracker, \
    FinalResult
from evovaq.tools.operators import sel_best
from evovaq.tools.operators import SELECTION_ARGS, CROSSOVER_ARGS, MUTATION_ARGS
from typing import Callable, Union
from evovaq.problem import Problem


class GA(object):

    """
    Genetic Algorithm (GA) is the simplest evolutionary algorithm inspired by Darwinian evolution principles: the evolution
    of a population of possible solutions to a given problem is linked to the concepts of randomness and survival of the
    fittest [1]. In detail, during an evolutionary cycle, the processes of selection, crossover and mutation, known as
    stochastic genetic operators, take place in order to produce the next generation of solutions. The evolution process
    stops when a maximum number of generations or fitness evaluations is reached. Typically, during the evolution
    process, the best individual of the current generation can be inserted into the next one in order to prevent its
    possible disappearance [2].

    References:
        [1] Giovanni Acampora, Angela Chiatto, and Autilia Vitiello, "Training variational quantum circuits through
        genetic algorithms", Proceedings of 2022 IEEE Congress on Evolutionary Computation (CEC), pp. 1–8, 2022.

        [2] Giovanni Acampora, Angela Chiatto, and Autilia Vitiello, "Genetic algorithms as classical optimizer for the
        quantum approximate optimization algorithm", Applied Soft Computing, pp. 110296, Elsevier, 2023.

    Args:
        selection: Selection operator used to select individuals to create the next generation. The selection function
                   is defined as ``sel_function(pop, fitness, *args) -> (sel_pop, sel_fit)``, where ``pop`` and ``fitness``
                   are two arrays of real parameters with (`pop_size`,`n_params`) and (`n_params`,) shape respectively,
                   ``args`` is a tuple of other fixed parameters needed to specify the function, and the output is the
                   resulting subset of individuals and fitness values.
        crossover: Crossover operator used to mate two individuals. The crossover function is defined as
                   ``cx_function(par1, par2, *args) -> (child1, child2)``, where ``par1`` and ``par2`` are two arrays of
                   real parameters with (`n_params`,) shape, ``args`` is a tuple of other fixed parameters needed to
                   specify the function, and the output is the resulting children with the same shape as the parents.
        mutation: Mutation operator used to mutate individuals. The mutation function is defined as
                   ``mut_function(ind, *args) -> mutated_ind``, where ``ind`` is an array of real parameters with
                   (`n_params`,) shape, ``args`` is a tuple of other fixed parameters needed to specify the function,
                   and the output is the mutated individual.
        elitism: If True, the best solution of current population is transferred directly into the next generation.
        cxpb: The probability of mating two individuals.
        mutpb: The probability of mutating an individual.
        kwargs: Additional keyword arguments used to set hyperparameter values of genetic operators.
    """

    def __init__(
            self, selection: Callable, crossover: Callable, mutation: Callable, elitism: bool = True,
            cxpb: float = 0.8, mutpb: float = 1.0, **kwargs):
        self.elitism = elitism
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.kwargs = kwargs

        if selection in SELECTION_ARGS:
            args = {arg: self.kwargs.get(arg) for arg in SELECTION_ARGS[selection] if arg in self.kwargs.keys()}
            self.selection = partial(selection, **args)
        else:
            self.selection = selection

        if crossover in CROSSOVER_ARGS:
            args = {arg: self.kwargs.get(arg) for arg in CROSSOVER_ARGS[crossover] if arg in self.kwargs.keys()}
            self.crossover = partial(crossover, **args)
        else:
            self.crossover = crossover

        if mutation in MUTATION_ARGS:
            args = {arg: self.kwargs.get(arg) for arg in MUTATION_ARGS[mutation] if arg in self.kwargs.keys()}
            self.mutation = partial(mutation, **args)
        else:
            self.mutation = mutation

    def evolve_population(
            self, problem: Problem, population: np.ndarray, fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:

        """
        Evolve the population by means of genetic operators.

        Args:
            problem : :class:`~.Problem` to be solved.
            population: A population of individuals as array of real parameters with (`pop_size`, `n_params`)
                        shape.
            fitness: A set of fitness values associated to the population as array of real values with (`pop_size`,)
                    shape.

        Returns:
            The offspring and fitness values obtained after evolution, and number of fitness evaluations completed
            during the evolution.
        """

        # Select the next generation individuals
        parents, fit_parents = self.selection(population=population, fitness=fitness, k=len(population))

        # Clone the individuals to produce the offspring
        offspring = parents.copy()
        fit_offspring = fit_parents.copy()
        recompute_fitness = np.zeros(len(fitness), dtype=bool)

        # Crossover step
        for i, (child1, child2) in enumerate(zip(offspring[::2], offspring[1::2])):
            if random.random() < self.cxpb:
                child1[:], child2[:] = self.crossover(child1, child2)
                recompute_fitness[i] = recompute_fitness[i + 1] = True

        # Mutation step
        for i, mutant in enumerate(offspring):
            if random.random() < self.mutpb:
                mutant_clone = mutant.copy()
                mutant[:] = self.mutation(mutant)
                if not np.array_equal(mutant_clone, mutant):
                    recompute_fitness[i] = True

        # Check the bounds of the parameters
        for child in offspring:
            child[:] = problem.check_bounds(child)

        # Evaluate the individuals with an invalid fitness
        fit_offspring[recompute_fitness] = np.array(list(map(problem.evaluate_fitness, offspring[recompute_fitness])))
        nfev = len(fit_offspring[recompute_fitness])
        return offspring, fit_offspring, nfev

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

            # Evolve the population based on the genetic operators chosen
            offspring, fit_offspring, nfev = self.evolve_population(problem, population, fitness)
            tot_nfev += nfev

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
