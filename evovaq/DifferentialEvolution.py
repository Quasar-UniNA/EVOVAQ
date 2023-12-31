import random
import numpy as np
from evovaq.tools.support import compute_statistics, print_info, Logbook, set_progress_bar, BestIndividualTracker, \
    FinalResult
from typing import Union
from evovaq.problem import Problem


class DE(object):
    """
    Differential Evolution (DE) is a parallel direct search method designed for minimizing possibly nonlinear and
    non-differentiable continuous space functions [1]. Starting from an initial random population, the evolutionary
    cycle that creates a new population for the next generation consists of an appropriate perturbation of each individual.
    In detail, for each target individual, a mutant individual is generated by adding a weighted difference between two or four
    randomly chosen individuals to a third individual selected as best or randomly. At this point, a uniform crossover
    is applied between the target and the mutant to obtain the trial individual. Lastly, the trial individual is compared
    to the target individual to determine the best member of the next generation [2].

    References:
        [1] R. Storn and K. Price, “Differential evolution–a simple and efficient heuristic for global optimization over
        continuous spaces,” Journal of global optimization, vol. 11, no. 4, pp. 341–359, 1997.

        [2] Giovanni Acampora, Angela Chiatto, and Autilia Vitiello, "A comparison of evolutionary algorithms for
        training variational quantum classifiers", Proceedings of 2023 IEEE Congress on Evolutionary Computation (CEC),
        pp. 1–8, 2023.

    Args:
        variant: The variants are denoted as <selection>/<n_diffs>/<crossover>, where <selection> is `best` or `rand`,
                 <n_diffs> is `1` or `2`, and <crossover> is `bin` or `exp`.
        differential_weight: Differential weight is a hyperparameter that controls the amplification of the individuals
                             difference used to create the mutant. If float, it must be in the range [0, 2].
                             If ``(min, max)`` tuple, dithering is employed. Dithering randomly changes the
                             differential weight in each generation so that it can help speed convergence significantly.
        CR: The probability that a parameter of the trial individual is replaced with the corresponding one of the
            mutant individual.
    """

    def __init__(self, variant: str = 'best/1/bin', differential_weight: Union[float, tuple] = 0.8, CR: float = 0.9):
        self.variant = variant
        self.differential_weight = differential_weight
        self.CR = CR

    def mut_de(self, population: np.ndarray, fitness: np.ndarray, target_idx: int, F: float) -> np.ndarray:
        """
        Create the mutant individual.

        Args:
            population: A population of possible solutions as array of real parameters with (`pop_size`, `n_params`)
                        shape.
            fitness: A set of fitness values associated to the population as array of real values with (`pop_size`,)
                     shape.
            target_idx: Index of the target individual in the population.
            F: Differential weight.

        Returns:
            Mutant individual as array of real parameters with (`n_params`,) shape.
        """
        indices = np.arange(len(population))
        if self.variant.split('/')[0].lower() == 'rand':
            mutant_idx = np.random.choice(indices[indices != target_idx])
        elif self.variant.split('/')[0].lower() == 'best':
            mutant_idx = np.argsort(fitness)[0]
        else:
            raise ValueError("Please select a valid mutation strategy, i.e. 'best' or 'rand'")

        mutant = population[mutant_idx].copy()
        diff_ind_idx = np.random.choice(indices[~np.isin(indices, [target_idx, mutant_idx])],
                                        size=int(self.variant.split('/')[1]) * 2, replace=False)
        if self.variant.split('/')[1] == '1':
            mutated = mutant + F * (population[diff_ind_idx[0]].copy() - population[diff_ind_idx[1]].copy())
        elif self.variant.split('/')[1] == '2':
            mutated = mutant + F * (population[diff_ind_idx[0]].copy() + population[diff_ind_idx[1]].copy() -
                                    population[diff_ind_idx[2]].copy() - population[diff_ind_idx[3]].copy())
        else:
            raise ValueError("Please select a valid mutation strategy, i.e. '1' or '2'")
        return mutated

    def cx_de(self, target: np.ndarray, mutated: np.ndarray) -> np.ndarray:
        """
        Mate the target and mutant individual.

        Args:
            target: Target individual as array of real parameters with (`n_params`,) shape.
            mutated: Mutant individual as array of real parameters with (`n_params`,) shape.

        Returns:
            Trial individual as array of real parameters with (`n_params`,) shape.
        """
        variant_type = self.variant.split('/')[-1].lower()
        trial = None
        if variant_type == 'bin':
            R = np.random.choice(len(target))
            pr = np.random.random(size=len(target))
            trial = np.array([mutated[i] if pr[i] < self.CR or i == R else target[i] for i in range(len(target))])
        elif variant_type == 'exp':
            print("'exp' crossover is under construction")
        else:
            raise ValueError("Please select a valid crossover strategy, i.e. 'bin' or 'exp'")
        return trial

    def evolve_population(
            self, problem: Problem, population: np.ndarray, fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Evolve the population by means of stochastic operators.

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

        # Check the dithering
        if isinstance(self.differential_weight, tuple):
            _min, _max = sorted(self.differential_weight)
            F = random.uniform(_min, _max)
        elif isinstance(self.differential_weight, float):
            F = self.differential_weight
        else:
            raise ValueError('The differential_weight must be a float in '
                             '(0, 2), or specified as a tuple(min, max)'
                             ' where min < max and min, max are in (0, 2).')

        # Clone the individuals to produce the offspring
        offspring = population.copy()
        fit_offspring = fitness.copy()

        for target_idx, child in enumerate(offspring):
            # Mutation step
            mutated = self.mut_de(population, fitness, target_idx, F)

            # Crossover step
            trial = self.cx_de(child, mutated)

            # Check the bounds of the parameters
            trial[:] = problem.check_bounds(trial)

            # Selection step
            fit_trial = problem.evaluate_fitness(trial)
            if fit_trial < fit_offspring[target_idx]:
                child[:] = trial
                fit_offspring[target_idx] = fit_trial
        nfev = len(offspring)
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

            offspring, fit_offspring, nfev = self.evolve_population(problem, population, fitness)
            tot_nfev += nfev

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
