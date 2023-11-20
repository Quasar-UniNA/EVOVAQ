import numpy as np
import random
from evovaq.tools.support import compute_statistics, print_info, Logbook, set_progress_bar, BestIndividualTracker, \
    FinalResult, Particle
from evovaq.tools.operators import sel_best
from typing import Union
from evovaq.problem import Problem


class PSO(object):
    """
    Particle Swarm Optimization (PSO) is an optimization method based on a swarm of candidate solutions, named particles,
    moving in the search space according to appropriate position and velocity equation [1]. Starting from a swarm of
    particles with random positions and velocities, each particle’s velocity is updated by combining its own best
    position (pbest) and the global best position (gbest) ever found in the search space with some random perturbations
    influenced by two hyperparameters, denoted as `phi1` and `phi2`. At this point, each particle’s position is updated
    by adding its resulting velocity to the current position. The final goal is to move the whole swarm close to an
    optimal position in the search space.
    In `evovaq`, PSO with inertia weight (`inertia_weight`) described in [2] is implemented.

    References:
        [1] Giovanni Acampora, Angela Chiatto, and Autilia Vitiello, "A comparison of evolutionary algorithms for
        training variational quantum classifiers", Proceedings of 2023 IEEE Congress on Evolutionary Computation (CEC),
        pp. 1–8, 2023.

        [2] Poli R., Kennedy J., & Blackwell T., "Particle swarm optimization: An overview", Swarm intelligence, vol. 1,
        pp. 33-57, 2007.

    Args:
        vmin: Lower value(s) of the velocity. If None, no limits are considered.
        vmax: Upper value(s) of the velocity. If None, no limits are considered.
        inertia_weight: Inertia weight.
        phi1: Acceleration coefficient determining the magnitude of the random force in the direction of personal
              best solution.
        phi2: Acceleration coefficient determining the magnitude of the random force in the direction of global best
              solution.
    """
    def __init__(self, vmin: Union[float, np.ndarray, None] = None, vmax: Union[float, np.ndarray, None] = None,
                 inertia_weight: float = 0.7298, phi1: float = 1.49618, phi2: float = 1.49618):
        self.vmin = vmin
        self.vmax = vmax
        self.inertia_weight = inertia_weight
        self.phi1 = phi1
        self.phi2 = phi2

    @staticmethod
    def initialize_swarm(population: np.ndarray, fitness: np.ndarray) -> list[Particle]:
        """
        Initialize the swarm.

        Args:
            population: A population of individuals as array of real parameters with (`pop_size`, `n_params`)
                        shape.
            fitness: A set of fitness values associated to the population as array of real values with (`pop_size`,)
                     shape.

        Returns:
            List of :class:`~.Particle` elements.
        """
        swarm = []
        for position, fit in zip(population, fitness):
            velocity = np.random.uniform(-1, 1, len(position))
            part = Particle(position, velocity, fit)
            swarm.append(part)
        return swarm

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

        swarm = self.initialize_swarm(population, fitness)

        # Begin the generational process
        for gen in range(1, max_gen + 1):

            # Check the stopping criterion
            if max_nfev is not None and tot_nfev >= max_nfev:
                pbar.close()
                break

            gbest, fit_gbest = sel_best(population, fitness, 1)
            offspring = population.copy()
            fit_offspring = fitness.copy()

            for idx, part in enumerate(swarm):
                part.update_velocity(gbest[0], self.inertia_weight, self.phi1, self.phi2, self.vmin, self.vmax)
                part.update_position(problem.param_bounds)
                new_fitness = problem.evaluate_fitness(part.position)
                part.update_fitness(new_fitness)
                part.update_pbest()
                offspring[idx] = part.position
                fit_offspring[idx] = part.fitness

            nfev = len(offspring)
            tot_nfev += len(offspring)

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
