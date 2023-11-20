import random
import numpy as np
from evovaq.tools.support import print_info, Logbook, FinalResult, set_progress_bar
from evovaq.problem import Problem
from typing import Callable, Union


class HC(object):
    """
    Hill Climbing (HC) is a local search optimization method. In contrast to evolutionary algorithms, HC starts from a
    single initial solution and considers its neighborhood to identify a better solution. Thus, the search takes place
    by iteratively trying to replace the current solution with a better neighbour. There are several variants of Hill
    Climbing search depending on how to find the next solution.
    In `evovaq`, stochastic variant of HC described in [1] is implemented due to its successful application in several
    domains.

    References:
        [1] Giovanni Acampora, Angela Chiatto, and Autilia Vitiello, "Training circuit-based quantum classifiers through
        memetic algorithms", Pattern Recognition Letters, Elsevier, 2023.

    Args:
        generate_neighbour: A way of generating a neighbour of a current solution. This function is defined as
                            ``generate_neighbour(prob, ind, *args) -> neighbour``, where ``prob`` is :class:`~.Problem`
                            to be solved; ``ind`` is an array of real parameters with (`n_params`,); ``args`` is a tuple
                            of other fixed parameters needed to specify the function; and the output is the neighbour
                            with the same shape as ``ind`` and ``fitness``.
    """
    def __init__(self, generate_neighbour: Callable):
        self.generate_neighbour = generate_neighbour

    def stochastic_var(
            self, problem: Problem, current_solution: np.ndarray, current_fitness: float) -> tuple[np.ndarray, float, int]:
        """
        Stochastic variant.

        Args:
            problem: :class:`~.Problem` to be solved.
            current_solution: A possible solution as array of real parameters with (`n_params`,) shape.
            current_fitness: The corresponding fitness value.

        Returns:
            The improved current solution and fitness value, and the number of fitness evaluations completed during one
            iter.
        """
        neighbour = self.generate_neighbour(problem, current_solution)
        fit_neighbour = problem.evaluate_fitness(neighbour)
        if fit_neighbour < current_fitness:
            current_solution = neighbour.copy()
            current_fitness = fit_neighbour
        return current_solution, current_fitness, 1

    def optimize(
            self, problem: Problem, init_point: Union[None, np.ndarray] = None, max_nfev: Union[int, None] = None,
            max_iter: int = 1000, n_run: int = 1, seed: Union[int, float, str, None] = None,
            verbose: bool = True) -> FinalResult:
        """
        Optimize the parameters of the problem to be solved.

        Args:
            problem: :class:`~.Problem` to be solved.
            init_point: Initial possible solution as array of real parameters with (`n_params`,) shape. If None, the
                        initial point is randomly generated from `param_bounds`.
            max_nfev: Maximum number of objective function evaluations used as stopping criterion. If None, the maximum
                      number of iterations `max_iter` is considered as stopping criterion.
            max_iter: Maximum number of iterations used as stopping criterion. If `max_nfev` is not None, this is
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

        # Create and initialize the starting solution
        if init_point is None:
            current_solution = problem.generate_individual()
            current_fitness = problem.evaluate_fitness(current_solution)
            nfev = 1
            tot_nfev = 1
        else:
            current_solution = init_point[:].copy()
            current_fitness = problem.evaluate_fitness(current_solution)
            nfev = 1
            tot_nfev = 1

        # Set the progress bar considering the stopping criterion
        pbar = set_progress_bar(max_iter, max_nfev, 'Iterations', 'iter')
        if max_nfev is not None:
            pbar.update(nfev)

        # Set the logbook
        lg = Logbook()
        lg.record(iter=0, nfev=nfev, fitness=current_fitness)

        # Print the evolution info
        if verbose:
            print_info(n_run=n_run, iter=0, nfev=nfev, fitness=current_fitness, header=True)

        for it in range(1, max_iter + 1):

            if max_nfev is not None and tot_nfev >= max_nfev:
                pbar.close()
                break

            current_solution[:], current_fitness, nfev = self.variant(problem, current_solution, current_fitness)
            tot_nfev += nfev

            # Record info in the logbook
            lg.record(iter=it, nfev=nfev, fitness=current_fitness)

            # Print the evolution info
            if verbose:
                print_info(n_run=n_run, iter=it, nfev=nfev, fitness=current_fitness)

            # Update the progress bar
            if max_nfev is not None:
                pbar.update(nfev)
            else:
                pbar.update()

        res = FinalResult(x=current_solution, fun=current_fitness, nfev=tot_nfev, iter=it,
                          log=lg.get_log())
        return res
