import numpy as np
from typing import Union, Callable


class Problem:
    """
    Class used to define the minimization problem to be solved.

    Args:
        n_params: Number of real parameters to be optimized.
        param_bounds: Parameter bounds expressed as a tuple (`min`, `max`) or as a list of these tuples of `n_params`
                      size.
        obj_function: The objective function to be minimized defined as ``obj_function(params, *args) -> float`` ,
                       where ``params`` is an array with (`n_params`,) shape and ``args`` is a tuple of other fixed
                       parameters needed to specify the function.
    """

    def __init__(self, n_params: int, param_bounds: Union[tuple, list[tuple]], obj_function: Callable):
        self.n_params = n_params
        self.param_bounds = param_bounds
        self.obj_function = obj_function

    def generate_individual(self) -> np.ndarray:
        """
        Generate a random possible solution, called individual.

        Returns:
            A possible solution randomly generated from `param_bounds`.
        """
        if isinstance(self.param_bounds, tuple):
            _min, _max = self.param_bounds
            individual = np.random.uniform(low=_min, high=_max, size=self.n_params)
        elif isinstance(self.param_bounds, list):
            _min = [self.param_bounds[_][0] for _ in range(len(self.param_bounds))]
            _max = [self.param_bounds[_][1] for _ in range(len(self.param_bounds))]
            individual = np.random.uniform(low=_min, high=_max, size=self.n_params)
            if not len(individual) == self.n_params:
                raise ValueError(
                    "Please insert the bounds as a tuple (min, max) or a list of tuples of length n_params")
        else:
            raise ValueError("Please insert the bounds as a tuple (min, max) or a list of tuples of length n_params")
        return individual

    def generate_random_pop(self, pop_size: int) -> np.ndarray:
        """
        Generate a population of possible random solutions.

        Args:
            pop_size: Population size.

        Returns:
            A set of possible solutions randomly generated from `param_bounds`.
        """
        if isinstance(self.param_bounds, tuple):
            _min, _max = self.param_bounds
            population = np.random.uniform(low=_min, high=_max, size=(pop_size, self.n_params))
        elif isinstance(self.param_bounds, list):
            _min = [self.param_bounds[_][0] for _ in range(len(self.param_bounds))]
            _max = [self.param_bounds[_][1] for _ in range(len(self.param_bounds))]
            population = np.random.uniform(low=_min, high=_max, size=(pop_size, self.n_params))
            if not len(population[0]) == self.n_params:
                raise ValueError(
                    "Please insert the bounds as a tuple (min, max) or a list of tuples of length n_params")
        else:
            raise ValueError(
                "Please insert the bounds as a tuple (min, max) or a list of tuples of length n_params")
        return population

    def evaluate_fitness(self, params: np.ndarray) -> float:
        """
        Evaluate the fitness function of the given parameters.

        Args:
            params: A possible solution as array of real parameters with (`n_params`,) shape.

        Returns:
            Value of the objective function.
        """
        return self.obj_function(params)

    def check_bounds(self, params: np.ndarray) -> np.ndarray:
        """
        Check if the solution `params` satisfies the parameters bounds set in `param_bounds`.

        Args:
            params: A possible solution as array of real parameters with (`n_params`,) shape.

        Returns:
            A possible solution with clipped values according to `param_bounds`.
        """
        if isinstance(self.param_bounds, tuple):
            _min, _max = self.param_bounds
            params[:] = np.clip(params, _min, _max)
        elif isinstance(self.param_bounds, list):
            _min = [self.param_bounds[_][0] for _ in range(len(self.param_bounds))]
            _max = [self.param_bounds[_][1] for _ in range(len(self.param_bounds))]
            params[:] = np.clip(params, _min, _max)
        return params
