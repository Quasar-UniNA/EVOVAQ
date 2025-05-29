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
        init_range: Initial sampling range used to generate random parameter values before optimization.
                    It can be a single tuple (`min`, `max`) applied uniformly, or a list of such tuples of length
                    `n_params` for individual ranges. If set to `None`, the initial range is inherited from
                    `param_bounds`. If any parameter bound contains `None`, the corresponding initial range
                    defaults to `(-1, 1)` for that parameter.
    """

    def __init__(self, n_params: int, param_bounds: Union[tuple, list[tuple]], obj_function: Callable, init_range: Union[tuple, list[tuple], None] = None):
        self.n_params = n_params
        self.obj_function = obj_function

        if isinstance(param_bounds, tuple):
                self.param_bounds = [param_bounds] * self.n_params
        elif isinstance(param_bounds, list):
            if not len(param_bounds) == self.n_params:
                raise ValueError(
                        "Please insert the bounds as a tuple  of the type (min, max), (None, None), (min, None), (None, max), or a list of such tuples of length n_params")
            else:
                self.param_bounds = param_bounds
        else:
            raise ValueError("Please insert the bounds as a tuple or a list of tuples of length n_params")
                
        if isinstance(init_range, tuple):
            self.init_range = [init_range] * self.n_params
        elif isinstance(init_range, list):
            if not len(init_range) == self.n_params:
                raise ValueError(
                    "Please insert the initial range as a tuple (min, max) or a list of such tuples of length n_params")
            else:
                self.init_range = init_range
        elif init_range is None:
            self.init_range = [params if None not in params else (-1, 1) for params in self.param_bounds]
        else:
            raise ValueError("Please insert the initial range as a tuple (min, max) or a list of tuples of length n_params")
        

    def generate_individual(self) -> np.ndarray:
        """
        Generate a random possible solution, called individual.

        Returns:
            A possible solution randomly generated from `param_bounds`.
        """
        _mins, _maxs = zip(*self.init_range)
        individual = np.random.uniform(low=_mins, high=_maxs, size=self.n_params)
        return individual

    def generate_random_pop(self, pop_size: int) -> np.ndarray:
        """
        Generate a population of possible random solutions.

        Args:
            pop_size: Population size.

        Returns:
            A set of possible solutions randomly generated from `param_bounds`.
        """
        _mins, _maxs = zip(*self.init_range)
        population = np.random.uniform(low=_mins, high=_maxs, size=(pop_size, self.n_params))
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
        _mins = [bound[0] if bound[0] is not None else -np.inf for bound in self.param_bounds]
        _maxs = [bound[1] if bound[1] is not None else np.inf for bound in self.param_bounds]
        params[:] = np.clip(params, _mins, _maxs)
        return params
