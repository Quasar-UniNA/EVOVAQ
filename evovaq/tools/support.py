import numpy as np
from tabulate import tabulate
from tqdm import tqdm
from typing import Union


def set_progress_bar(
        max_gen: int, max_nfev: Union[int, None], desc: Union[str, None] = None, unit: Union[str, None] = None) -> tqdm:
    """
    Set the progress bar based on the stopping criterion.

    Args:
        max_gen: Maximum number of generations.
        max_nfev: Maximum number of fitness evaluations.
        desc: Description related to the stopping criterion.
        unit: Unit describing the progress.

    Returns:
        The progress bar.
    """
    if max_nfev is not None:
        pbar = tqdm(total=max_nfev, desc='Fitness Evaluations', unit='nfev', dynamic_ncols=True)
    else:
        desc = desc if desc is not None else 'Generations'
        unit = unit if unit is not None else 'gen'
        pbar = tqdm(total=max_gen, desc=desc, unit=unit, dynamic_ncols=True)
    return pbar


def compute_statistics(fitness: np.ndarray) -> dict:
    """
    Compute the statistics of fitness values.

    Args:
        fitness: Fitness values as an array of real values with (`pop_size`,) shape.

     Returns:
        Dictionary containing the min, max, mean, and standard deviation value.
    """
    stats = {'min': np.min(fitness), 'max': np.max(fitness), 'mean': np.mean(fitness), 'std': np.std(fitness)}
    return stats


def print_info(n_run: int, header: bool = False, **kwargs):
    """
    Print info.

    Args:
        n_run: Independent execution number of the algorithm.
        header: If True, the string indicating the independent execution number is printed.
        kwargs: Information to be printed defined in a dictionary.
    """
    if header:
        print(f'********** Execution #{n_run} **********')
        print(tabulate([kwargs], headers="keys", tablefmt="simple", numalign="left"))
    else:
        formatted_table = [f"{str(item):<{len(k) + 2}}" for k, item in kwargs.items()]
        print("\n" + tabulate([formatted_table], tablefmt="plain"))


class Logbook(dict):
    """
    Class used to store info during the evolution.
    """
    def record(self, **infos):
        """
        Record info.

        Args:
            infos: Info to be stored defined in a dictionary.
        """
        for k, v in infos.items():
            if k in self:
                self[k].append(v)
            else:
                self[k] = [v]

    def get_log(self):
        """
        Get the logbook.
        """
        return self


class BestIndividualTracker:
    """
    Class used to track the best solution ever found during the algorithm execution.
    """
    def __init__(self):
        self.best_ind = None
        self.best_fit = float('inf')

    def update(self, population, fitness):
        """
        Update the tracker.

        Args:
            population: A population of individuals as array of real parameters with (`pop_size`, `n_params`)
                        shape.
            fitness: A set of fitness values associated to the population as array of real values with (`pop_size`,)
                    shape.
        """
        for ind, fit in zip(population, fitness):
            if fit < self.best_fit:
                self.best_ind = ind
                self.best_fit = fit

    def get_best(self):
        """
        Get the best individual ever found.
        """
        return self.best_ind

    def get_best_fit(self):
        """
        Get the best fitness value ever found.
        """
        return self.best_fit


class FinalResult(dict):
    """
    Class used to store the final result.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class Particle:
    """
    Class used to define a particle described by position, velocity and fitness in :class:`~.PSO` algorithm.

    Args:
        position: A possible solution as array of real parameters with (`n_params`,) shape.
        velocity: Velocity as array of real parameters with (`n_params`,) shape.
        fitness: The corresponding fitness value.
    """
    def __init__(self, position: np.ndarray, velocity: np.ndarray, fitness: float):
        self.position = position
        self.velocity = velocity
        self.fitness = fitness
        self.pbest = position.copy()
        self.fit_pbest = fitness

    def update_velocity(
            self, gbest: np.ndarray, inertia_weight: float, phi1: float, phi2: float, vmin: Union[float, np.ndarray, None],
            vmax: Union[float, np.ndarray, None]):
        """
        Update the velocity of the particle according to the equation proposed in [1].

        References:
            [1] Shi, Y., & Eberhart, R. C., "A modified particle swarm optimizer", Proceedings of the IEEE international
            conference on evolutionary computation, pp. 69–73, 1998.

        Args:
            gbest: Global best position ever visited.
            inertia_weight: Inertia weight.
            phi1: Acceleration coefficient determining the magnitude of the random force in the direction of personal
                  best solution.
            phi2: Acceleration coefficient determining the magnitude of the random force in the direction of global best
                  solution.
            vmin: Lower value(s) of the velocity. If None, no limits are considered.
            vmax: Upper value(s) of the velocity. If None, no limits are considered.
        """
        rand1 = np.random.uniform(0, phi1, size=self.position.size)
        rand2 = np.random.uniform(0, phi2, size=self.position.size)
        new_velocity = inertia_weight * self.velocity + rand1 * (self.pbest - self.position) + rand2 * \
                       (gbest - self.position)

        if vmin is not None and vmax is not None:
            new_velocity[:] = np.clip(new_velocity, vmin, vmax)

        self.velocity = new_velocity

    def update_position(self, param_bounds: Union[tuple, list[tuple]]):
        """
        Update the position of the particle by adding the updated velocity.

        Args:
            param_bounds: Parameter bounds expressed as a tuple (`min`, `max`) or as a list of these tuples of `n_params`
                          size.
        """
        new_position = self.velocity + self.position
        if isinstance(param_bounds, tuple):
            _min, _max = param_bounds
            new_position[:] = np.clip(new_position, _min, _max)
        elif isinstance(param_bounds, list):
            _min = [param_bounds[_][0] for _ in range(len(param_bounds))]
            _max = [param_bounds[_][1] for _ in range(len(param_bounds))]
            new_position[:] = np.clip(new_position, _min, _max)
        self.position = new_position

    def update_pbest(self):
        """
        Update the personal best solution so far.
        """
        if self.fitness < self.fit_pbest:
            self.pbest = self.position.copy()
            self.fit_pbest = self.fitness

    def update_fitness(self, new_fitness: float):
        """
        Update the fitness value.

        Args:
            new_fitness: New fitness value computed.
        """
        self.fitness = new_fitness
