import numpy as np
from typing import Union


def euclidean(a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> float:
    """
    Euclidean distance between two points.

    Args:
        a: First point.
        b: Second point.

    Returns:
        Euclidean distance.
    """
    return np.linalg.norm(a - b)
