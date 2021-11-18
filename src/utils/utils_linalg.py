import warnings

import utils.linalg as backend

EPS = 1e-12

def to_Hermite(X, axis1=-2, axis2=-1):
    warnings.warn("Use utils.linalg.to_Hermite", DeprecationWarning)
    return backend.to_Hermite(X, axis1=axis1, axis2=axis2)

def to_PSD(X, axis1=-2, axis2=-1, eps=EPS):
    warnings.warn("Use utils.linalg.to_PSD", DeprecationWarning)
    return backend.to_PSD(X, axis1=axis1, axis2=axis2, eps=eps)

def parallel_sort(x, order, axis=-2):
    """
    Args:
        x: (*, n_elements, *)
        order: (*, order_elements)
        axis <int>
    Returns:
        x_sorted: (*, n_elements, *)
    """
    warnings.warn("Use utils.linalg.parallel_sort", DeprecationWarning)
    return backend.parallel_sort(x, order, axis=axis)