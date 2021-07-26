import numpy as np

EPS = 1e-12

def to_Hermite(X, axis1=-2, axis2=-1):
    X = (X + X.swapaxes(axis1, axis2).conj()) / 2
    return X

def to_PSD(X, axis1=-2, axis2=-1, eps=EPS):
    shape = X.shape
    n_dims = len(shape)
    if axis1 < 0:
        axis1 = n_dims + axis1
    if axis2 < 0:
        axis2 = n_dims + axis2
    
    assert axis1 == n_dims - 2 and axis2 == n_dims - 1, "`axis1` == -2 and `axis2` == -1"

    if np.iscomplexobj(X):
        X = (X + X.swapaxes(axis1, axis2).conj()) / 2
    else:
        X = (X + X.swapaxes(axis1, axis2)) / 2

    eigvals = np.linalg.eigvalsh(X)
    delta = np.min(eigvals, axis=-1)
    delta = np.minimum(delta, 0)
    trace = np.trace(X, axis1=axis1, axis2=axis2).real

    X = X - delta[..., np.newaxis, np.newaxis] * np.eye(shape[-1]) + eps * trace[..., np.newaxis, np.newaxis] * np.eye(shape[-1])
    
    return X

def parallel_sort(x, order, axis=-2):
    """
    Args:
        x: (*, n_elements, *)
        order: (*, order_elements)
        axis <int>
    Returns:
        x_sorted: (*, n_elements, *)
    """
    repeats = np.prod(x.shape[:axis])
    n_elements = x.shape[axis]
    tensor_shape = x.shape[axis+1:]
    order_elements = order.shape[-1]

    x_flatten = x.reshape(-1, *tensor_shape)
    order_flatten = order.reshape(-1)
    tmp = n_elements * np.arange(repeats)
    shift = np.repeat(tmp, order_elements)
    x_sorted = x_flatten[order_flatten + shift]
    x_sorted = x_sorted.reshape(*x.shape[:axis], order_elements, *tensor_shape)

    return x_sorted