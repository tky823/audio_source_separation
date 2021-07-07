import numpy as np

EPS=1e-12

def solve_Riccati(A, B):
    """
    Args:
        A (*, M, M): 
        B (*, M, M): 
    """
    M = A.shape[-1]
    O = np.zeros_like(A)
    L = np.block([
        [O, -A],
        [-B, O]
    ])

    w, v = np.linalg.eig(L)
    v_transposed = v.swapaxes(-1, -2)
    w = np.real(w)
    order = np.argsort(w, axis=2)[..., :M]

    FG = _parallel_sort(v_transposed, order=order, axis=2)
    FG = FG.swapaxes(-1, -2)

    F, G = np.split(FG, 2, axis=-2)
    H = G @ np.linalg.inv(F)
    H = (H + H.swapaxes(-1, -2).conj()) / 2

    return H

def _parallel_sort(x, order, axis=-2):
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