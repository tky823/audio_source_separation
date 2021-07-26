import numpy as np

from utils.utils_linalg import parallel_sort

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

    FG = parallel_sort(v_transposed, order=order, axis=2)
    FG = FG.swapaxes(-1, -2)

    F, G = np.split(FG, 2, axis=-2)
    H = G @ np.linalg.inv(F)
    H = (H + H.swapaxes(-1, -2).conj()) / 2

    return H