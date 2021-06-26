import numpy as np

EPS=1e-12

def solve_Riccati(A, B, eps=EPS):
    """
    Args:
        A (*, M, M): 
        B (*, M, M): 
    """
    I, N, _, M = A.shape
    O = np.zeros_like(A)
    L = np.block([
        [O, -A],
        [-B, O]
    ])

    w, v = np.linalg.eig(L)
    w = np.real(w) # (2049, 3, 4), (2049, 3, 4, 4)
    order = np.argsort(w, axis=2)[:, :, :M] # (2049, 3, 2)

    # TODO: sort
    FG = np.zeros((*A.shape[:-2], 2 * M, M), dtype=np.complex128)
    for i in range(I):
        for n in range(N):
            v_in = v[i, n, :, :]
            order_in = order[i, n, :]
            FG[i, n, :, :] = v_in[:, order_in]

    F, G = np.split(FG, M, axis=-2)
    H = G @ np.linalg.inv(F)
    H = (H + H.transpose(0, 1, 3, 2).conj()) / 2

    return H