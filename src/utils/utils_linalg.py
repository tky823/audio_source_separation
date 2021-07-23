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

    eigvals = np.linalg.eigvals(X).real
    delta = np.min(eigvals, axis=-1)
    delta = np.minimum(delta, 0)
    trace = np.trace(X, axis1=axis1, axis2=axis2).real

    X = X - delta[..., np.newaxis, np.newaxis] * np.eye(shape[-1]) + eps * trace[..., np.newaxis, np.newaxis] * np.eye(shape[-1])
    
    return X