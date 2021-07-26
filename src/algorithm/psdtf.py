import numpy as np

from utils.utils_linalg import to_PSD
from criterion.divergence import logdet_divergence

"""
Positive Semidefinite Tensor Factorization
"""

EPS = 1e-12

class PSDTFbase:
    def __init__(self, n_basis=2, normalize=True, eps=EPS):
        self.n_basis = n_basis
        self.normalize = normalize
        self.loss = []

        self.eps = eps

    def __call__(self, target, iteration=100, **kwargs):
        """
        Args:
            target <np.ndarry>: (n_bins, n_bins, n_frames)
            iteration <int>: Default: 100
        """
        self.target = target

        self._reset(**kwargs)

        self.update(iteration=iteration)

        V, H = self.basis, self.activation

        return V.copy(), H.copy()

    def _reset(self, **kwargs):
        assert self.target is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        
        n_basis = self.n_basis
        n_bins, _, n_frames = self.target.shape

        self.is_complex = np.iscomplexobj(self.target)

        if not hasattr(self, 'basis'):
            V = np.random.rand(n_basis, n_bins) # should be positive semi-definite
            eye = np.eye(n_bins, dtype=self.target.dtype)
            eye = np.tile(eye, reps=(n_basis, 1, 1))
            V = V[:, :, np.newaxis] * eye
            self.basis = V.transpose(1, 2, 0)
        else:
            self.basis = self.basis.copy()
        
        if not hasattr(self, 'activation'):
            self.activation = np.random.rand(n_basis, n_frames)
        else:
            self.activation = self.activation.copy()
        
        if self.normalize:
            V, H = self.basis, self.activation
            trace = np.trace(V, axis1=0, axis2=1).real
            V = V / trace
            H = H * trace[:, np.newaxis]

            self.basis, self.activation = V, H

    def update(self, iteration=100):
        X = self.target.transpose(2, 0, 1)

        eps = self.eps

        for idx in range(iteration):
            self.update_once()

            # V: (n_bins, n_bins, n_basis), H: (n_basis, n_frames)
            V, H = self.basis, self.activation
            VH = np.sum(V[:, :, :, np.newaxis] * H[np.newaxis, np.newaxis, :, :], axis=2) # (n_frames, n_bins, n_bins)
            VH = to_PSD(VH.transpose(2, 0, 1), eps=eps)

            loss = self.criterion(VH, X)
            self.loss.append(loss.sum())

    def update_once(self):
        raise NotImplementedError("Implement `update_once` method.")

class LDPSDTF(PSDTFbase):
    """
    Reference: "Beyond NMF: Time-Domain Audio Source Separation without Phase Reconstruction"
    See https://archives.ismir.net/ismir2013/paper/000032.pdf
    """
    def __init__(self, n_basis=2, algorithm='mm', normalize=True, eps=EPS):
        super().__init__(n_basis=n_basis, normalize=normalize, eps=eps)

        self.algorithm = algorithm
        self.criterion = logdet_divergence
    
    def update_once(self):
        if self.algorithm == 'mm':
            self.update_once_mm()
        elif self.algorithm == 'em':
            raise NotImplementedError
            # self.update_once_em()
        else:
            raise ValueError("Not support {} based update.".format(self.algorithm))
        
        if self.normalize:
            V, H = self.basis, self.activation
            trace = np.trace(V, axis1=0, axis2=1).real
            V = V / trace
            H = H * trace[:, np.newaxis]

            self.basis, self.activation = V, H
    
    def update_once_mm(self):
        self.update_basis_mm()
        self.update_activation_mm()

    def update_basis_mm(self):
        X = self.target # (n_bins, n_bins, n_frames)
        V, H = self.basis, self.activation # V: (n_bins, n_bins, n_basis), H: (n_basis, n_frames)
        eps = self.eps

        X = X.transpose(2, 0, 1) # (n_frames, n_bins, n_bins)
        V = V.transpose(2, 0, 1) # (n_basis, n_bins, n_bins)

        Y = np.sum(V[:, np.newaxis, :, :] * H[:, :, np.newaxis, np.newaxis], axis=0) # (n_frames, n_bins, n_bins)
        Y = to_PSD(Y, eps=eps)
        inv_Y = np.linalg.inv(Y)
        inv_Y = to_PSD(inv_Y, eps=eps)

        YXY = inv_Y @ X @ inv_Y # (n_frames, n_bins, n_bins)
        YXY = to_PSD(YXY, eps=eps)
        P = np.sum(H[:, :, np.newaxis, np.newaxis] * inv_Y[np.newaxis, :, :, :], axis=1) # (n_basis, n_bins, n_bins)
        Q = np.sum(H[:, :, np.newaxis, np.newaxis] * YXY[np.newaxis, :, :, :], axis=1) # (n_basis, n_bins, n_bins)
        P, Q = to_PSD(P, eps=eps), to_PSD(Q, eps=eps)
        
        L = np.linalg.cholesky(Q).real # (n_basis, n_bins, n_bins)
        LVPVL = L.transpose(0, 2, 1) @ V @ P @ V @ L # (n_basis, n_bins, n_bins)        
        LVPVL = to_PSD(LVPVL, eps=eps)
        
        w, v = np.linalg.eigh(LVPVL)
        w[w < 0] = 0
        w = np.sqrt(w)
        w = w[..., np.newaxis] * np.eye(w.shape[-1])
        LVPVL = v @ w @ np.linalg.inv(v)
        LVPVL = to_PSD(LVPVL, eps=eps)
        LVPVL = np.linalg.inv(LVPVL)

        V = V @ L @ LVPVL @ L.transpose(0, 2, 1) @ V
        V = to_PSD(V, eps=eps)

        self.basis, self.activation = V.transpose(1, 2, 0), H

    def update_activation_mm(self):
        X = self.target # (n_bins, n_bins, n_frames)
        V, H = self.basis, self.activation # V: (n_bins, n_bins, n_basis), H: (n_basis, n_frames)
        eps = self.eps

        X = X.transpose(2, 0, 1) # (n_frames, n_bins, n_bins)
        V = V.transpose(2, 0, 1) # (n_basis, n_bins, n_bins)

        Y = np.sum(V[:, np.newaxis, :, :] * H[:, :, np.newaxis, np.newaxis], axis=0) # (n_frames, n_bins, n_bins)
        Y = to_PSD(Y, eps=eps)
        inv_Y = np.linalg.inv(Y)
        inv_Y = to_PSD(inv_Y, eps=eps)
        inv_YV = inv_Y[np.newaxis, :, :, :] @ V[:, np.newaxis, :, :] # (n_basis, n_frames, n_bins, n_bins)
        inv_YX = inv_Y @ X # (n_frames, n_bins, n_bins)
        numerator = np.trace(inv_YV @ inv_YX[np.newaxis, :, :, :], axis1=-2, axis2=-1).real # (n_basis, n_frames)
        denominator = np.trace(inv_YV, axis1=-2, axis2=-1).real # (n_basis, n_frames)
        numerator[numerator < 0] = 0
        denominator[denominator < eps] = eps
        H = H * np.sqrt(numerator / denominator) # (n_basis, n_frames)

        self.basis, self.activation = V.transpose(1, 2, 0), H

def _to_symmetric(X, axis1=-2, axis2=-1):
    X = (X + X.swapaxes(axis1, axis2)) / 2
    return X

def nonparallel_inv(X, use_cholesky=True):
    mat_size, _mat_size = X.shape[-2:]

    assert mat_size == _mat_size, "Invalid shape"
    batch_size = X.shape[:-2]
    n_batches = np.prod(batch_size)

    X = X.reshape(n_batches, mat_size, mat_size)
    inv_X = []

    for n in range(n_batches):
        X_n = X[n]
        if use_cholesky:
            X_n = _to_symmetric(X_n)
            cholesky = np.linalg.cholesky(X_n).real # cholesky @ cholesky.transpose(0,2,1) == covariance
            inv_cholesky = np.linalg.inv(cholesky)
            inv_X_n = inv_cholesky.transpose(1, 0) @ inv_cholesky
        else:
            inv_X_n = np.linalg.inv(X_n)
    
        inv_X.append(inv_X_n)
    
    inv_X = np.array(inv_X)
    inv_X.reshape(*batch_size, mat_size, mat_size)

    return inv_X