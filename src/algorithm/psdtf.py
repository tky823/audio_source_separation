import numpy as np
import scipy.linalg

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
        is_complex = isinstance(self.target.dtype, np.complex)

        self.is_complex = is_complex

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
            trace = np.trace(V, axis1=0, axis2=1)
            V = V / trace
            H = H * trace[:, np.newaxis]

            self.basis, self.activation = V, H

    def update(self, iteration=100):
        target = self.target

        for idx in range(iteration):
            self.update_once()

            # V: (n_bins, n_bins, n_basis), H: (n_basis, n_frames)
            V, H = self.basis, self.activation
            VH = np.sum(V[:, :, :, np.newaxis] * H[np.newaxis, np.newaxis, :, :], axis=2) # (n_frames, n_bins, n_bins)

            loss = self.criterion(VH.transpose(2, 0, 1), target.transpose(2, 0, 1))
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
    
    def update_once_mm(self):
        X = self.target # (n_bins, n_bins, n_frames)
        V, H = self.basis, self.activation # V: (n_bins, n_bins, n_basis), H: (n_basis, n_frames)
        n_bins, _, n_frames = X.shape
        eps = self.eps

        X = X.transpose(2, 0, 1) # (n_frames, n_bins, n_bins)
        V = V.transpose(2, 0, 1) # (n_basis, n_bins, n_bins)

        if self.is_complex:
            raise NotImplementedError("Not support complex input.")
        # Update activation
        Y = np.sum(V[:, np.newaxis, :, :] * H[:, :, np.newaxis, np.newaxis], axis=0) # (n_frames, n_bins, n_bins)
        Y = _to_symmetric(Y)
        inv_Y = np.linalg.inv(Y + np.eye(n_bins))
        inv_YV = inv_Y[np.newaxis, :, :, :] @ V[:, np.newaxis, :, :] # (n_basis, n_frames, n_bins, n_bins)
        inv_YX = inv_Y @ X # (n_frames, n_bins, n_bins)
        numerator = np.trace(inv_YV @ inv_YX[np.newaxis, :, :, :], axis1=-2, axis2=-1).real # (n_basis, n_frames)
        denominator = np.trace(inv_YV, axis1=-2, axis2=-1).real # (n_basis, n_frames)
        denominator[denominator < eps] = eps
        H = H * np.sqrt(numerator / denominator) # (n_basis, n_frames)

        # Update basis
        Y = np.sum(V[:, np.newaxis, :, :] * H[:, :, np.newaxis, np.newaxis], axis=0) # (n_frames, n_bins, n_bins)
        inv_Y = np.linalg.inv(Y + np.eye(n_bins))

        YXY = inv_Y @ X @ inv_Y # (n_frames, n_bins, n_bins)
        YXY = _to_symmetric(YXY)
        P = np.sum(H[:, :, np.newaxis, np.newaxis] * inv_Y[np.newaxis, :, :, :], axis=1) # (n_basis, n_bins, n_bins)
        Q = np.sum(H[:, :, np.newaxis, np.newaxis] * YXY[np.newaxis, :, :, :], axis=1) # (n_basis, n_bins, n_bins)
        P, Q = _to_symmetric(P), _to_symmetric(Q)

        L = np.linalg.cholesky(Q + np.eye(n_bins)).real # (n_basis, n_bins, n_bins)
        LVPVL = L.transpose(0, 2, 1) @ V @ P @ V @ L # (n_basis, n_bins, n_bins)

        for m in range(len(LVPVL)):
            LVPVL[m] = scipy.linalg.sqrtm(LVPVL[m]).real
        LVPVL = np.linalg.inv(LVPVL + np.eye(n_bins))

        V = V @ L @ LVPVL @ L.transpose(0, 2, 1) @ V
        V = _to_symmetric(V)

        self.basis, self.activation = V.transpose(1, 2, 0), H

def _to_symmetric(X, axis1=-2, axis2=-1):
    X = (X + X.swapaxes(axis1, axis2)) / 2
    return X

def _to_Hermite(X, axis1=-2, axis2=-1):
    X = (X + X.swapaxes(axis1, axis2).conj()) / 2
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