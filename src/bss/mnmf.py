import warnings

import numpy as np

from algorithm.linalg import solve_Riccati

EPS=1e-12
THRESHOLD=1e+12

__metrics__ = ['EUC', 'KL', 'IS']
__authors__ = ['sawada', 'ozerov']

"""
__kwargs_ozerov_mnmf___ = {
    "hoge": None
}

__kwargs_sawada_mnmf___ = {
    "hoge": None
}
"""

class MultichannelNMFbase:
    def __init__(self, n_bases=10, n_sources=None, callbacks=None, eps=EPS):
        """
        Args:
            n_bases: number of bases 
        """
        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None
        self.eps = eps
        self.input = None
        self.n_bases = n_bases
        self.n_sources = n_sources
        self.loss = []
    
    def _reset(self, **kwargs):
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        n_sources = self.n_sources

        X = self.input
        n_channels, n_bins, n_frames = X.shape

        if n_sources is None:
            n_sources = n_channels
        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames
    
    def __call__(self, input, iteration=100, **kwargs):
        """
        Args:
            input (n_channels, n_bins, n_frames)
        Returns:
            output (n_channels, n_bins, n_frames)
        """
        self.input = input

        self._reset(**kwargs)

        loss = self.compute_negative_loglikelihood()    
        self.loss.append(loss)

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self)

        for idx in range(iteration):
            self.update_once()

            loss = self.compute_negative_loglikelihood()
            self.loss.append(loss)

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)
        
        X = input
        output = self.separate(X)

        return output
        
    def update_once(self):
        raise NotImplementedError("Implement 'update_once' function")
    
    def separate(self, input):
        """
        Args:
            input (n_channels, n_bins, n_frames):
        Returns:
            output (n_channels, n_bins, n_frames): 
        """

        return 0
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")

class ISMultichannelNMF(MultichannelNMFbase):
    """
    References:
        Sawada's MNMF: "Multichannel Extensions of Non-Negative Matrix Factorization With Complex-Valued Data"
        Ozerov's MNMF: "Multichannel Nonnegative Matrix Factorization in Convolutive Mixtures for Audio Source Separation"
    See https://ieeexplore.ieee.org/document/6410389 and https://ieeexplore.ieee.org/document/5229304
    """
    def __init__(self, n_bases=10, n_sources=None, normalize=True, callbacks=None, reference_id=0, author='Sawada', eps=EPS):
        """
        Args:
            n_bases
            n_clusters
            n_sources
            normalize
            callbacks <callable> or <list<callable>>: Callback function. Default: None
            reference_id <int>
            author <str>: 'Sawada' or 'Ozerov'
            eps <float>: Machine epsilon
        """
        super().__init__(n_bases=n_bases, n_sources=n_sources, callbacks=callbacks, eps=eps)

        self.normalize = normalize

        self.reference_id = reference_id

        assert author.lower() in __authors__, "Choose from {}".format(__authors__)

        self.author = author
    
    def __call__(self, input, iteration=100, **kwargs):
        """
        Args:
            input (n_channels, n_bins, n_frames)
        Returns:
            output (n_channels, n_bins, n_frames)
        """
        self.input = input

        self._reset(**kwargs)

        loss = self.compute_negative_loglikelihood()    
        self.loss.append(loss)

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self)

        for idx in range(iteration):
            self.update_once()

            loss = self.compute_negative_loglikelihood()
            self.loss.append(loss)

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)
        
        X = self.input
        output = self.separate(X)
        self.estimation = output

        return output
    
    def _reset(self, **kwargs):
        super()._reset(**kwargs)

        n_bases = self.n_bases
        n_sources = self.n_sources
        eps = self.eps

        x = self.input
        n_channels, n_bins, n_frames = x.shape

        if n_sources is None:
            n_sources = n_channels
        
        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        X = x[:, np.newaxis, :, :] * x[np.newaxis, :, :, :].conj()
        self.covariance_input = X.transpose(2, 3, 0, 1)

        if not hasattr(self, 'latent'):
            variance_latent = 1e-2
            Z = np.random.rand(n_sources, n_bases) * variance_latent + 1 / n_sources
            Zsum = Z.sum(axis=0)
            Zsum[Zsum < eps] = eps
            self.latent = Z / Zsum
        else:
            self.latent = self.latent.copy()
        if not hasattr(self, 'spatial'):
            H = np.eye(n_channels)
            self.spatial = np.tile(H, reps=(n_bins, n_sources, 1, 1))
        else:
            self.spatial = self.spatial.copy()
        if not hasattr(self, 'base'):
            self.base = np.random.rand(n_bins, n_bases)
        else:
            self.base = self.base.copy()
        if not hasattr(self, 'activation'):
            self.activation = np.random.rand(n_bases, n_frames)
        else:
            self.activation = self.activation.copy()

        self.estimation = self.separate(x)
    
    def __repr__(self):
        s = "IS-MNMF("
        s += "n_bases={n_bases}"
        if hasattr(self, 'n_sources'):
            s += ", n_sources={n_sources}"
        if hasattr(self, 'n_channels'):
            s += ", n_channels={n_channels}"
        s += ", normalize={normalize}"
        s += ")"

        return s.format(**self.__dict__)

    def separate(self, input):
        """
        Args:
            input (n_channels, n_bins, n_frames):
        Returns:
            output (n_channels, n_bins, n_frames): 
        """
        x = input
        H, Z = self.spatial, self.latent # (n_bins, n_sources, n_channels, n_channels), (n_sources, n_bases)
        T, V = self.base, self.activation # (n_bins, n_bases), (n_bases, n_bins)

        n_channels = self.n_channels
        reference_id = self.reference_id
        eps = self.eps

        HZ = np.sum(H[:, :, np.newaxis, :, :] * Z[np.newaxis, :, :, np.newaxis, np.newaxis], axis=1) # (n_bins, n_bases, n_channels, n_channels)
        TV = T[:, :, np.newaxis] * V[np.newaxis, :, :] # (n_bins, n_bases, n_frames)
        X_hat = np.sum(HZ[:, :, np.newaxis, :, :] * TV[:, :, :, np.newaxis, np.newaxis], axis=1) # (n_bins, n_frames, n_channels, n_channels)
        inv_X_hat = np.linalg.inv(X_hat + eps * np.eye(n_channels))
        HX = H[:, :, np.newaxis, :, :] @ inv_X_hat[:, np.newaxis, :, :, :] # (n_bins, n_sources, n_frames, n_channels, n_channels)
        x = x.transpose(1, 2, 0) # (n_bins, n_frames, n_channels)
        HX = HX.transpose(1, 0, 2, 3, 4) # (n_sources, n_bins, n_frames, n_channels, n_channels)
        HXx = HX @ x[:, :, :, np.newaxis] # (n_sources, n_bins, n_frames, n_channels, 1)
        HXx = HXx[..., 0].transpose(3, 0, 1, 2)

        ZTV = np.sum(Z[:, np.newaxis, :, np.newaxis] * T[np.newaxis, :, :, np.newaxis] * V[np.newaxis, np.newaxis, :, :], axis=2) # (n_sources, n_bins, n_frames)

        y = ZTV * HXx
        
        return y[reference_id]
    
    def update_once(self):
        author = self.author.lower()
        if author == 'sawada':
            self.update_once_sawada()
        elif author == 'ozerov':
            self.update_once_ozerov()
        else:
            raise ValueError("Not support {}'s MNMF.".format(self.author))
    
    def update_once_sawada(self):
        self.update_base()
        self.update_activation()
        self.update_latent()
        self.update_spatial()
    
    def update_once_ozerov(self):
        raise ValueError("Not support {}'s MNMF.".format(self.author))
    
    def update_base(self):
        n_channels = self.n_channels
        eps = self.eps

        X = self.covariance_input # + eps * np.eye(n_channels) # (n_bins, n_frames, n_channels, n_channels)
        H, Z = self.spatial, self.latent # (n_bins, n_sources, n_channels, n_channels), (n_sources, n_bases)
        T, V = self.base, self.activation # (n_bins, n_bases), (n_bases, n_bins)

        X_hat = self.reconstruct_covariance()
        inv_X_hat = np.linalg.inv(X_hat + eps * np.eye(n_channels)) # (n_bins, n_frames, n_channels, n_channels)
        XXX = inv_X_hat @ X @ inv_X_hat # (n_bins, n_frames, n_channels, n_channels)
        
        numerator = np.trace(XXX[:, np.newaxis, :, :, :] @ H[:, :, np.newaxis, :, :], axis1=-2, axis2=-1).real # (n_bins, 1, n_frames, n_channels, n_channels), (n_bins, n_sources, 1, n_channels, n_channels) -> (n_bins, n_sources, n_frames)
        numerator = np.sum(V[np.newaxis, np.newaxis, :, :] * numerator[:, :, np.newaxis, :], axis=3) # (n_bins, n_sources, n_bases)
        numerator = np.sum(Z * numerator, axis=1) # (n_bins, n_bases)
        denominator = np.trace(inv_X_hat[:, np.newaxis, :, :, :] @ H[:, :, np.newaxis, :, :], axis1=-2, axis2=-1).real # (n_bins, 1, n_frames, n_channels, n_channels), (n_bins, n_sources, 1, n_channels, n_channels) -> (n_bins, n_sources, n_frames)
        denominator = np.sum(V[np.newaxis, np.newaxis, :, :] * denominator[:, :, np.newaxis, :], axis=3) # (n_bins, n_sources, n_bases)
        denominator = np.sum(Z * denominator, axis=1) # (n_bins, n_bases)
        denominator[denominator < eps] = eps

        T = T * np.sqrt(numerator / denominator)
        self.base = T

    def update_activation(self):
        n_channels = self.n_channels
        eps = self.eps

        X = self.covariance_input # + eps * np.eye(n_channels) # (n_bins, n_frames, n_channels, n_channels)
        H, Z = self.spatial, self.latent # (n_bins, n_sources, n_channels, n_channels), (n_sources, n_bases)
        T, V = self.base, self.activation # (n_bins, n_bases), (n_bases, n_bins)

        X_hat = self.reconstruct_covariance()
        inv_X_hat = np.linalg.inv(X_hat + eps * np.eye(n_channels)) # (n_bins, n_frames, n_channels, n_channels)
        XXX = inv_X_hat @ X @ inv_X_hat # (n_bins, n_frames, n_channels, n_channels)

        numerator = np.trace(XXX[:, np.newaxis, :, :, :] @ H[:, :, np.newaxis, :, :], axis1=-2, axis2=-1).real # (n_bins, 1, n_frames, n_channels, n_channels), (n_bins, n_sources, 1, n_channels, n_channels) -> (n_bins, n_sources, n_frames)
        numerator = np.sum(T[:, np.newaxis, :, np.newaxis] * numerator[:, :, np.newaxis, :], axis=0) # (n_sources, n_bases, n_frames)
        numerator = np.sum(Z[:, :, np.newaxis] * numerator, axis=0) # (n_bases, n_frames)
        denominator = np.trace(inv_X_hat[:, np.newaxis, :, :, :] @ H[:, :, np.newaxis, :, :], axis1=-2, axis2=-1).real # (n_bins, 1, n_frames, n_channels, n_channels), (n_bins, n_sources, 1, n_channels, n_channels) -> (n_bins, n_sources, n_frames)
        denominator = np.sum(T[:, np.newaxis, :, np.newaxis] * denominator[:, :, np.newaxis, :], axis=0) # (n_sources, n_bases, n_frames)
        denominator = np.sum(Z[:, :, np.newaxis] * denominator, axis=0) # (n_bases, n_frames)
        denominator[denominator < eps] = eps

        V = V * np.sqrt(numerator / denominator)
        self.activation = V
    
    def update_latent(self):
        n_channels = self.n_channels
        eps = self.eps

        X = self.covariance_input # + eps * np.eye(n_channels) # (n_bins, n_frames, n_channels, n_channels)
        H, Z = self.spatial, self.latent # (n_bins, n_sources, n_channels, n_channels), (n_sources, n_bases)
        T, V = self.base, self.activation # (n_bins, n_bases), (n_bases, n_bins)

        X_hat = self.reconstruct_covariance()
        TV = T[:, :, np.newaxis] * V[np.newaxis, :, :] # (n_bins, n_bases, n_frames)
        inv_X_hat = np.linalg.inv(X_hat + eps * np.eye(n_channels)) # (n_bins, n_frames, n_channels, n_channels)
        XXX = inv_X_hat @ X @ inv_X_hat # (n_bins, n_frames, n_channels, n_channels)

        numerator = np.trace(XXX[:, np.newaxis, :, :, :] @ H[:, :, np.newaxis, :, :], axis1=-2, axis2=-1).real # (n_bins, 1, n_frames, n_channels, n_channels), (n_bins, n_sources, 1, n_channels, n_channels) -> (n_bins, n_sources, n_frames)
        numerator = np.sum(TV[:, np.newaxis, :, :] * numerator[:, :, np.newaxis, :], axis=(0, 3)) # (n_sources, n_bases)
        denominator = np.trace(inv_X_hat[:, np.newaxis, :, :, :] @ H[:, :, np.newaxis, :, :], axis1=-2, axis2=-1).real # (n_bins, 1, n_frames, n_channels, n_channels), (n_bins, n_sources, 1, n_channels, n_channels) -> (n_bins, n_sources, n_frames)
        denominator = np.sum(TV[:, np.newaxis, :, :] * denominator[:, :, np.newaxis, :], axis=(0, 3)) # (n_sources, n_bases)
        denominator[denominator < eps] = eps

        Z = Z * np.sqrt(numerator / denominator) # (n_sources, n_bases)
        Zsum = Z.sum(axis=0)
        Zsum[Zsum < eps] = eps
        Z = Z / Zsum

        self.latent = Z
    
    def update_spatial(self):
        n_channels = self.n_channels
        eps = self.eps

        X = self.covariance_input # + eps * np.eye(n_channels) # (n_bins, n_frames, n_channels, n_channels)
        H, Z = self.spatial, self.latent # (n_bins, n_sources, n_channels, n_channels), (n_sources, n_bases)
        T, V = self.base, self.activation # (n_bins, n_bases), (n_bases, n_bins)

        X_hat = self.reconstruct_covariance()
        inv_X_hat = np.linalg.inv(X_hat + eps * np.eye(n_channels)) # (n_bins, n_frames, n_channels, n_channels)
        XXX = inv_X_hat @ X @ inv_X_hat # (n_bins, n_frames, n_channels, n_channels)
        VX = np.sum(V[np.newaxis, :, :, np.newaxis, np.newaxis] * inv_X_hat[:, np.newaxis, :, :, :], axis=2) # (n_bins, bases, n_channels, n_channels)
        VXXX = np.sum(V[np.newaxis, :, :, np.newaxis, np.newaxis] * XXX[:, np.newaxis, :, :, :], axis=2) # (n_bins, n_bases, n_channels, n_channels)
        ZT = Z[np.newaxis, :, :] * T[:, np.newaxis, :] # (n_bins, n_sources, n_bases)
        
        A = np.sum(ZT[:, :, :, np.newaxis, np.newaxis] * VX[:, np.newaxis, :, :, :], axis=2) # (n_bins, n_sources, n_bases), (n_bins, bases, n_channels, n_channels) -> (n_bins, n_sources, n_channels, n_channels)
        ZTVXXX = np.sum(ZT[:, :, :, np.newaxis, np.newaxis] * VXXX[:, np.newaxis, :, :, :], axis=2) # (n_bins, n_sources, n_bases), (n_bins, bases, n_channels, n_channels) -> (n_bins, n_sources, n_channels, n_channels)
        B = H @ ZTVXXX @ H
        H = solve_Riccati(A, B)
        H = H + eps * np.eye(n_channels)

        if self.normalize:
            H = H / np.trace(H, axis1=2, axis2=3)[..., np.newaxis, np.newaxis]
            
        self.spatial = H
    
    def reconstruct_covariance(self):
        H, Z = self.spatial, self.latent # (n_bins, n_sources, n_channels, n_channels), (n_sources, n_bases)
        T, V = self.base, self.activation # (n_bins, n_bases), (n_bases, n_bins)

        HZ = np.sum(H[:, :, np.newaxis, :, :] * Z[np.newaxis, :, :, np.newaxis, np.newaxis], axis=1) # (n_bins, n_bases, n_channels, n_channels)
        TV = T[:, :, np.newaxis] * V[np.newaxis, :, :] # (n_bins, n_bases, n_frames)
        X_hat = np.sum(HZ[:, :, np.newaxis, :, :] * TV[:, :, :, np.newaxis, np.newaxis], axis=1) # (n_bins, n_frames, n_channels, n_channels)
        
        return X_hat

    def compute_negative_loglikelihood(self):
        X = self.covariance_input # (n_bins, n_frames, n_channels, n_channels)
        X_hat = self.reconstruct_covariance()
        
        loss = is_divergence(X_hat, X) # (n_bins, n_frames)
        loss = loss.sum()

        return loss

class tMultichannelNMF(MultichannelNMFbase):
    """
    Reference: "Student's t multichannel nonnegative matrix factorization for blind source separation"
    See https://ieeexplore.ieee.org/document/7602889
    """
    def __init__(self, n_bases=10, n_sources=None, reference_id=0, callbacks=None, eps=EPS):
        """
        Args:
        """
        warnings.warn("in progress", UserWarning)

        super().__init__(n_bases=n_bases, n_sources=n_sources, callbacks=callbacks, eps=eps)

        self.reference_id = reference_id
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")

class FastISMultichannelNMF(MultichannelNMFbase):
    """
    Reference: "Fast Multichannel Source Separation Based on Jointly Diagonalizable Spatial Covariance Matrices"
    """
    def __init__(self, n_bases=10, n_sources=None, partitioning=False, normalize='power', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        """
        Args:
        """
        super().__init__(n_bases=n_bases, n_sources=n_sources, callbacks=callbacks, eps=eps)

        self.partitioning = partitioning
        self.normalize = normalize
        self.reference_id = reference_id

        self.recordable_loss = recordable_loss
        self.threshold = threshold

    def _reset(self, **kwargs):
        super()._reset(**kwargs)
        
        n_bins, n_frames = self.n_bins, self.n_frames
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bases = self.n_bases

        Q = np.tile(np.eye(n_channels, dtype=np.complex128), (n_bins, 1, 1))
        G = np.ones((n_sources, n_bins, n_channels)) * 1e-2
        for m in range(n_channels):
            G[m % n_sources, :, m] = 1
        
        if self.partitioning:
            if not hasattr(self, 'latent'):
                self.latent = np.ones((n_sources, n_bases), dtype=np.float64) / n_sources
            else:
                self.latent = self.latent.copy()
            if not hasattr(self, 'base'):
                self.base = np.random.rand(n_bins, n_bases)
            else:
                self.base = self.base.copy()
            if not hasattr(self, 'activation'):
                self.activation = np.random.rand(n_bases, n_frames)
            else:
                self.activation = self.activation.copy()
        else:
            if not hasattr(self, 'base'):
                self.base = np.random.rand(n_sources, n_bins, n_bases)
            else:
                self.base = self.base.copy()
            if not hasattr(self, 'activation'):
                self.activation = np.random.rand(n_sources, n_bases, n_frames)
            else:
                self.activation = self.activation.copy()
        # TODO: normalize Q
        self.diagonalizer = Q
        self.spatial_covariance = G
    
    def __call__(self, input, iteration=100, **kwargs):
        """
        Args:
            input (n_channels, n_bins, n_frames)
        Returns:
            output (n_channels, n_bins, n_frames)
        """
        self.input = input

        self._reset(**kwargs)

        if self.recordable_loss:
            loss = self.compute_negative_loglikelihood()    
            self.loss.append(loss)
 
        for idx in range(iteration):
            self.update_once()

            if self.recordable_loss:
                loss = self.compute_negative_loglikelihood()
                self.loss.append(loss)

            if self.callbacks is not None:
                self.estimation = self.separate(self.input)
                for callback in self.callbacks:
                    callback(self)
        
        X = input
        output = self.separate(X)
        self.estimation = output

        return output
    
    def __repr__(self):
        s = "FastMNMF("
        s += "n_bases={n_bases}"
        if hasattr(self, 'n_sources'):
            s += ", n_sources={n_sources}"
        if hasattr(self, 'n_channels'):
            s += ", n_channels={n_channels}"
        s += ", partitioning={partitioning}"
        s += ", normalize={normalize}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        eps = self.eps

        self.update_NMF()
        self.update_SCM()
        self.update_diagonalizer()
        if self.normalize:
            # normalize
            if self.normalize == 'power':
                Q = self.diagonalizer
                g = self.spatial_covariance
                W, H = self.base, self.activation

                if self.partitioning:
                    raise ValueError("Not support partitioning function.")
                else:
                    QQ = Q * Q.conj()
                    QQsum = np.real(QQ.sum(axis=2).mean(axis=1)) # (n_bins,)
                    QQsum[QQsum < eps] = eps
                    Q /= np.sqrt(QQsum)[:, np.newaxis, np.newaxis]
                    g /= QQsum[np.newaxis, :, np.newaxis] # (n_sources, n_bins, n_channels)

                    g_sum = g.sum(axis=2)
                    g_sum[g_sum < eps] = eps
                    g /= g_sum[:, :, np.newaxis]
                    W *= g_sum[:, :, np.newaxis]

                    Wsum = W.sum(axis=1)
                    Wsum[Wsum < eps] = eps
                    W /= Wsum[:, np.newaxis]
                    H *= Wsum[:, :, np.newaxis]

                    self.base, self.activation = W, H
                    self.diagonalizer = Q
                    self.spatial_covariance = g
            else:
                raise ValueError("Not support normalization based on {}. Choose 'power'".format(self.normalize))

    def update_NMF(self):
        eps = self.eps
        X = self.input.transpose(1, 2, 0)
        g = self.spatial_covariance
        Q = self.diagonalizer
        W, H = self.base, self.activation

        QX = np.sum(Q[:, np.newaxis, :, :] * X[:, :, np.newaxis, :], axis=3) # (n_bins, n_channels, n_channels) (n_bins, n_frames, n_channels) -> (n_bins, n_frames, n_channels)
        x_tilde = np.abs(QX)**2

        if self.partitioning:
            raise ValueError("Not support partitioning function.")
        else:
            # TODO: check re-calculation of Lambda
            # update W
            Lambda = W @ H
            R = np.sum(Lambda[..., np.newaxis] * g[:, :, np.newaxis], axis=0) # (n_bins, n_frames, n_channels)
            R[R < eps] = eps
            xR = x_tilde / (R ** 2)
            gxR = np.sum(g[:, :, np.newaxis] * xR[np.newaxis], axis=3)
            gR = np.sum(g[:, :, np.newaxis] / R[np.newaxis], axis=3)

            numerator = np.sum(H[:, np.newaxis, :, :] * gxR[:, :, np.newaxis], axis=3)
            denominator = np.sum(H[:, np.newaxis, :, :] * gR[:, :, np.newaxis], axis=3)
            denominator[denominator < eps] = eps
            W = W * np.sqrt(numerator / denominator)

            # update H
            Lambda = W @ H
            R = np.sum(Lambda[...,np.newaxis] * g[:, :, np.newaxis], axis=0) # (n_bins, n_frames, n_channels)
            R[R < eps] = eps
            xR = x_tilde / (R ** 2)
            gxR = np.sum(g[:, :, np.newaxis] * xR[np.newaxis], axis=3)
            gR = np.sum(g[:, :, np.newaxis] / R[np.newaxis], axis=3)

            numerator = np.sum(W[:, :, :, np.newaxis] * gxR[:, :, np.newaxis], axis=1)
            denominator = np.sum(W[:, :, :, np.newaxis] * gR[:, :, np.newaxis], axis=1)
            denominator[denominator < eps] = eps
            H = H * np.sqrt(numerator / denominator)

            self.base, self.activation = W, H
    
    def update_SCM(self):
        eps = self.eps
        g = self.spatial_covariance
        W, H = self.base, self.activation
        X = self.input.transpose(1, 2, 0)
        Q = self.diagonalizer

        if self.partitioning:
            Z = self.latent # (n_sources, n_bases)
            W, H = self.base, self.activation
            ZW = Z[:, np.newaxis, :] * W[np.newaxis, :, :] # (n_sources, n_bins, n_bases)
            Lambda = ZW @ H[np.newaxis, :, :] # (n_sources, n_bins, n_frames)

            raise ValueError("Not support partitioning function.")
        else:
            Lambda = W @ H

        R = np.sum(Lambda[..., np.newaxis] * g[:, :, np.newaxis], axis=0)
        QX = np.sum(Q[:, np.newaxis, :, :] * X[:, :, np.newaxis, :], axis=3) # (n_bins, n_channels, n_channels) (n_bins, n_frames, n_channels) -> (n_bins, n_frames, n_channels)
        x_tilde = np.abs(QX)**2

        R[R < eps] = eps
        xR = x_tilde / (R ** 2)
        
        A = np.sum(Lambda[..., np.newaxis] * xR[np.newaxis], axis=2) # (n_sources, n_bins, n_frames, n_channels)
        B = np.sum(Lambda[..., np.newaxis] / R[np.newaxis], axis=2)
        B[B < eps] = eps
        g = g * np.sqrt(A / B)

        self.spatial_covariance = g
    
    def update_diagonalizer(self):
        n_bins = self.n_bins
        n_channels = self.n_channels
        eps, threshold = self.eps, self.threshold

        X = self.input.transpose(1, 2, 0)
        Q = self.diagonalizer
        g = self.spatial_covariance
        XX = X[:, :, :, np.newaxis] @ X[:, :, np.newaxis, :].conj()

        if self.partitioning:
            Z = self.latent # (n_sources, n_bases)
            W, H = self.base, self.activation
            ZW = Z[:, np.newaxis, :] * W[np.newaxis, :, :] # (n_sources, n_bins, n_bases)
            Lambda = ZW @ H[np.newaxis, :, :] # (n_sources, n_bins, n_frames)
        else:
            W, H = self.base, self.activation
            Lambda = W @ H

        R = np.sum(Lambda[..., np.newaxis] * g[:, :, np.newaxis], axis=0)
        R[R < eps] = eps
        E = np.eye(n_channels)
        E = np.tile(E, reps=(n_bins, 1, 1)) # (n_bins, n_channels, n_channels)

        for channel_idx in range(n_channels):
            # Q: (n_bins, n_channels, n_channels)
            q_m_Hermite = Q[:, channel_idx, :]
            V = (XX / R[:, :, channel_idx, np.newaxis, np.newaxis]).mean(axis=1)
            QV = Q @ V
            condition = np.linalg.cond(QV) < threshold # (n_bins,)
            condition = condition[:,np.newaxis] # (n_bins, 1)
            e_m = E[:, channel_idx, :]
            q_m = np.linalg.solve(QV, e_m)
            qVq = q_m.conj()[:, np.newaxis, :] @ V @ q_m[:, :, np.newaxis]
            denominator = np.sqrt(qVq[...,0])
            denominator[denominator < eps] = eps
            # if condition number is too big, only `denominator[denominator < eps] = eps` may diverge of cost function.
            q_m_Hermite = np.where(condition, q_m.conj() / denominator, q_m_Hermite)
            Q[:, channel_idx, :] = q_m_Hermite
        
        self.diagonalizer = Q
    
    def compute_negative_loglikelihood(self):
        n_frames = self.n_frames
        eps = self.eps
        
        X = self.input.transpose(1, 2, 0)
        Q = self.diagonalizer
        g = self.spatial_covariance

        if self.partitioning:
            Z = self.latent # (n_sources, n_bases)
            W, H = self.base, self.activation
            ZW = Z[:, np.newaxis, :] * W[np.newaxis, :, :] # (n_sources, n_bins, n_bases)
            Lambda = ZW @ H[np.newaxis, :, :] # (n_sources, n_bins, n_frames)
        else:
            W, H = self.base, self.activation
            Lambda = W @ H
        
        y_tilde = np.sum(Lambda[..., np.newaxis] * g[:, :, np.newaxis, :], axis=0) # (n_bins, n_frames, n_channels)
        # TODO: shape
        QX = np.sum(Q[:, np.newaxis, :, :] * X[:, :, np.newaxis, :], axis=3) # (n_bins, n_channels, n_channels) (n_bins, n_frames, n_channels) -> (n_bins, n_frames, n_channels)
        x_tilde = np.abs(QX)**2 # (n_bins, n_frames, n_channels)
        detQQ = np.abs(np.linalg.det(Q @ Q.transpose(0, 2, 1))) # (n_bins,)

        x_tilde, y_tilde = x_tilde + eps, y_tilde + eps

        loss = np.sum(x_tilde / y_tilde + np.log(y_tilde)) - n_frames * np.sum(np.log(detQQ))

        return loss
    
    def separate(self, input):
        reference_id = self.reference_id
        eps = self.eps

        X = input.transpose(1, 2, 0)
        Q = self.diagonalizer # (n_bins, n_channels, n_channels)
        g = self.spatial_covariance

        if self.partitioning:
            Z = self.latent # (n_sources, n_bases)
            W, H = self.base, self.activation
            ZW = Z[:, np.newaxis, :] * W[np.newaxis, :, :] # (n_sources, n_bins, n_bases)
            Lambda = ZW @ H[np.newaxis, :, :] # (n_sources, n_bins, n_frames)
        else:
            W, H = self.base, self.activation
            Lambda = W @ H
        
        LambdaG = Lambda[..., np.newaxis] * g[:, :, np.newaxis, :] # (n_sources, n_bins, n_frames, n_channels)
        y_tilde = np.sum(LambdaG, axis=0) # (n_bins, n_frames, n_channels)
        Q_inverse = np.linalg.inv(Q) # (n_bins, n_channels, n_channels)
        QX = np.sum(Q[:, np.newaxis, :] * X[:, :, np.newaxis], axis=3) # (n_bins, n_frames, n_channels)
        y_tilde[y_tilde < eps] = eps
        QXLambdaGy = QX * (LambdaG / y_tilde) # (n_sources, n_bins, n_frames, n_channels)
        
        x_hat = np.sum(Q_inverse[:, np.newaxis, :, :] * QXLambdaGy[:, :, :, np.newaxis, :], axis=4) # (n_sources, n_bins, n_frames, n_channels)
        x_hat = x_hat.transpose(0, 3, 1, 2) # (n_sources, n_channels, n_bins, n_frames)
        
        return x_hat[:, reference_id, :, :]

def is_divergence(input, target, eps=EPS):
    """
    Multichannel Itakura-Saito divergence
    Args:
        input (*, n_channels, n_channels)
        target (*, n_channels, n_channels)
    """
    shape_input, shape_target = input.shape, target.shape
    assert shape_input[-2] == shape_input[-1] and shape_target[-2] == shape_target[-1], "Invalid input shape"
    n_channels = shape_input[-1]
    
    input, target = input + eps * np.eye(n_channels), target + eps * np.eye(n_channels)
    XX = target @ np.linalg.inv(input)

    loss = np.trace(XX, axis1=-2, axis2=-1).real - np.log(np.linalg.det(XX)).real - n_channels

    return loss

def _convolve_mird(titles, reverb=0.160, degrees=[0], mic_intervals=[8,8,8,8,8,8,8], mic_indices=[0], samples=None):
    intervals = '-'.join([str(interval) for interval in mic_intervals])

    T_min = None

    for title in titles:
        source, sr = read_wav("data/single-channel/{}.wav".format(title))
        T = len(source)
        if T_min is None or T < T_min:
            T_min = T

    mixed_signals = []

    for mic_idx in mic_indices:
        _mixture = 0
        for title_idx in range(len(titles)):
            degree = degrees[title_idx]
            title = titles[title_idx]
            rir_path = "data/MIRD/Reverb{:.3f}_{}/Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_{:.3f}s)_{}_1m_{:03d}.mat".format(reverb, intervals, reverb, intervals, degree)
            rir_mat = loadmat(rir_path)

            rir = rir_mat['impulse_response']

            if samples is not None:
                rir = rir[:samples]

            source, sr = read_wav("data/single-channel/{}.wav".format(title))
            _mixture = _mixture + np.convolve(source[:T_min], rir[:, mic_idx])
        
        mixed_signals.append(_mixture)
    
    mixed_signals = np.array(mixed_signals)

    return mixed_signals

def _test(method, n_bases=10, partitioning=False):
    np.random.seed(111)
    
    # Room impulse response
    sr = 16000
    reverb = 0.16
    duration = 0.5
    samples = int(duration * sr)
    mic_intervals = [8, 8, 8, 8, 8, 8, 8]
    mic_indices = [2, 5]
    degrees = [60, 300]
    titles = ['sample-song/sample3_source3', 'sample-song/sample3_source2']

    mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_intervals=mic_intervals, mic_indices=mic_indices, samples=samples)
    write_wav("data/MNMF/{}MNMF/partitioning{}/mixture.wav".format(method, int(partitioning)), signal=mixed_signal.T, sr=sr)

    n_sources, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 4096, 2048
    mixture = stft(mixed_signal, fft_size=fft_size, hop_size=hop_size)

    # MNMF
    n_channels = len(titles)
    iteration = 50

    if method == 'IS':
        mnmf = ISMultichannelNMF(n_bases=n_bases)
    elif method == 'FastIS':
        mnmf = FastISMultichannelNMF(n_bases=n_bases)
    else:
        raise ValueError("Not support {}-MNMF.".format(method))

    print(mnmf)
    estimation = mnmf(mixture, iteration=iteration)
    print(mnmf)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_channels):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/MNMF/{}MNMF/partitioning{}/mixture-{}_estimated-iter{}-{}.wav".format(method, int(partitioning), sr, iteration, idx), signal=_estimated_signal, sr=sr)
    
    plt.figure()
    plt.plot(mnmf.loss[1:], color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/MNMF/{}MNMF/partitioning{}/loss.png'.format(method, int(partitioning)), bbox_inches='tight')
    plt.close()

def _test_conv():
    sr = 16000
    reverb = 0.16
    duration = 0.5
    samples = int(duration * sr)
    mic_indices = [2, 5]
    degrees = [60, 300]
    titles = ['man-16000', 'woman-16000']

    wav_path = "data/multi-channel/mixture-{}.wav".format(sr)

    if not os.path.exists(wav_path):
        mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_indices=mic_indices, samples=samples)
        write_wav(wav_path, mixed_signal.T, sr=sr)


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.io import loadmat

    from utils.utils_audio import read_wav, write_wav
    from transform.stft import stft, istft

    plt.rcParams['figure.dpi'] = 200

    os.makedirs("data/multi-channel", exist_ok=True)
    os.makedirs("data/MNMF/ISMNMF/partitioning0", exist_ok=True)
    os.makedirs("data/MNMF/FastISMNMF/partitioning0", exist_ok=True)

    """
    Use multichannel room impulse response database.
    Download database from "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/"
    """

    _test_conv()
    _test(method='IS', n_bases=2, partitioning=False)
    _test(method='FastIS', n_bases=4, partitioning=False)