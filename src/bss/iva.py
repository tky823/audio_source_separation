import numpy as np
import scipy.sparse as sci_sparse

from bss.prox import PDSBSSbase
from utils.utils_linalg import parallel_sort
from algorithm.projection_back import projection_back

EPS = 1e-12
THRESHOLD = 1e+12

__algorithms_spatial__ = ['IP', 'IVA', 'ISS', 'IPA', 'pairwise', 'IP1', 'IP2']

"""
References for `algorithm_spatial`:
    IP: "Stable and fast update rules for independent vector analysis based on auxiliary function technique"
        same as `IP1`
    ISS: "Fast and Stable Blind Source Separation with Rank-1 Updates"
    IPA:
    IP2:
"""

class IVAbase:
    def __init__(self, callbacks=None, recordable_loss=True, eps=EPS):
        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None
        self.eps = eps

        self.input = None
        self.recordable_loss = recordable_loss
        if self.recordable_loss:
            self.loss = []
        else:
            self.loss = None
    
    def _reset(self, **kwargs):
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        X = self.input

        n_channels, n_bins, n_frames = X.shape
        n_sources = n_channels # n_channels == n_sources

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        if not hasattr(self, 'demix_filter'):
            W = np.eye(n_sources, n_channels, dtype=np.complex128)
            self.demix_filter = np.tile(W, reps=(n_bins, 1, 1))
        else:
            W = self.demix_filter.copy()
            self.demix_filter = W
        self.estimation = self.separate(X, demix_filter=W)
        
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
        
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self)

        for idx in range(iteration):
            self.update_once()

            if self.recordable_loss:
                loss = self.compute_negative_loglikelihood()
                self.loss.append(loss)
            
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)
        
        X, W = input, self.demix_filter
        output = self.separate(X, demix_filter=W)

        return output
    
    def __repr__(self):
        s = "IVA("
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        raise NotImplementedError("Implement 'update_once' function")
    
    def separate(self, input, demix_filter):
        """
        Args:
            input (n_channels, n_bins, n_frames): 
            demix_filter (n_bins, n_sources, n_channels): 
        Returns:
            output (n_channels, n_bins, n_frames): 
        """
        input = input.transpose(1,0,2)
        estimation = demix_filter @ input
        output = estimation.transpose(1,0,2)

        return output
    
    def compute_demix_filter(self, estimation, input):
        X, Y = input, estimation
        X_Hermite = X.transpose(1, 2, 0).conj()
        XX_Hermite = X.transpose(1, 0, 2) @ X_Hermite
        demix_filter = Y.transpose(1, 0, 2) @ X_Hermite @ np.linalg.inv(XX_Hermite)

        return demix_filter
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' function.")

class GradIVAbase(IVAbase):
    """
    Reference: "Independent Vector Analysis: Definition and Algorithms"
    See https://ieeexplore.ieee.org/document/4176796
    """
    def __init__(self, lr=1e-1, reference_id=0, callbacks=None, apply_projection_back=True, recordable_loss=True, eps=EPS):
        super().__init__(callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.lr = lr
        self.reference_id = reference_id
        self.apply_projection_back = apply_projection_back
    
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
        
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self)

        for idx in range(iteration):
            self.update_once()

            if self.recordable_loss:
                loss = self.compute_negative_loglikelihood()
                self.loss.append(loss)
            
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)

        reference_id = self.reference_id
        X, W = input, self.demix_filter
        output = self.separate(X, demix_filter=W)

        if self.apply_projection_back:
            scale = projection_back(output, reference=X[reference_id])
            output = output * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def __repr__(self):
        s = "GradIVA("
        s += "lr={lr}"
        s += ")"

        return s.format(**self.__dict__)
    
    def update_once(self):
        raise NotImplementedError("Implement 'update_once' function")

    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' function.")

class GradLaplaceIVA(GradIVAbase):
    def __init__(self, lr=1e-1, reference_id=0, callbacks=None, apply_projection_back=True, recordable_loss=True, eps=EPS):
        super().__init__(lr=lr, reference_id=reference_id, callbacks=callbacks, apply_projection_back=apply_projection_back, recordable_loss=recordable_loss, eps=eps)
    
    def __repr__(self):
        s = "NaturalGradIVA("
        s += "lr={lr}"
        s += ")"

        return s.format(**self.__dict__)
    
    def update_once(self):
        n_frames = self.n_frames
        lr = self.lr
        eps = self.eps

        X = self.input
        W = self.demix_filter
        Y = self.separate(X, demix_filter=W)

        X_Hermite = X.transpose(1,2,0).conj() # (n_bins, n_frames, n_sources)
        W_inverse = np.linalg.inv(W)
        W_inverseHermite = W_inverse.transpose(0,2,1).conj() # (n_bins, n_channels, n_sources)

        Y = Y.transpose(1,0,2) # (n_bins, n_sources, n_frames)
        P = np.abs(Y)**2
        denominator = np.sqrt(P.sum(axis=0))
        denominator[denominator < eps] = eps
        Phi = Y / denominator # (n_bins, n_sources, n_frames)

        delta = (Phi @ X_Hermite) / n_frames - W_inverseHermite
        W = W - lr * delta # (n_bins, n_sources, n_channels)
        
        X = self.input
        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.estimation = Y
    
    def compute_negative_loglikelihood(self):
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        P = np.sum(np.abs(Y)**2, axis=1)
        loss = 2 * np.sum(np.sqrt(P), axis=0).mean() - 2 * np.log(np.abs(np.linalg.det(W))).sum()

        return loss

class NaturalGradLaplaceIVA(GradIVAbase):
    def __init__(self, lr=1e-1, reference_id=0, callbacks=None, apply_projection_back=True, recordable_loss=True, eps=EPS):
        super().__init__(lr=lr, reference_id=reference_id, callbacks=callbacks, apply_projection_back=apply_projection_back, recordable_loss=recordable_loss, eps=eps)
    
    def __repr__(self):
        s = "NaturalGradLaplaceIVA("
        s += "lr={lr}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        n_frames = self.n_frames
        lr = self.lr
        eps = self.eps

        X = self.input
        W = self.demix_filter
        Y = self.separate(X, demix_filter=W)
        eye = np.eye(n_sources, n_channels, dtype=np.complex128)

        Y = Y.transpose(1,0,2) # (n_bins, n_sources, n_frames)
        Y_Hermite = Y.transpose(0,2,1).conj() # (n_bins, n_frames, n_sources)
        P = np.abs(Y)**2
        denominator = np.sqrt(P.sum(axis=0))
        denominator[denominator < eps] = eps
        Phi = Y / denominator # (n_bins, n_sources, n_frames)

        delta = ((Phi @ Y_Hermite) / n_frames - eye) @ W
        W = W - lr * delta # (n_bins, n_sources, n_channels)
        
        X = self.input
        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.estimation = Y
    
    def compute_negative_loglikelihood(self):
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        P = np.sum(np.abs(Y)**2, axis=1)
        loss = 2 * np.sum(np.sqrt(P), axis=0).mean() - 2 * np.log(np.abs(np.linalg.det(W))).sum()

        return loss

class AuxIVAbase(IVAbase):
    """
    References:
        "Stable and Fast Update Rules for Independent Vector Analysis Based on Auxiliary Function Technique"
        "Auxiliary-function-based Independent Vector Analysis with Power of Vector-norm Type Weighting Functions"
        "Fast and Stable Blind Source Separation with Rank-1 Updates"
        "Independent Vector Analysis via Log-Quadratically Penalized Quadratic Minimization"
        "Independent Vector Analysis with more Microphones than Sources"
    """
    def __init__(self, algorithm_spatial='IP', reference_id=0, callbacks=None, apply_projection_back=True, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        super().__init__(callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.algorithm_spatial = algorithm_spatial
        self.reference_id = reference_id
        self.apply_projection_back = apply_projection_back
        self.threshold = threshold

        if not self.algorithm_spatial in __algorithms_spatial__:
            raise ValueError("Not support {} based spatial updates.".format(self.algorithm_spatial))
        
        if self.algorithm_spatial in ['pairwise', 'IP2']:
            self.update_pair = None
    
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
        
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self)

        for idx in range(iteration):
            if self.algorithm_spatial in ['pairwise', 'IP2']:
                self._select_update_pair()
            
            self.update_once()
            if self.recordable_loss:
                loss = self.compute_negative_loglikelihood()
                self.loss.append(loss)

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)

        reference_id = self.reference_id
        X, W = input, self.demix_filter
        output = self.separate(X, demix_filter=W)

        if self.apply_projection_back:
            scale = projection_back(output, reference=X[reference_id])
            output = output * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        
        self.estimation = output

        return output
    
    def _reset(self, **kwargs):
        super()._reset(**kwargs)

        if self.algorithm_spatial == 'ISS':
            self.demix_filter = None
    
    def __repr__(self):
        s = "AuxIVA("
        s += "algorithm_spatial={algorithm_spatial}"
        s += ")"

        return s.format(**self.__dict__)
    
    def update_once(self):
        raise NotImplementedError("Implement 'update_once' function.")
    
    def _select_update_pair(self):
        # For pairwise update
        n_sources = self.n_sources

        if self.update_pair is None:
            m, n = 0, 1
        else:
            m, n = self.update_pair
            m, n = m + 1, n + 1
            m, n = m % n_sources, n % n_sources

        self.update_pair = m, n

    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' function.")

class AuxLaplaceIVA(AuxIVAbase):
    def __init__(self, algorithm_spatial='IP', reference_id=0, callbacks=None, apply_projection_back=True, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        super().__init__(algorithm_spatial=algorithm_spatial, reference_id=reference_id, callbacks=callbacks, apply_projection_back=apply_projection_back, recordable_loss=recordable_loss, eps=eps, threshold=threshold)

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
        
        if self.callbacks is not None:
            if self.algorithm_spatial == 'ISS':
                # In `update_once()`, demix_filter isn't updated
                # because we don't have to compute demixing filter explicitly by AuxIVA-ISS.
                X, Y = self.input, self.estimation
                self.demix_filter = self.compute_demix_filter(Y, X)
            
            for callback in self.callbacks:
                callback(self)
            
            if self.algorithm_spatial == 'ISS':
                self.demix_filter = None

        for idx in range(iteration):
            if self.algorithm_spatial in ['pairwise', 'IP2']:
                self._select_update_pair()
            
            self.update_once()

            if self.recordable_loss:
                loss = self.compute_negative_loglikelihood()
                self.loss.append(loss)

            if self.callbacks is not None:
                if self.algorithm_spatial == 'ISS':
                    # In `update_once()`, demix_filter isn't updated
                    # because we don't have to compute demixing filter explicitly by AuxIVA-ISS.
                    X, Y = self.input, self.estimation
                    self.demix_filter = self.compute_demix_filter(Y, X)
                
                for callback in self.callbacks:
                    callback(self)
                
                if self.algorithm_spatial == 'ISS':
                    self.demix_filter = None

        reference_id = self.reference_id

        if self.algorithm_spatial == 'ISS':
            X, Y = self.input, self.estimation
            self.demix_filter = self.compute_demix_filter(Y, X)
        else:
            X, W = input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
        
        output = Y

        if self.apply_projection_back:
            scale = projection_back(output, reference=X[reference_id])
            output = output * scale[..., np.newaxis] # (n_sources, n_bins, n_frames)
        
        self.estimation = output

        return output
    
    def __repr__(self):
        s = "AuxLaplaceIVA("
        s += "algorithm_spatial={algorithm_spatial}"
        s += ")"

        return s.format(**self.__dict__)
    
    def update_once(self):
        if self.algorithm_spatial in ['IP', 'IP1']:
            self.update_once_ip()
        elif self.algorithm_spatial == 'ISS':
            self.update_once_iss()
        elif self.algorithm_spatial in ['pairwise', 'IP2']:
            self.update_once_pairwise()
        elif self.algorithm_spatial == 'IPA':
            self.update_once_ipa()
        else:
            raise ValueError("Not support {} based spatial updates.".format(self.algorithm_spatial))
    
    def update_once_ip(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins
        eps, threshold = self.eps, self.threshold

        X, Y = self.input, self.estimation
        W = self.demix_filter

        P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)
        R = np.sqrt(P.sum(axis=1)) # (n_sources, n_frames)
        R = R[:, np.newaxis, :, np.newaxis, np.newaxis] # (n_sources, 1, n_frames, 1, 1)

        X = X.transpose(1, 2, 0) # (n_bins, n_frames, n_channels)
        X = X[..., np.newaxis]
        X_Hermite = X.transpose(0, 1, 3, 2).conj()
        XX = X @ X_Hermite # (n_bins, n_frames, n_channels, n_channels)
        R[R < eps] = eps
        U = XX / R # (n_sources, n_bins, n_frames, n_channels, n_channels)
        U = U.mean(axis=2) # (n_sources, n_bins, n_channels, n_channels)
        E = np.eye(n_sources, n_channels)
        E = np.tile(E, reps=(n_bins, 1, 1)) # (n_bins, n_sources, n_channels)

        for n in range(n_sources):
            # W: (n_bins, n_sources, n_channels), U: (n_sources, n_bins, n_channels, n_channels)
            w_n_Hermite = W[:, n, :] # (n_bins, n_channels)
            U_n = U[n] # (n_bins, n_channels, n_channels)
            WU = W @ U_n # (n_bins, n_sources, n_channels)
            condition = np.linalg.cond(WU) < threshold # (n_bins,)
            condition = condition[:, np.newaxis] # (n_bins, 1)
            e_n = E[:, n, :]
            w_n = np.linalg.solve(WU, e_n)
            wUw = w_n[:, np.newaxis, :].conj() @ U_n @ w_n[:, :, np.newaxis]
            denominator = np.sqrt(wUw[..., 0])
            w_n_Hermite = np.where(condition, w_n.conj() / denominator, w_n_Hermite)
            # if condition number is too big, `denominator[denominator < eps] = eps` may occur divergence of cost function.
            W[:, n, :] = w_n_Hermite
            
        self.demix_filter = W

        X = self.input
        Y = self.separate(X, demix_filter=W)
        
        self.estimation = Y
    
    def update_once_iss(self):
        n_sources = self.n_sources
        eps = self.eps

        Y = self.estimation

        P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)
        R = np.sqrt(P.sum(axis=1)) # (n_sources, n_frames)
        R[R < eps] = eps

        for n in range(n_sources):
            U_n = np.sum(Y * Y[n].conj() / R[:, np.newaxis, :], axis=2) # (n_sources, n_bins)
            D_n = np.sum(np.abs(Y[n])**2 / R[:, np.newaxis, :], axis=2) # (n_sources, n_bins)
            V_n = U_n / D_n # (n_sources, n_bins)
            V_n[n] = 1 - 1 / np.sqrt(D_n[n])
            Y = Y - V_n[:, :, np.newaxis] * Y[n]
        
        self.estimation = Y

    def update_once_pairwise(self):
        n_channels = self.n_channels
        n_bins = self.n_bins
        eps, threshold = self.eps, self.threshold

        X, Y = self.input, self.estimation
        W = self.demix_filter
        m, n = self.update_pair

        Y_m, Y_n = Y[m], Y[n]
        P_m, P_n = np.abs(Y_m)**2, np.abs(Y_n)**2 # (n_bins, n_frames)
        R_m, R_n = np.sqrt(P_m.sum(axis=0)), np.sqrt(P_n.sum(axis=0)) # (n_frames,)
        R_m, R_n = R_m[np.newaxis, ..., np.newaxis, np.newaxis], R_n[np.newaxis, ..., np.newaxis, np.newaxis] # (1, n_frames, 1, 1)
            
        X = X.transpose(1, 2, 0) # (n_bins, n_frames, n_channels)
        X = X[..., np.newaxis]
        X_Hermite = X.transpose(0, 1, 3, 2).conj()
        XX = X @ X_Hermite # (n_bins, n_frames, n_channels, n_channels)
        R_m[R_m < eps] = eps
        R_n[R_n < eps] = eps
        U_m, U_n = XX / R_m, XX / R_n
        U_m, U_n = U_m.mean(axis=1), U_n.mean(axis=1) # (n_bins, n_channels, n_channels)
        e_m, e_n = np.zeros((n_bins, n_channels, 1)), np.zeros((n_bins, n_channels, 1))
        e_m[:, m, :], e_n[:, n, :] = 1, 1
        E_mn = np.concatenate([e_m, e_n], axis=2) # (n_bins, n_channels, 2)
            
        WU_m, WU_n = W @ U_m, W @ U_n # (n_bins, n_channels, n_channels)
        condition_m, condition_n = np.linalg.cond(WU_m) < threshold, np.linalg.cond(WU_n) < threshold  # (n_bins,)
        condition_m, condition_n = condition_m[:, np.newaxis], condition_n[:, np.newaxis]

        WU_m_inv, WU_n_inv = np.linalg.inv(WU_m), np.linalg.inv(WU_n)
        P_m, P_n = WU_m_inv @ E_mn, WU_n_inv @ E_mn # (n_bins, n_channels, n_channels), (n_bins, n_channels, 2) -> (n_bins, n_channels, 2)
        P_m_Hermite, P_n_Hermite = P_m.transpose(0, 2, 1).conj(), P_n.transpose(0, 2, 1).conj()
        V_m, V_n = P_m_Hermite @ U_m @ P_m, P_n_Hermite @ U_n @ P_n
        VV = np.linalg.inv(V_n) @ V_m # (n_bins, 2, 2)
        eig_values, v = np.linalg.eig(VV) # (n_bins, 2), # (n_bins, 2, 2)
        order = np.argsort(eig_values, axis=-1)[:, ::-1] # (n_bins, 2)
        v_transpose = v.swapaxes(-2, -1)
        v_mn = parallel_sort(v_transpose, order=order, axis=-2)
        v_m, v_n = np.split(v_mn, 2, axis=1) # (n_bins, 1, 2), (n_bins, 1, 2)
        v_m, v_n = v_m.squeeze(axis=1), v_n.squeeze(axis=1)
        vUv_m, vUv_n = v_m[:, np.newaxis, :].conj() @ V_m @ v_m[:, :, np.newaxis], v_n[:, np.newaxis, :].conj() @ V_n @ v_n[:, :, np.newaxis]
        denominator_m, denominator_n = np.sqrt(vUv_m.squeeze(axis=-1)), np.sqrt(vUv_n.squeeze(axis=-1))
        v_m, v_n = v_m / denominator_m, v_n / denominator_n
        w_m, w_n = P_m @ v_m[..., np.newaxis], P_n @ v_n[..., np.newaxis]
        w_m, w_n = w_m.squeeze(axis=-1).conj(), w_n.squeeze(axis=-1).conj()

        W[:, m, :] = np.where(condition_m, w_m, W[:, m, :])
        W[:, n, :] = np.where(condition_n, w_n, W[:, n, :])

        self.demix_filter = W

        X = self.input
        Y = self.separate(X, demix_filter=W)
        
        self.estimation = Y
    
    def update_once_ipa(self):
        raise ValueError("in progress...")

    def compute_negative_loglikelihood(self):
        n_frames = self.n_frames

        X = self.input
        if self.demix_filter is None:
            Y = self.estimation
            W = self.compute_demix_filter(Y, X)
        else:
            W = self.demix_filter
            Y = self.separate(X, demix_filter=W)
        
        P = np.sum(np.abs(Y)**2, axis=1)
        R = 2 * np.sqrt(P)
        loss = R.sum() - 2 * n_frames * np.log(np.abs(np.linalg.det(W))).sum()

        return loss

class AuxGaussIVA(AuxIVAbase):
    def __init__(self, algorithm_spatial='IP', reference_id=0, callbacks=None, apply_projection_back=True, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        super().__init__(algorithm_spatial=algorithm_spatial, reference_id=reference_id, callbacks=callbacks, apply_projection_back=apply_projection_back, recordable_loss=recordable_loss, eps=eps, threshold=threshold)
    
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
        
        if self.callbacks is not None:
            if self.algorithm_spatial == 'ISS':
                # In `update_once()`, demix_filter isn't updated
                # because we don't have to compute demixing filter explicitly by AuxIVA-ISS.
                X, Y = self.input, self.estimation
                self.demix_filter = self.compute_demix_filter(Y, X)
            
            for callback in self.callbacks:
                callback(self)
            
            if self.algorithm_spatial == 'ISS':
                self.demix_filter = None

        for idx in range(iteration):
            if self.algorithm_spatial in ['pairwise', 'IP2']:
                self._select_update_pair()
            
            self.update_once()

            if self.recordable_loss:
                loss = self.compute_negative_loglikelihood()
                self.loss.append(loss)

            if self.callbacks is not None:
                if self.algorithm_spatial == 'ISS':
                    # In `update_once()`, demix_filter isn't updated
                    # because we don't have to compute demixing filter explicitly by AuxIVA-ISS.
                    X, Y = self.input, self.estimation
                    self.demix_filter = self.compute_demix_filter(Y, X)
                
                for callback in self.callbacks:
                    callback(self)
                
                if self.algorithm_spatial == 'ISS':
                    self.demix_filter = None

        reference_id = self.reference_id

        if self.algorithm_spatial == 'ISS':
            X, Y = self.input, self.estimation
            self.demix_filter = self.compute_demix_filter(Y, X)
        else:
            X, W = input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
        
        output = Y

        if self.apply_projection_back:
            scale = projection_back(output, reference=X[reference_id])
            output = output * scale[..., np.newaxis] # (n_sources, n_bins, n_frames)
        
        self.estimation = output

        return output

    def __repr__(self):
        s = "AuxGaussIVA("
        s += "algorithm_spatial={algorithm_spatial}"
        s += ")"

        return s.format(**self.__dict__)
    
    def update_once(self):
        if self.algorithm_spatial in ['IP', 'IP1']:
            self.update_once_ip()
        elif self.algorithm_spatial == 'ISS':
            self.update_once_iss()
        elif self.algorithm_spatial in ['pairwise', 'IP2']:
            self.update_once_pairwise()
        elif self.algorithm_spatial == 'IPA':
            self.update_once_ipa()
        else:
            raise ValueError("Not support {} based spatial updates.".format(self.algorithm_spatial))
    
    def update_once_ip(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins
        eps, threshold = self.eps, self.threshold

        X, Y = self.input, self.estimation
        W = self.demix_filter

        P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)
        R = P.mean(axis=1) # (n_sources, n_frames)
        R = R[:, np.newaxis, :, np.newaxis, np.newaxis] # (n_sources, 1, n_frames, 1, 1)

        X = X.transpose(1, 2, 0) # (n_bins, n_frames, n_channels)
        X = X[..., np.newaxis]
        X_Hermite = X.transpose(0, 1, 3, 2).conj()
        XX = X @ X_Hermite # (n_bins, n_frames, n_channels, n_channels)
        R[R < eps] = eps
        U = XX / R # (n_sources, n_bins, n_frames, n_channels, n_channels)
        U = U.mean(axis=2) # (n_sources, n_bins, n_channels, n_channels)
        E = np.eye(n_sources, n_channels)
        E = np.tile(E, reps=(n_bins, 1, 1)) # (n_bins, n_sources, n_channels)

        for n in range(n_sources):
            # W: (n_bins, n_sources, n_channels), U: (n_sources, n_bins, n_channels, n_channels)
            w_n_Hermite = W[:, n, :] # (n_bins, n_channels)
            U_n = U[n] # (n_bins, n_channels, n_channels)
            WU = W @ U_n # (n_bins, n_sources, n_channels)
            condition = np.linalg.cond(WU) < threshold # (n_bins,)
            condition = condition[:, np.newaxis] # (n_bins, 1)
            e_n = E[:, n, :]
            w_n = np.linalg.solve(WU, e_n)
            wUw = w_n[:, np.newaxis, :].conj() @ U_n @ w_n[:, :, np.newaxis]
            denominator = np.sqrt(wUw[..., 0])
            w_n_Hermite = np.where(condition, w_n.conj() / denominator, w_n_Hermite)
            # if condition number is too big, `denominator[denominator < eps] = eps` may occur divergence of cost function.
            W[:, n, :] = w_n_Hermite
        
        self.demix_filter = W

        X = self.input
        Y = self.separate(X, demix_filter=W)
        
        self.estimation = Y
    
    def update_once_iss(self):
        n_sources = self.n_sources
        eps = self.eps

        Y = self.estimation

        P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)
        R = P.mean(axis=1) # (n_sources, n_frames)
        R[R < eps] = eps

        for n in range(n_sources):
            U_n = np.sum(Y * Y[n].conj() / R[:, np.newaxis, :], axis=2) # (n_sources, n_bins)
            D_n = np.sum(np.abs(Y[n])**2 / R[:, np.newaxis, :], axis=2) # (n_sources, n_bins)
            V_n = U_n / D_n # (n_sources, n_bins)
            V_n[n] = 1 - 1 / np.sqrt(D_n[n])
            Y = Y - V_n[:, :, np.newaxis] * Y[n]
        
        self.estimation = Y
    
    def update_once_pairwise(self):
        raise NotImplementedError("In progress...")
    
    def update_once_ipa(self):
        raise NotImplementedError("In progress...")

    def compute_negative_loglikelihood(self):
        X = self.input

        if self.demix_filter is None:
            Y = self.estimation
            W = self.compute_demix_filter(Y, X)
        else:
            W = self.demix_filter
            Y = self.separate(X, demix_filter=W)
        
        n_bins, n_frames = self.n_bins, self.n_frames
        eps = self.eps

        Y = self.separate(X, demix_filter=W)
        P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)
        R = P.mean(axis=1) # (n_sources, n_frames)
        R[R < eps] = eps
        loss = n_bins * np.sum(np.log(R)) - 2 * n_frames * np.log(np.abs(np.linalg.det(W))).sum()

        return loss

class SparseAuxIVA(AuxIVAbase):
    """
    Reference: "A computationally cheaper method for blind speech separation based on AuxIVA and incomplete demixing transform"
    See https://ieeexplore.ieee.org/document/7602921
    """
    def __init__(self, algorithm_spatial='IP', reference_id=0, callbacks=None, apply_projection_back=True, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        super().__init__(algorithm_spatial=algorithm_spatial, reference_id=reference_id, callbacks=callbacks, apply_projection_back=apply_projection_back, recordable_loss=recordable_loss, eps=eps, threshold=threshold)

        raise NotImplementedError("in progress")
    
    def update_once(self):
        raise NotImplementedError("in progress...")

class OverAuxIVAbase(AuxIVAbase):
    def __init__(self, algorithm_spatial, n_sources=None, reference_id=0, callbacks=None, apply_projection_back=True, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        super().__init__(algorithm_spatial=algorithm_spatial, reference_id=reference_id, callbacks=callbacks, apply_projection_back=apply_projection_back, recordable_loss=recordable_loss, eps=eps, threshold=threshold)

        self.n_sources = n_sources

class OverAuxLaplaceIVA(OverAuxIVAbase):
    def __init__(self, algorithm_spatial, n_sources=None, reference_id=0, callbacks=None, apply_projection_back=True, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        super().__init__(algorithm_spatial=algorithm_spatial, n_sources=n_sources, reference_id=reference_id, callbacks=callbacks, apply_projection_back=apply_projection_back, recordable_loss=recordable_loss, eps=eps, threshold=threshold)

    def __call__(self, input, iteration=100, **kwargs):

        return super().__call__(input, iteration=iteration, **kwargs)

class ProxLaplaceIVA(PDSBSSbase):
    def __init__(self, regularizer=1, step_prox_logdet=1e+0, step_prox_penalty=1e+0, step=1e+0, reference_id=0, callbacks=None, apply_projection_back=True, recordable_loss=True, eps=EPS):
        super().__init__(regularizer=regularizer, step_prox_logdet=step_prox_logdet, step_prox_penalty=step_prox_penalty, step=step, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.reference_id = reference_id
        self.apply_projection_back = apply_projection_back
    
    def __call__(self, input, iteration, **kwargs):
        """
        Args:
            input (n_channels, n_bins, n_frames)
            iteration <int>
        Returns:
            output (n_channels, n_bins, n_frames)
        """
        Y = super().__call__(input, iteration=iteration, **kwargs)

        reference_id = self.reference_id
        X, W = input, self.demix_filter
        output = self.separate(X, demix_filter=W)

        if self.apply_projection_back:
            scale = projection_back(output, reference=X[reference_id])
            output = output * scale[..., np.newaxis] # (n_sources, n_bins, n_frames)
        
        self.estimation = output

        return output
    
    def __repr__(self):
        s = "ProxLaplaceIVA("
        s += "algorithm_spatial={algorithm_spatial}"
        s += ")"

        return s.format(**self.__dict__)
    
    def prox_penalty(self, z, mu=1, is_sparse=True):
        """
        Args:
            z (n_bins * n_sources * n_frames, 1) <scipy.sparse.lil_matrix>
            mu <float>: 
        Returns:
            z_tilde (n_bins * n_sources * n_frames, 1) <scipy.sparse.lil_matrix>
        """
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins, n_frames = self.n_bins, self.n_frames
        C = self.regularizer

        assert is_sparse, "`is_sparse` is expected True."

        z = z.toarray().reshape(n_bins, n_sources, n_frames) # (n_bins, n_sources, n_frames)
        zsum = np.sum(np.abs(z)**2, axis=0)
        denominator = np.sqrt(zsum) # (n_sources, n_frames)
        denominator = np.where(denominator <= 0, mu, denominator)
        z_tilde = C * np.maximum(0, 1 - mu / denominator) * z # TODO: correct?
        z_tilde = z_tilde.reshape(n_bins * n_sources * n_frames, 1)
        z_tilde = sci_sparse.lil_matrix(z_tilde)

        return z_tilde
    
    def compute_penalty(self):
        """
        Returns:
            loss <float>
        """
        C = self.regularizer
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        
        loss = np.sum(np.abs(Y)**2, axis=1) # (n_sources, n_frames)
        loss = np.sqrt(loss) # (n_sources, n_frames)
        loss = C * loss.sum(axis=(0, 1))
        
        return loss

class SparseProxIVA(PDSBSSbase):
    """
    Reference: "Time-frequency-masking-based Determined BSS with Application to Sparse IVA"
    See https://ieeexplore.ieee.org/document/8682217
    """
    def __init__(self, regularizer=1, step_prox_logdet=1e+0, step_prox_penalty=1e+0, step=1e+0, reference_id=0, callbacks=None, apply_projection_back=True, recordable_loss=True, eps=EPS):
        super().__init__(regularizer=regularizer, step_prox_logdet=step_prox_logdet, step_prox_penalty=step_prox_penalty, step=step, callbacks=callbacks, apply_projection_back=apply_projection_back, recordable_loss=recordable_loss, eps=eps)

        self.reference_id = reference_id

        raise NotImplementedError("coming soon")

def _convolve_mird(titles, reverb=0.160, degrees=[0], mic_intervals=[8,8,8,8,8,8,8], mic_indices=[0], samples=None):
    intervals = '-'.join([str(interval) for interval in mic_intervals])

    T_min = None

    for title in titles:
        source, _ = read_wav("data/single-channel/{}.wav".format(title))
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

def _test_aux_iva(method='AuxLaplaceIVA'):
    np.random.seed(111)
    
    # Room impulse response
    sr = 16000
    reverb = 0.16
    duration = 0.5
    samples = int(duration * sr)
    mic_intervals = [8, 8, 8, 8, 8, 8, 8]
    mic_indices = [2, 5]
    degrees = [60, 300]
    titles = ['man-16000', 'woman-16000']
    # titles = ['sample-song/sample2_source1', 'sample-song/sample2_source2']

    mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_intervals=mic_intervals, mic_indices=mic_indices, samples=samples)
    
    n_channels, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 2048, 1024
    mixture = stft(mixed_signal, fft_size=fft_size, hop_size=hop_size)

    # IVA
    n_sources = len(titles)
    iteration = 200

    if method == 'AuxLaplaceIVA-IP':
        iva = AuxLaplaceIVA(algorithm_spatial='IP')
        iteration = 50
    elif method == 'AuxGaussIVA-IP':
        iva = AuxGaussIVA(algorithm_spatial='IP')
        iteration = 50
    elif method == 'AuxLaplaceIVA-ISS':
        iva = AuxLaplaceIVA(algorithm_spatial='ISS')
        iteration = 100
    elif method == 'AuxGaussIVA-ISS':
        iva = AuxGaussIVA(algorithm_spatial='ISS')
        iteration = 100
    else:
        raise ValueError("Not support method {}".format(method))

    estimation = iva(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_sources):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/IVA/{}/mixture-{}_estimated-iter{}-{}.wav".format(method, sr, iteration, idx), signal=_estimated_signal, sr=sr)

    plt.figure()
    plt.plot(iva.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/IVA/{}/loss.png'.format(method), bbox_inches='tight')
    plt.close()

def _test_grad_iva(method='GradLaplaceIVA'):
    np.random.seed(111)
    
    # Room impulse response
    sr = 16000
    reverb = 0.16
    duration = 0.5
    samples = int(duration * sr)
    mic_intervals = [8, 8, 8, 8, 8, 8, 8]
    mic_indices = [2, 5]
    degrees = [60, 300]
    titles = ['man-16000', 'woman-16000']

    mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_intervals=mic_intervals, mic_indices=mic_indices, samples=samples)
    
    n_channels, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 2048, 1024
    mixture = stft(mixed_signal, fft_size=fft_size, hop_size=hop_size)

    # IVA
    n_sources = len(titles)
    iteration = 200

    if method == 'GradLaplaceIVA':
        lr = 0.1
        iva = GradLaplaceIVA(lr=lr)
        iteration = 5000
    elif method == 'NaturalGradLaplaceIVA':
        lr = 0.1
        iva = NaturalGradLaplaceIVA(lr=lr)
        iteration = 200
    else:
        raise ValueError("Not support method {}".format(method))

    estimation = iva(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_sources):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/IVA/{}/mixture-{}_estimated-iter{}-{}.wav".format(method, sr, iteration, idx), signal=_estimated_signal, sr=sr)

    plt.figure()
    plt.plot(iva.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/IVA/{}/loss.png'.format(method), bbox_inches='tight')
    plt.close()

def _test_over_iva(method='OverAuxLaplaceIVA'):
    from transform.pca import pca
    from algorithm.projection_back import projection_back

    np.random.seed(111)
    
    # Room impulse response
    sr = 16000
    reverb = 0.16
    duration = 0.5
    samples = int(duration * sr)
    mic_intervals = [8, 8, 8, 8, 8, 8, 8]
    mic_indices = [2, 4, 5]
    degrees = [60, 300]
    titles = ['sample-song/sample3_source2', 'sample-song/sample3_source3']

    mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_intervals=mic_intervals, mic_indices=mic_indices, samples=samples)
    
    n_channels, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 2048, 1024
    mixture = stft(mixed_signal, fft_size=fft_size, hop_size=hop_size)

    # IVA
    n_sources = 2 # Overdetermined
    iteration = 200

    if method == 'PCA+IVA':
        reference_id = 0
        mixture_pca = pca(mixture)[:n_sources]

        iva = AuxLaplaceIVA(algorithm_spatial='IP', apply_projection_back=False)
        iteration = 50

        estimation = iva(mixture_pca, iteration=iteration)

        scale = projection_back(estimation, mixture[reference_id])
        estimation = estimation * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
    elif method == 'OverAuxLaplaceIVA':
        raise ValueError("Not support method {}".format(method))
    else:
        raise ValueError("Not support method {}".format(method))

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_sources):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/IVA/{}/mixture-{}_estimated-iter{}-{}.wav".format(method, sr, iteration, idx), signal=_estimated_signal, sr=sr)

    plt.figure()
    plt.plot(iva.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/IVA/{}/loss.png'.format(method), bbox_inches='tight')
    plt.close()

def _test_prox_iva(method='ProxLaplaceIVA'):
    np.random.seed(111)
    
    # Room impulse response
    sr = 16000
    reverb = 0.16
    duration = 0.5
    samples = int(duration * sr)
    mic_intervals = [8, 8, 8, 8, 8, 8, 8]
    mic_indices = [2, 5]
    degrees = [60, 300]
    titles = ['man-16000', 'woman-16000']

    mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_intervals=mic_intervals, mic_indices=mic_indices, samples=samples)
    
    n_channels, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 2048, 1024
    mixture = stft(mixed_signal, fft_size=fft_size, hop_size=hop_size)

    # IVA
    n_sources = len(titles)
    iteration = 200

    if method == 'ProxLaplaceIVA':
        step = 1.75
        iva = ProxLaplaceIVA(step=step)
        iteration = 100
    else:
        raise ValueError("Not support method {}".format(method))

    estimation = iva(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_sources):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/IVA/{}/mixture-{}_estimated-iter{}-{}.wav".format(method, sr, iteration, idx), signal=_estimated_signal, sr=sr)

    plt.figure()
    plt.plot(iva.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/IVA/{}/loss.png'.format(method), bbox_inches='tight')
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
    from scipy.io import loadmat

    from utils.utils_audio import read_wav, write_wav
    from transform.stft import stft, istft
    from transform.whitening import whitening

    plt.rcParams['figure.dpi'] = 200

    os.makedirs("data/multi-channel", exist_ok=True)
    os.makedirs("data/IVA/GradLaplaceIVA", exist_ok=True)
    os.makedirs("data/IVA/NaturalGradLaplaceIVA", exist_ok=True)
    os.makedirs("data/IVA/AuxLaplaceIVA-IP", exist_ok=True)
    os.makedirs("data/IVA/AuxGaussIVA-IP", exist_ok=True)
    os.makedirs("data/IVA/AuxLaplaceIVA-ISS", exist_ok=True)
    os.makedirs("data/IVA/AuxGaussIVA-ISS", exist_ok=True)
    os.makedirs("data/IVA/PCA+IVA", exist_ok=True)
    os.makedirs("data/IVA/OverIVA", exist_ok=True)
    os.makedirs("data/IVA/ProxLaplaceIVA", exist_ok=True)

    """
    Use multichannel room impulse response database.
    Download database from "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/"
    """

    _test_conv()

    print("="*10, "GradLaplaceIVA", "="*10)
    _test_grad_iva(method='GradLaplaceIVA')
    print()

    print("="*10, "NaturalGradLaplaceIVA", "="*10)
    _test_grad_iva(method='NaturalGradLaplaceIVA')
    print()
    
    print("="*10, "AuxIVA-IP", "="*10)
    print("-"*10, "AuxLaplaceIVA-IP", "-"*10)
    _test_aux_iva(method='AuxLaplaceIVA-IP')
    print()
    print("-"*10, "AuxGaussIVA-IP", "-"*10)
    _test_aux_iva(method='AuxGaussIVA-IP')
    print()

    print("="*10, "AuxIVA-ISS", "="*10)
    print("-"*10, "AuxLaplaceIVA-ISS", "-"*10)
    _test_aux_iva(method='AuxLaplaceIVA-ISS')
    print()
    print("-"*10, "AuxGaussIVA-ISS", "-"*10)
    _test_aux_iva(method='AuxGaussIVA-ISS')
    print()

    """
    print("="*10, "AuxIVA-IPA", "="*10)
    print("-"*10, "AuxLaplaceIVA-IPA", "-"*10)
    _test_aux_iva(method='AuxLaplaceIVA-IPA')
    print()
    print("-"*10, "AuxGaussIVA-IPA", "-"*10)
    _test_aux_iva(method='AuxGaussIVA-IPA')
    print()
    """
    
    print("-"*10, "PCA+IVA", "-"*10)
    _test_over_iva(method='PCA+IVA')
    print()

    print("-"*10, "OverIVA", "-"*10)
    _test_over_iva(method='OverIVA')
    print()
    
    print("="*10, "ProxIVA", "="*10)
    print("-"*10, "ProxLaplaceIVA", "-"*10)
    _test_prox_iva(method='ProxLaplaceIVA')
