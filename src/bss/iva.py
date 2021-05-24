import numpy as np
import scipy.sparse as sci_sparse

from bss.prox import PDSBSSbase
from algorithm.projection_back import projection_back

EPS=1e-12
THRESHOLD=1e+12
__algorithms_spatial__ = ['IP', 'ISS', 'IPA']

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
    
    def __repr__(self) -> str:
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
    def __init__(self, lr=1e-1, reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        super().__init__(callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.lr = lr
        self.reference_id = reference_id
    
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
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def __repr__(self) -> str:
        s = "GradIVA("
        s += "lr={lr}"
        s += ")"

        return s.format(**self.__dict__)
    
    def update_once(self):
        raise NotImplementedError("Implement 'update_once' function")

    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' function.")

class GradLaplaceIVA(GradIVAbase):
    def __init__(self, lr=1e-1, reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        super().__init__(callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.lr = lr
        self.reference_id = reference_id
    
    def __repr__(self) -> str:
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
    def __init__(self, lr=1e-1, reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        super().__init__(lr=lr, reference_id=reference_id, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.lr = lr
        self.reference_id = reference_id
    
    def __repr__(self) -> str:
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
    """
    def __init__(self, algorithm_spatial='IP', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        super().__init__(callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.algorithm_spatial = algorithm_spatial
        self.reference_id = reference_id
        self.threshold = threshold

        if not self.algorithm_spatial in __algorithms_spatial__:
            raise ValueError("Not support {} based spatial updates.".format(self.algorithm_spatial))
    
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
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def __repr__(self) -> str:
        s = "AuxIVA("
        s += "algorithm_spatial={algorithm_spatial}"
        s += ")"

        return s.format(**self.__dict__)
    
    def update_once(self):
        raise NotImplementedError("Implement 'update_once' function.")

    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' function.")


class AuxLaplaceIVA(AuxIVAbase):
    def __init__(self, algorithm_spatial='IP', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        super().__init__(algorithm_spatial=algorithm_spatial, reference_id=reference_id, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps, threshold=threshold)
    
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

        for idx in range(iteration):
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

        reference_id = self.reference_id
        X, W = input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def __repr__(self) -> str:
        s = "AuxLaplaceIVA("
        s += "algorithm_spatial={algorithm_spatial}"
        s += ")"

        return s.format(**self.__dict__)
    
    def update_once(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins
        eps, threshold = self.eps, self.threshold

        X, Y = self.input, self.estimation

        if self.algorithm_spatial == 'IP':
            W = self.demix_filter
            X = X.transpose(1,2,0) # (n_bins, n_frames, n_channels)
            X = X[...,np.newaxis]
            X_Hermite = X.transpose(0,1,3,2).conj()
            XX = X @ X_Hermite # (n_bins, n_frames, n_channels, n_channels)
            P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)
            R = np.sqrt(P.sum(axis=1))[:,np.newaxis,:,np.newaxis,np.newaxis] # (n_sources, 1, n_frames, 1, 1)
            R[R < eps] = eps
            U = XX / R # (n_sources, n_bins, n_frames, n_channels, n_channels)
            U = U.mean(axis=2) # (n_sources, n_bins, n_channels, n_channels)
            E = np.eye(n_sources, n_channels)
            E = np.tile(E, reps=(n_bins,1,1)) # (n_bins, n_sources, n_channels)

            for source_idx in range(n_sources):
                # W: (n_bins, n_sources, n_channels), U: (n_sources, n_bins, n_channels, n_channels)
                w_n_Hermite = W[:,source_idx,:] # (n_bins, n_channels)
                U_n = U[source_idx] # (n_bins, n_channels, n_channels)
                WU = W @ U_n # (n_bins, n_sources, n_channels)
                condition = np.linalg.cond(WU) < threshold # (n_bins,)
                condition = condition[:,np.newaxis] # (n_bins, 1)
                e_n = E[:,source_idx,:]
                w_n = np.linalg.solve(WU, e_n)
                wUw = w_n[:,np.newaxis,:].conj() @ U_n @ w_n[:,:,np.newaxis]
                denominator = np.sqrt(wUw[...,0])
                w_n_Hermite = np.where(condition, w_n.conj() / denominator, w_n_Hermite)
                # if condition number is too big, `denominator[denominator < eps] = eps` may occur divergence of cost function.
                W[:,source_idx,:] = w_n_Hermite
            
            self.demix_filter = W

            X = self.input
            Y = self.separate(X, demix_filter=W)
        elif self.algorithm_spatial == 'ISS':
            P = np.sum(np.abs(Y)**2, axis=1) # (n_sources, n_frames)
            R = 2 * np.sqrt(P) # (n_sources, n_frames)
            R[R < eps] = eps

            for n in range(n_sources):
                U_n = np.sum(Y * Y[n].conj() / R[:, np.newaxis, :], axis=2) # (n_sources, n_bins)
                D_n = np.sum(np.abs(Y[n])**2 / R[:, np.newaxis, :], axis=2) # (n_sources, n_bins)
                V_n = U_n / D_n # (n_sources, n_bins)
                V_n[n] = 1 - 1 / np.sqrt(D_n[n])
                Y = Y - V_n[:, :, np.newaxis] * Y[n]
            
            if self.recordable_loss:
                # In `update_once()`, demix_filter isn't updated ordinally
                # because we don't have to compute demixing filter explicitly by AuxIVA-ISS.
                self.demix_filter = self.compute_demix_filter(Y, X)
        elif self.algorithm_spatial == 'IPA':
            raise ValueError("in progress...")
        else:
            raise ValueError("Not support {} based spatial updates.".format(self.algorithm_spatial))
        
        self.estimation = Y
    
    def compute_negative_loglikelihood(self):
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        P = np.sum(np.abs(Y)**2, axis=1)
        R = 2 * np.sqrt(P)
        loss = np.sum(R, axis=0).mean() - 2 * np.log(np.abs(np.linalg.det(W))).sum()

        return loss

class AuxGaussIVA(AuxIVAbase):
    def __init__(self, algorithm_spatial='IP', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        super().__init__(algorithm_spatial=algorithm_spatial, reference_id=reference_id, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps, threshold=threshold)
    
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

        for idx in range(iteration):
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

        reference_id = self.reference_id
        X, W = input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output

    def __repr__(self) -> str:
        s = "AuxGaussIVA("
        s += "algorithm_spatial={algorithm_spatial}"
        s += ")"

        return s.format(**self.__dict__)
    
    def update_once(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins
        eps, threshold = self.eps, self.threshold

        X, Y = self.input, self.estimation

        if self.algorithm_spatial == 'IP':
            W = self.demix_filter
            X = X.transpose(1,2,0) # (n_bins, n_frames, n_channels)
            X = X[...,np.newaxis]
            X_Hermite = X.transpose(0,1,3,2).conj()
            XX = X @ X_Hermite # (n_bins, n_frames, n_channels, n_channels)
            P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)
            R = P.mean(axis=1)[:,np.newaxis,:,np.newaxis,np.newaxis] # (n_sources, 1, n_frames, 1, 1)
            R[R < eps] = eps
            U = XX / R # (n_sources, n_bins, n_frames, n_channels, n_channels)
            U = U.mean(axis=2) # (n_sources, n_bins, n_channels, n_channels)
            E = np.eye(n_sources, n_channels)
            E = np.tile(E, reps=(n_bins,1,1)) # (n_bins, n_sources, n_channels)
            for source_idx in range(n_sources):
                # W: (n_bins, n_sources, n_channels), U: (n_sources, n_bins, n_channels, n_channels)
                w_n_Hermite = W[:,source_idx,:] # (n_bins, n_channels)
                U_n = U[source_idx] # (n_bins, n_channels, n_channels)
                WU = W @ U_n # (n_bins, n_sources, n_channels)
                condition = np.linalg.cond(WU) < threshold # (n_bins,)
                condition = condition[:,np.newaxis] # (n_bins, 1)
                e_n = E[:,source_idx,:]
                w_n = np.linalg.solve(WU, e_n)
                wUw = w_n[:,np.newaxis,:].conj() @ U_n @ w_n[:,:,np.newaxis]
                denominator = np.sqrt(wUw[...,0])
                w_n_Hermite = np.where(condition, w_n.conj() / denominator, w_n_Hermite)
                # if condition number is too big, `denominator[denominator < eps] = eps` may occur divergence of cost function.
                W[:,source_idx,:] = w_n_Hermite
            
            self.demix_filter = W

            X = self.input
            Y = self.separate(X, demix_filter=W)
        elif self.algorithm_spatial == 'ISS':
            P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)
            R = P.mean(axis=1) # (n_sources, n_frames)
            R[R < eps] = eps

            for n in range(n_sources):
                U_n = np.sum(Y * Y[n].conj() / R[:, np.newaxis, :], axis=2) # (n_sources, n_bins)
                D_n = np.sum(np.abs(Y[n])**2 / R[:, np.newaxis, :], axis=2) # (n_sources, n_bins)
                V_n = U_n / D_n # (n_sources, n_bins)
                V_n[n] = 1 - 1 / np.sqrt(D_n[n])
                Y = Y - V_n[:, :, np.newaxis] * Y[n]
            
            if self.recordable_loss:
                # In `update_once()`, demix_filter isn't updated ordinally
                # because we don't have to compute demixing filter explicitly by AuxIVA-ISS.
                self.demix_filter = self.compute_demix_filter(Y, X)
        else:
            raise ValueError("Not support {} based spatial updates.".format(self.algorithm_spatial))
        
        self.estimation = Y
    
    def compute_negative_loglikelihood(self):
        X, W = self.input, self.demix_filter
        n_bins = self.n_bins
        eps = self.eps
        Y = self.separate(X, demix_filter=W)
        P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)
        R = P.mean(axis=1) # (n_sources, n_frames)
        R[R < eps] = eps
        loss = (2 * n_bins) * np.sum(np.log(R), axis=0).mean() - 2 * np.log(np.abs(np.linalg.det(W))).sum()

        return loss

class SparseAuxIVA(AuxIVAbase):
    """
    Reference: "A computationally cheaper method for blind speech separation based on AuxIVA and incomplete demixing transform"
    See https://ieeexplore.ieee.org/document/7602921
    """
    def __init__(self, algorithm_spatial='IP', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        super().__init__(algorithm_spatial=algorithm_spatial, reference_id=reference_id, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps, threshold=threshold)

        raise NotImplementedError("in progress")
    
    def update_once(self):
        raise NotImplementedError("in progress...")

class OverIVAbase(AuxIVAbase):
    def __init__(self, algorithm_spatial, reference_id=0, callbacks=None, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        super().__init__(algorithm_spatial=algorithm_spatial, reference_id=reference_id, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps, threshold=threshold)

    def __call__(self, input, iteration=100, **kwargs):

        return super().__call__(input, iteration=iteration, **kwargs)

class ProxLaplaceIVA(PDSBSSbase):
    def __init__(self, regularizer=1, step_prox_logdet=1e+0, step_prox_penalty=1e+0, step=1e+0, reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        super().__init__(regularizer=regularizer, step_prox_logdet=step_prox_logdet, step_prox_penalty=step_prox_penalty, step=step, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.reference_id = reference_id
    
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
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def __repr__(self) -> str:
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
    def __init__(self, regularizer=1, step_prox_logdet=1e+0, step_prox_penalty=1e+0, step=1e+0, reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        super().__init__(regularizer=regularizer, step_prox_logdet=step_prox_logdet, step_prox_penalty=step_prox_penalty, step=step, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

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

def _test_over_iva(method='ProxLaplaceIVA'):
    from transform.pca import pca

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
    n_sources = 2
    
    # STFT
    fft_size, hop_size = 2048, 1024
    mixture = stft(mixed_signal, fft_size=fft_size, hop_size=hop_size)

    # IVA
    n_sources = len(titles)
    iteration = 200

    if method == 'PCAIVA':
        mixture_pca = pca(mixture)

        iva = AuxLaplaceIVA(algorithm_spatial='IP')
        iteration = 50

        estimation = iva(mixture_pca, iteration=iteration)
    elif method == 'OverIVA':
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
    os.makedirs("data/IVA/PCAIVA", exist_ok=True)
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

    print("-"*10, "PCAIVA", "-"*10)
    _test_over_iva(method='PCAIVA')
    print()

    print("-"*10, "OverIVA", "-"*10)
    _test_over_iva(method='OverIVA')
    print()

    print("="*10, "ProxIVA", "="*10)
    print("-"*10, "ProxLaplaceIVA", "-"*10)
    _test_prox_iva(method='ProxLaplaceIVA')
