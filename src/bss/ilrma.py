import warnings
import numpy as np

from utils.utils_linalg import parallel_sort
from transform.stft import stft, istft
from algorithm.projection_back import projection_back

EPS=1e-12
THRESHOLD=1e+12

__algorithms_spatial__ = ['IP', 'IVA', 'ISS', 'IPA', 'pairwise', 'IP1', 'IP2']

"""
References for `algorithm_spatial`:
    IP: "Stable and fast update rules for independent vector analysis based on auxiliary function technique"
        same as `IP1`
    ISS: "Fast and Stable Blind Source Separation with Rank-1 Updates"
    IPA: 
    IP2: "Faster independent low-rank matrix analysiswith pairwise updates of demixing vectors"
"""

class ILRMAbase:
    """
    Independent Low-rank Matrix Analysis
    """
    def __init__(self, n_basis=10, partitioning=False, normalize=True, algorithm_spatial='IP', callbacks=None, recordable_loss=True, eps=EPS):
        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None
        self.eps = eps
        
        self.n_basis = n_basis
        self.partitioning = partitioning
        self.normalize = normalize

        assert algorithm_spatial in __algorithms_spatial__, "Choose from {} as `algorithm_spatial`.".format(__algorithms_spatial__)
        assert algorithm_spatial in ['IP', 'ISS', 'pairwise', 'IP1', 'IP2'], "Not support {}-based demixing filter updates.".format(algorithm_spatial)
        self.algorithm_spatial = algorithm_spatial

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

        n_basis = self.n_basis
        eps = self.eps

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
        
        if self.algorithm_spatial == 'ISS':
            self.demix_filter = None
        
        self.estimation = self.separate(X, demix_filter=W)

        if self.partitioning:
            if not hasattr(self, 'latent'):
                variance_latent = 1e-2
                Z = np.random.rand(n_sources, n_basis) * variance_latent + 1 / n_sources
                Zsum = Z.sum(axis=0)
                Zsum[Zsum < eps] = eps
                self.latent = Z / Zsum
            else:
                self.latent = self.latent.copy()
            if not hasattr(self, 'basis'):
                self.basis = np.random.rand(n_bins, n_basis)
            else:
                self.basis = self.basis.copy()
            if not hasattr(self, 'activation'):
                self.activation = np.random.rand(n_basis, n_frames)
            else:
                self.activation = self.activation.copy()
        else:
            if not hasattr(self, 'basis'):
                self.basis = np.random.rand(n_sources, n_bins, n_basis)
            else:
                self.basis = self.basis.copy()
            if not hasattr(self, 'activation'):
                self.activation = np.random.rand(n_sources, n_basis, n_frames)
            else:
                self.activation = self.activation.copy()
        
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
        s = "ILRMA("
        s += "n_basis={n_basis}"
        s += ", partitioning={partitioning}"
        s += ", normalize={normalize}"
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
        input = input.transpose(1, 0, 2)
        estimation = demix_filter @ input
        output = estimation.transpose(1, 0, 2)

        return output
    
    def compute_demix_filter(self, estimation, input):
        X, Y = input, estimation
        X_Hermite = X.transpose(1, 2, 0).conj()
        XX_Hermite = X.transpose(1, 0, 2) @ X_Hermite
        demix_filter = Y.transpose(1, 0, 2) @ X_Hermite @ np.linalg.inv(XX_Hermite)

        return demix_filter
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' function.")

class GaussILRMA(ILRMAbase):
    """
    Reference: "Determined Blind Source Separation Unifying Independent Vector Analysis and Nonnegative Matrix Factorization"
    See https://ieeexplore.ieee.org/document/7486081
    """
    def __init__(self, n_basis=10, domain=2, partitioning=False, normalize='power', algorithm_spatial='IP', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        """
        Args:
            normalize <str>: 'power': power based normalization, or 'projection-back': projection back based normalization.
            threshold <float>: threshold for condition number when computing (WU)^{-1}.
        """
        super().__init__(n_basis=n_basis, partitioning=partitioning, normalize=normalize, algorithm_spatial=algorithm_spatial, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        assert 1 <= domain <= 2, "1 <= `domain` <= 2 is not satisfied."

        self.domain = domain
        self.reference_id = reference_id
        self.threshold = threshold

        if self.algorithm_spatial == 'ISS':
            warnings.warn("in progress", UserWarning)
        
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
            if self.algorithm_spatial == 'ISS':
                # In `update_once()`, demix_filter isn't updated
                # because we don't have to compute demixing filter explicitly by AuxIVA-ISS.
                X, Y = self.input, self.estimation
                self.demix_filter = self.compute_demix_filter(Y, X)
            
                for callback in self.callbacks:
                    callback(self)
                
                self.demix_filter = None
            else:
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
                if self.algorithm_spatial == 'ISS':
                    # In `update_once()`, demix_filter isn't updated
                    # because we don't have to compute demixing filter explicitly by AuxIVA-ISS.
                    X, Y = self.input, self.estimation
                    self.demix_filter = self.compute_demix_filter(Y, X)
                
                    for callback in self.callbacks:
                        callback(self)
                    
                    self.demix_filter = None
                else:
                    for callback in self.callbacks:
                        callback(self)
        
        if self.algorithm_spatial == 'ISS':
            # In `update_once()`, demix_filter isn't updated
            # because we don't have to compute demixing filter explicitly by AuxIVA-ISS.
            X, Y = self.input, self.estimation
            self.demix_filter = self.compute_demix_filter(Y, X)
        else:
            X, W = input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
        
        reference_id = self.reference_id
        
        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[..., np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output

    def __repr__(self):
        s = "Gauss-ILRMA("
        s += "n_basis={n_basis}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalize={normalize}"
        s += ", algorithm_spatial={algorithm_spatial}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        domain = self.domain
        eps = self.eps

        self.update_source_model()
        self.update_spatial_model()
        
        if self.normalize:
            if self.algorithm_spatial == 'ISS':
                X, Y = self.input, self.estimation
                W = self.compute_demix_filter(Y, X)
            else:
                X, W = self.input, self.demix_filter
                Y = self.separate(X, demix_filter=W)
                self.estimation = Y
            
            T = self.basis

            if self.normalize == 'power':
                P = np.abs(Y)**2
                aux = np.sqrt(P.mean(axis=(1, 2))) # (n_sources,)
                aux[aux < eps] = eps

                # Normalize
                W = W / aux[np.newaxis, :, np.newaxis]
                Y = Y / aux[:, np.newaxis, np.newaxis]

                if self.partitioning:
                    Z = self.latent
                    
                    Zaux = Z / (aux[:, np.newaxis]**domain) # (n_sources, n_basis)
                    Zauxsum = np.sum(Zaux, axis=0) # (n_basis,)
                    T = T * Zauxsum # (n_bins, n_basis)
                    Z = Zaux / Zauxsum # (n_sources, n_basis)
                    self.latent = Z
                else:
                    T = T / (aux[:, np.newaxis, np.newaxis]**domain)
            elif self.normalize == 'projection-back':
                if self.partitioning:
                    raise NotImplementedError("Not support 'projection-back' based normalization for partitioninig function. Choose 'power' based normalization.")
                scale = projection_back(Y, reference=X[self.reference_id])
                Y = Y * scale[..., np.newaxis] # (n_sources, n_bins, n_frames)
                transposed_scale = scale.transpose(1, 0) # (n_sources, n_bins) -> (n_bins, n_sources)
                W = W * transposed_scale[..., np.newaxis] # (n_bins, n_sources, n_channels)
                T = T * np.abs(scale[..., np.newaxis])**domain
            else:
                raise ValueError("Not support normalization based on {}. Choose 'power' or 'projection-back'".format(self.normalize))

            self.estimation = Y
            self.basis = T

            if self.demix_filter is not None:
                self.demix_filter = W
    
    def update_source_model(self):
        if self.algorithm_spatial in ['pairwise', 'IP2']:
            self.update_source_model_pairwise()
        else:
            self.update_source_model_basic()
    
    def update_spatial_model(self):
        if self.algorithm_spatial in ['IP', 'IP1']:
            self.update_spatial_model_ip()
        elif self.algorithm_spatial == 'ISS':
            self.update_spatial_model_iss()
        elif self.algorithm_spatial in ['pairwise', 'IP2']:
            self.update_spatial_model_pairwise()
        else:
            raise NotImplementedError("Not support {}-based spatial update.".format(self.algorithm_spatial))
    
    def update_source_model_basic(self):
        domain = self.domain
        eps = self.eps

        if self.demix_filter is None:
            Y = self.estimation
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
        
        P = np.abs(Y)**2
        
        if self.partitioning:
            assert domain == 2, "Not support domain = {}".format(domain)

            Z = self.latent # (n_sources, n_basis)
            T, V = self.basis, self.activation

            TV = T[:, :, np.newaxis] * V[np.newaxis, :, :] # (n_bins, n_basis, n_frames)
            ZT = Z[:, np.newaxis, :] * T[np.newaxis, :, :] # (n_sources, n_bins, n_basis)
            ZTV = ZT @ V[np.newaxis, :, :] # (n_sources, n_bins, n_frames)
            ZTV[ZTV < eps] = eps
            division, ZTV_inverse = P / (ZTV**2), 1 / ZTV # (n_sources, n_bins, n_frames)
            numerator = np.sum(division[:, :, np.newaxis, :] * TV, axis=(1, 3)) # (n_sources, n_basis)
            denominator = np.sum(ZTV_inverse[:, :, np.newaxis, :] * TV, axis=(1, 3)) # (n_sources, n_basis)
            denominator[denominator < eps] = eps
            Z = np.sqrt(numerator / denominator) # (n_sources, n_basis)
            Z = Z / Z.sum(axis=0) # (n_sources, n_basis)

            # Update basis
            ZT = Z[:, np.newaxis, :] * T[np.newaxis, :, :] # (n_sources, n_bins, n_basis)
            ZTV = ZT @ V[np.newaxis, :, :] # (n_sources, n_bins, n_frames)
            ZTV[ZTV < eps] = eps
            division, ZTV_inverse = P / (ZTV**2), 1 / ZTV # (n_sources, n_bins, n_frames)
            ZV = Z[:, :, np.newaxis] * V[np.newaxis, :, :] # (n_sources, n_basis, n_frames)
            numerator = np.sum(division[:, :, np.newaxis, :] * ZV[:, np.newaxis, :, :], axis=(0, 3)) # (n_bins, n_basis)
            denominator = np.sum(ZTV_inverse[:, :, np.newaxis, :] * ZV[:, np.newaxis, :, :], axis=(0, 3)) # (n_bins, n_basis)
            denominator[denominator < eps] = eps
            T = T * np.sqrt(numerator / denominator) # (n_bins, n_basis)

            # Update activations
            ZT = Z[:, np.newaxis, :] * T[np.newaxis, :, :] # (n_sources, n_bins, n_basis)
            ZTV = ZT @ V[np.newaxis, :, :] # (n_sources, n_bins, n_frames)
            ZTV[ZTV < eps] = eps
            division, ZTV_inverse = P / (ZTV**2), 1 / ZTV # (n_sources, n_bins, n_frames)
            ZT = Z[:, np.newaxis, :] * T[np.newaxis, :, :] # (n_sources, n_bins, n_basis)
            numerator = np.sum(division[:, :, np.newaxis, :] * ZT[:, :, :, np.newaxis], axis=(0, 1)) # (n_basis, n_frames)
            denominator = np.sum(ZTV_inverse[:, :, np.newaxis, :] * ZT[:, :, :, np.newaxis], axis=(0, 1)) # (n_basis, n_frames)
            denominator[denominator < eps] = eps
            V = V * np.sqrt(numerator / denominator) # (n_bins, n_basis)

            self.latent = Z
            self.basis, self.activation = T, V
        else:
            T, V = self.basis, self.activation

            # Update basis
            V_transpose = V.transpose(0, 2, 1)
            TV = T @ V
            TV[TV < eps] = eps
            division, TV_inverse = P / (TV**((domain + 2) / domain)), 1 / TV
            TVV = TV_inverse @ V_transpose
            TVV[TVV < eps] = eps
            T = T * (division @ V_transpose / TVV)**(domain / (domain + 2))
            
            # Update activations
            T_transpose = T.transpose(0, 2, 1)
            TV = T @ V
            TV[TV < eps] = eps
            division, TV_inverse = P / (TV**((domain + 2) / domain)), 1 / TV
            TTV = T_transpose @ TV_inverse
            TTV[TTV < eps] = eps
            V = V * (T_transpose @ division / TTV)**(domain / (domain + 2))

            self.basis, self.activation = T, V

    def update_source_model_pairwise(self):
        domain = self.domain
        m, n = self.update_pair

        eps = self.eps

        if self.demix_filter is None:
            Y = self.estimation
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
        
        Y_m, Y_n = Y[m], Y[n]
        P_m, P_n = np.abs(Y_m)**2, np.abs(Y_n)**2

        T, V = self.basis, self.activation
        T_m, T_n = T[m], T[n] # (n_bins, n_basis), (n_bins, n_basis)
        V_m, V_n = V[m], V[n] # (n_basis, n_frames), (n_basis, n_frames)
        
        if self.partitioning:
            assert domain == 2, "Not support domain = {}".format(domain)
            raise NotImplementedError("Not support partitioning function.")
        else:
            # Update basis
            V_m_transpose, V_n_transpose = V_m.transpose(1, 0), V_n.transpose(1, 0)
            TV_m, TV_n = T_m @ V_m, T_n @ V_n
            TV_m[TV_m < eps], TV_n[TV_n < eps] = eps, eps
            division_m, TV_m_inverse = P_m / (TV_m**((domain + 2) / domain)), 1 / TV_m
            division_n, TV_n_inverse = P_n / (TV_n**((domain + 2) / domain)), 1 / TV_n
            TVV_m, TVV_n = TV_m_inverse @ V_m_transpose, TV_n_inverse @ V_n_transpose
            TVV_m[TVV_m < eps], TVV_n[TVV_n < eps] = eps, eps
            T_m = T_m * (division_m @ V_m_transpose / TVV_m)**(domain / (domain + 2))
            T_n = T_n * (division_n @ V_n_transpose / TVV_n)**(domain / (domain + 2))

            T_m_transpose, T_n_transpose = T_m.transpose(1, 0), T_n.transpose(1, 0)
            TV_m, TV_n = T_m @ V_m, T_n @ V_n
            TV_m[TV_m < eps], TV_n[TV_n < eps] = eps, eps
            division_m, TV_m_inverse = P_m / (TV_m**((domain + 2) / domain)), 1 / TV_m
            division_n, TV_n_inverse = P_n / (TV_n**((domain + 2) / domain)), 1 / TV_n
            TTV_m, TTV_n = T_m_transpose @ TV_m_inverse, T_n_transpose @ TV_n_inverse
            TTV_m[TTV_m < eps], TTV_n[TTV_n < eps] = eps, eps
            V_m = V_m * (T_m_transpose @ division_m / TTV_m)**(domain / (domain + 2))
            V_n = V_n * (T_n_transpose @ division_n / TTV_n)**(domain / (domain + 2))

            T[m] = T_m
            T[n] = T_n
            V[m] = V_m
            V[n] = V_n

            self.basis, self.activation = T, V
    
    def update_spatial_model_ip(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins

        domain = self.domain
        eps, threshold = self.eps, self.threshold

        if self.partitioning:
            assert domain == 2, "Not support domain = {}".format(domain)

            Z = self.latent
            T, V = self.basis, self.activation
            R = np.sum(Z[:, np.newaxis, :, np.newaxis] * T[:, :, np.newaxis] * V[np.newaxis, :, :], axis=2) # (n_sources, n_bins, n_frames)
        else:
            T, V = self.basis, self.activation
            TV = T @ V
            R = TV**(2 / domain) # (n_sources, n_bins, n_frames)

        X = self.input
        W = self.demix_filter
        R = R[..., np.newaxis, np.newaxis] # (n_sources, n_bins, n_frames, 1, 1)
        
        X = X.transpose(1, 2, 0) # (n_bins, n_frames, n_channels)
        X = X[..., np.newaxis]
        X_Hermite = X.transpose(0, 1, 3, 2).conj()
        XX = X @ X_Hermite # (n_bins, n_frames, n_channels, n_channels)
        R[R < eps] = eps
        U = XX / R
        U = U.mean(axis=2) # (n_sources, n_bins, n_channels, n_channels)
        E = np.eye(n_sources, n_channels)
        E = np.tile(E, reps=(n_bins, 1, 1)) # (n_bins, n_sources, n_channels)

        for source_idx in range(n_sources):
            # W: (n_bins, n_sources, n_channels), U: (n_sources, n_bins, n_channels, n_channels)
            w_n_Hermite = W[:, source_idx, :] # (n_bins, n_channels)
            U_n = U[source_idx] # (n_bins, n_channels, n_channels)
            WU = W @ U_n # (n_bins, n_sources, n_channels)
            condition = np.linalg.cond(WU) < threshold # (n_bins,)
            condition = condition[:, np.newaxis] # (n_bins, 1)
            e_n = E[:, source_idx, :]
            w_n = np.linalg.solve(WU, e_n)
            wUw = w_n[:, np.newaxis, :].conj() @ U_n @ w_n[:, :, np.newaxis]
            denominator = np.sqrt(wUw.squeeze(axis=-1))
            w_n_Hermite = np.where(condition, w_n.conj() / denominator, w_n_Hermite)
            # if condition number is too big, `denominator[denominator < eps] = eps` may diverge of cost function.
            W[:, source_idx, :] = w_n_Hermite

        self.demix_filter = W

        X = self.input
        Y = self.separate(X, demix_filter=W)
        
        self.estimation = Y
    
    def update_spatial_model_iss(self):
        n_sources = self.n_sources

        domain = self.domain
        eps = self.eps

        if self.partitioning:
            assert domain == 2, "Not support domain = {}".format(domain)

            Z = self.latent
            T, V = self.basis, self.activation
            R = np.sum(Z[:, np.newaxis, :, np.newaxis] * T[:, :, np.newaxis] * V[np.newaxis, :, :], axis=2) # (n_sources, n_bins, n_frames)
        else:
            T, V = self.basis, self.activation
            TV = T @ V
            R = TV**(2 / domain) # (n_sources, n_bins, n_frames)

        Y = self.estimation
        R[R < eps] = eps # (n_sources, n_bins, n_frames)

        for n in range(n_sources):
            U_n = np.sum(Y * Y[n].conj() / R, axis=2) # (n_sources, n_bins)
            D_n = np.sum(np.abs(Y[n])**2 / R, axis=2) # (n_sources, n_bins)
            V_n = U_n / D_n # (n_sources, n_bins)
            V_n[n] = 1 - 1 / np.sqrt(D_n[n])
            Y = Y - V_n[:, :, np.newaxis] * Y[n]
        
        self.estimation = Y
    
    def update_spatial_model_pairwise(self):
        n_channels = self.n_channels
        n_bins = self.n_bins

        domain = self.domain
        eps, threshold = self.eps, self.threshold

        if self.partitioning:
            assert domain == 2, "Not support domain = {}".format(domain)

            Z = self.latent
            T, V = self.basis, self.activation
            R = np.sum(Z[:, np.newaxis, :, np.newaxis] * T[:, :, np.newaxis] * V[np.newaxis, :, :], axis=2) # (n_sources, n_bins, n_frames)
        else:
            T, V = self.basis, self.activation
            TV = T @ V
            R = TV**(2 / domain) # (n_sources, n_bins, n_frames)

        X = self.input
        m, n = self.update_pair

        W = self.demix_filter
        R_m, R_n = R[m], R[n]
        R_m, R_n = R_m[..., np.newaxis, np.newaxis], R_n[..., np.newaxis, np.newaxis] # (n_bins, n_frames, 1, 1)
            
        X = X.transpose(1, 2, 0) # (n_bins, n_frames, n_channels)
        X = X[..., np.newaxis]
        X_Hermite = X.transpose(0, 1, 3, 2).conj()
        XX = X @ X_Hermite # (n_bins, n_frames, n_channels, n_channels)
        R_m[R_m < eps] = eps
        R_n[R_n < eps] = eps
        U_m, U_n = XX / R_m, XX / R_n
        U_m, U_n = U_m.mean(axis=1), U_n.mean(axis=1) # (n_bins, n_channels, n_channels)
        e_m, e_n = np.zeros((n_bins, n_channels, 1)), np.zeros((n_bins, n_channels, 1))
        e_m[:, m, :] = 1
        e_n[:, n, :] = 1
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
        n_frames = self.n_frames
        domain = self.domain
        eps = self.eps

        X = self.input

        if self.demix_filter is None:
            Y = self.estimation
            W = self.compute_demix_filter(Y, X)
        else:
            W = self.demix_filter
            Y = self.separate(X, demix_filter=W)
        
        P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation
            ZTV = np.sum(Z[:, np.newaxis, :, np.newaxis] * T[:, :, np.newaxis] * V[np.newaxis, :, :], axis=2) # (n_sources, n_bins, n_frames)
            R = ZTV**(2 / domain)
        else:
            T, V = self.basis, self.activation
            TV = T @ V # (n_sources, n_bins, n_frames)
            R = TV**(2 / domain)
        
        R[R < eps] = eps
        loss = np.sum(P / R + np.log(R)) - 2 * n_frames * np.sum(np.log(np.abs(np.linalg.det(W))))

        return loss

class GGDILRMA(ILRMAbase):
    """
    Reference: "Generalized independent low-rank matrix analysis using heavy-tailed distributions for blind source separation"
    See: https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-018-0549-5
    """
    def __init__(self, n_basis=10, beta=1, domain=2, partitioning=False, normalize='power', algorithm_spatial='IP', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        """
        Args:
            beta: shape parameter. beta = 1: Laplace distribution, beta = 2: Gaussian distribution.
            normalize <str>: 'power': power based normalization, or 'projection-back': projection back based normalization.
            threshold <float>: threshold for condition number when computing (WU)^{-1}.
        """
        super().__init__(n_basis=n_basis, partitioning=partitioning, normalize=normalize, algorithm_spatial=algorithm_spatial, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.beta = beta
        self.domain = domain
        self.reference_id = reference_id

        assert self.algorithm_spatial == 'IP', "Supports only IP-based spatial update."

        raise NotImplementedError("Implement GGD-ILRMA")
    
    def __repr__(self):
        s = "GGD-ILRMA("
        s += "n_basis={n_basis}"
        s += ", beta={beta}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalize={normalize}"
        s += ", algorithm_spatial={algorithm_spatial}"
        s += ")"

        return s.format(**self.__dict__)

class tILRMA(ILRMAbase):
    """
    Reference: "Independent low-rank matrix analysis based on complex student's t-distribution for blind audio source separation"
    See: https://ieeexplore.ieee.org/document/8168129
    """
    def __init__(self, n_basis=10, nu=1, domain=2, partitioning=False, normalize='power', algorithm_spatial='IP', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        """
        Args:
            nu: degree of freedom. nu = 1: Cauchy distribution, nu -> infty: Gaussian distribution.
            normalize <str>: 'power': power based normalization, or 'projection-back': projection back based normalization.
            threshold <float>: threshold for condition number when computing (WU)^{-1}.
        """
        super().__init__(n_basis=n_basis, partitioning=partitioning, normalize=normalize, algorithm_spatial=algorithm_spatial, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.nu = nu
        self.domain = domain
        self.reference_id = reference_id

        assert self.algorithm_spatial == 'IP', "Supports only IP-based spatial update."
    
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
                
                self.demix_filter = None
            else:
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
                    
                    self.demix_filter = None
                else:
                    for callback in self.callbacks:
                        callback(self)
        
        reference_id = self.reference_id

        if self.algorithm_spatial == 'ISS':
            # In `update_once()`, demix_filter isn't updated
            # because we don't have to compute demixing filter explicitly by AuxIVA-ISS.
            X, Y = self.input, self.estimation
            self.demix_filter = self.compute_demix_filter(Y, X)
        else:
            X, W = input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[..., np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def __repr__(self):
        s = "t-ILRMA("
        s += "n_basis={n_basis}"
        s += ", nu={nu}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalize={normalize}"
        s += ", algorithm_spatial={algorithm_spatial}"
        s += ")"

        return s.format(**self.__dict__)
    
    def update_once(self):
        eps = self.eps

        self.update_source_model()
        self.update_spatial_model()

        if self.normalize:
            if self.algorithm_spatial == 'ISS':
                X, Y = self.input, self.estimation
                W = self.compute_demix_filter(Y, X)
            else:
                X, W = self.input, self.demix_filter
                Y = self.separate(X, demix_filter=W)
                self.estimation = Y

            if self.normalize == 'power':
                P = np.abs(Y)**2
                aux = np.sqrt(P.mean(axis=(1, 2))) # (n_sources,)
                aux[aux < eps] = eps

                # Normalize
                W = W / aux[np.newaxis, :, np.newaxis]
                Y = Y / aux[:, np.newaxis, np.newaxis]

                if self.partitioning:
                    Z = self.latent
                    T = self.basis
                    Zaux = Z / (aux[:, np.newaxis]**2) # (n_sources, n_basis)
                    Zauxsum = np.sum(Zaux, axis=0) # (n_basis,)
                    T = T * Zauxsum # (n_bins, n_basis)
                    Z = Zaux / Zauxsum # (n_sources, n_basis)
                    self.latent = Z
                    self.basis = T
                else:
                    T = self.basis
                    T = T / (aux[:, np.newaxis, np.newaxis]**2)
                    self.basis = T
            else:
                raise ValueError("Not support normalization based on {}. Choose 'power' or 'projection-back'".format(self.normalize))
            
            self.estimation = Y

            if self.demix_filter is not None:
                self.demix_filter = W

    def update_source_model(self):
        nu = self.nu
        domain = self.domain
        eps = self.eps

        assert domain == 2, "Only domain = 2 is supported."

        if self.demix_filter is None:
            Y = self.estimation
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        P = np.abs(Y)**2
        
        if self.partitioning:
            raise NotImplementedError("Only support when `partitioning=False` ")
            Z = self.latent # (n_sources, n_basis)
            T, V = self.basis, self.activation

            TV = T[:,:,np.newaxis] * V[np.newaxis,:,:] # (n_bins, n_basis, n_frames)
            ZT = Z[:,np.newaxis,:] * T[np.newaxis,:,:] # (n_sources, n_bins, n_basis)
            ZTV = ZT @ V[np.newaxis,:,:] # (n_sources, n_bins, n_frames)
            ZTV[ZTV < eps] = eps
            division, ZTV_inverse = P / (ZTV**2), 1 / ZTV # (n_sources, n_bins, n_frames)
            numerator = np.sum(division[:,:,np.newaxis,:,] * TV, axis=(1,3)) # (n_sources, n_basis)
            denominator = np.sum(ZTV_inverse[:,:,np.newaxis,:,] * TV, axis=(1,3)) # (n_sources, n_basis)
            denominator[denominator < eps] = eps
            Z = np.sqrt(numerator / denominator) # (n_sources, n_basis)
            Zsum = Z.sum(axis=0)
            Z = Z / Zsum # (n_sources, n_basis)

            # Update basis
            ZT = Z[:,np.newaxis,:] * T[np.newaxis,:,:] # (n_sources, n_bins, n_basis)
            ZTV = ZT @ V[np.newaxis,:,:] # (n_sources, n_bins, n_frames)
            ZTV[ZTV < eps] = eps
            division, ZTV_inverse = P / (ZTV**2), 1 / ZTV # (n_sources, n_bins, n_frames)
            ZV = Z[:,:,np.newaxis] * V[np.newaxis,:,:] # (n_sources, n_basis, n_frames)
            numerator = np.sum(division[:,:,np.newaxis,:] * ZV[:,np.newaxis,:,:], axis=(0,3)) # (n_bins, n_basis)
            denominator = np.sum(ZTV_inverse[:,:,np.newaxis,:] * ZV[:,np.newaxis,:,:], axis=(0,3)) # (n_bins, n_basis)
            denominator[denominator < eps] = eps
            T = T * np.sqrt(numerator / denominator) # (n_bins, n_basis)

            # Update activations
            ZT = Z[:,np.newaxis,:] * T[np.newaxis,:,:] # (n_sources, n_bins, n_basis)
            ZTV = ZT @ V[np.newaxis,:,:] # (n_sources, n_bins, n_frames)
            ZTV[ZTV < eps] = eps
            division, ZTV_inverse = P / (ZTV**2), 1 / ZTV # (n_sources, n_bins, n_frames)
            ZT = Z[:,np.newaxis,:] * T[np.newaxis,:,:] # (n_sources, n_bins, n_basis)
            numerator = np.sum(division[:,:,np.newaxis,:] * ZT[:,:,:,np.newaxis], axis=(0,1)) # (n_basis, n_frames)
            denominator = np.sum(ZTV_inverse[:,:,np.newaxis,:] * ZT[:,:,:,np.newaxis], axis=(0,1)) # (n_basis, n_frames)
            denominator[denominator < eps] = eps
            V = V * np.sqrt(numerator / denominator) # (n_bins, n_basis)

            self.latent = Z
            self.basis, self.activation = T, V
        else:
            T, V = self.basis, self.activation # (n_sources, n_bins, n_basis), (n_sources, n_basis, n_frames)

            # Update basis
            V_transpose = V.transpose(0, 2, 1) # (n_sources, n_frames, n_basis)
            TV = T @ V # (n_sources, n_bins, n_frames)
            TV[TV < eps] = eps
            harmonic = 1 / (2 / ((2 + nu) * TV) + nu / ((2 + nu) * P))
            division, TV_inverse = harmonic / (TV**2), 1 / TV # (n_sources, n_bins, n_frames), (n_sources, n_bins, n_frames)
            TVV = TV_inverse @ V_transpose # (n_sources, n_bins, n_basis)
            TVV[TVV < eps] = eps
            T = T * np.sqrt(division @ V_transpose / TVV) # (n_sources, n_bins, n_basis)

            # Update activations
            T_transpose = T.transpose(0, 2, 1) # (n_sources, n_basis, n_bins)
            TV = T @ V # (n_sources, n_bins, n_frames)
            TV[TV < eps] = eps
            harmonic = 1 / (2 / ((2 + nu) * TV) + nu / ((2 + nu) * P))
            division, TV_inverse = harmonic / (TV**2), 1 / TV # (n_sources, n_bins, n_frames)
            TTV = T_transpose @ TV_inverse # (n_sources, n_basis, n_frames)
            TTV[TTV < eps] = eps
            V = V * np.sqrt(T_transpose @ division / TTV) # (n_sources, n_basis, n_frames)

            self.basis, self.activation = T, V
    
    def update_spatial_model(self):
        n_sources = self.n_sources
        nu = self.nu
        eps = self.eps

        if self.demix_filter is None:
            Y = self.estimation
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)
        
        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation
            R = np.sum(Z[:, np.newaxis, :, np.newaxis] * T[:, :, np.newaxis] * V[np.newaxis, :, :], axis=2) # (n_sources, n_bins, n_frames)
        else:
            T, V = self.basis, self.activation
            R = T @ V
        
        if self.algorithm_spatial == 'IP':
            R = R[..., np.newaxis, np.newaxis] # (n_sources, n_bins, n_frames, 1, 1)
            R[R < eps] = eps
            Xi = (nu * R + 2 * P[..., np.newaxis, np.newaxis]) / (nu + 2)

            X = X.transpose(1, 2, 0) # (n_bins, n_frames, n_channels)
            X = X[..., np.newaxis]
            X_Hermite = X.transpose(0, 1, 3, 2).conj()
            XX = X @ X_Hermite # (n_bins, n_frames, n_channels, n_channels)
            U = XX / Xi
            U = U.mean(axis=2) # (n_sources, n_bins, n_channels, n_channels)

            for source_idx in range(n_sources):
                # W: (n_bins, n_sources, n_channels), U: (n_sources, n_bins, n_channels, n_channels)
                U_n = U[source_idx] # (n_bins, n_channels, n_channels)
                WU = W @ U_n # (n_bins, n_sources, n_channels)
                WU_inverse = np.linalg.inv(WU) # (n_bins, n_sources, n_channels)
                w = WU_inverse[..., source_idx] # (n_bins, n_channels)
                wUw = w[:, np.newaxis, :].conj() @ U_n @ w[:, :, np.newaxis]
                denominator = np.sqrt(wUw.squeeze(axis=-1))
                denominator[denominator < eps] = eps
                W[:, source_idx, :] = w.conj() / denominator

            self.demix_filter = W

            X = self.input
            Y = self.separate(X, demix_filter=W)
            
            self.estimation = Y
        else:
            raise ValueError("Not support {}-based spatial update.")

    def compute_negative_loglikelihood(self):
        n_frames = self.n_frames
        nu = self.nu
        eps = self.eps

        X = self.input

        if self.demix_filter is None:
            Y = self.estimation
            W = self.compute_demix_filter(Y, X)
        else:
            W = self.demix_filter
            Y = self.separate(X, demix_filter=W)

        P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)

        if self.partitioning:
            Z = self.latent
            T, V = self.basis, self.activation
            R = np.sum(Z[:, np.newaxis, :, np.newaxis] * T[:, :, np.newaxis] * V[np.newaxis, :, :], axis=2) # (n_sources, n_bins, n_frames)
        else:
            T, V = self.basis, self.activation
            R = T @ V # (n_sources, n_bins, n_frames)
        
        R[R < eps] = eps
        loss = np.sum((1 + nu / 2) * np.log(1 + (2 / nu) * (P / R)) + np.log(R)) - 2 * n_frames * np.sum(np.log(np.abs(np.linalg.det(W))))

        return loss

class KLILRMA(ILRMAbase):
    """
    Reference: "Independent Low-Rank Matrix Analysis Based on Generalized Kullback-Leibler Divergence"
    """
    def __init__(self, n_basis=10, partitioning=False, normalize='power', algorithm_spatial='IP', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        super().__init__(n_basis=n_basis, partitioning=partitioning, normalize=normalize, algorithm_spatial=algorithm_spatial, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.reference_id = reference_id

        assert self.algorithm_spatial == 'IP', "Supports only IP-based spatial update."

        raise NotImplementedError("Implement KL-ILRMA")
    
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
        
        reference_id = self.reference_id
        X, W = input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[..., np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def __repr__(self):
        s = "KL-ILRMA("
        s += "n_basis={n_basis}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalize={normalize}"
        s += ", algorithm_spatial={algorithm_spatial}"
        s += ")"

        return s.format(**self.__dict__)

class RegularizedILRMA(ILRMAbase):
    """
    Reference: "Blind source separation based on independent low-rank matrix analysis with sparse regularization for time-series activity"
    See https://ieeexplore.ieee.org/document/7486081
    """
    def __init__(self, n_basis=10, partitioning=False, normalize='power', algorithm_spatial='IP', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        """
        Args:
            normalize <str>
        """
        super().__init__(n_basis=n_basis, partitioning=partitioning, normalize=normalize, algorithm_spatial=algorithm_spatial, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.reference_id = reference_id

        assert self.algorithm_spatial == 'IP', "Supports only IP-based spatial update."

        raise NotImplementedError("Implement Regularized ILRMA")

class ConsistentGaussILRMA(GaussILRMA):
    """
    Reference: "Consistent independent low-rank matrix analysis for determined blind source separation"
    See https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-020-00704-4
    """
    def __init__(self, n_basis=10, partitioning=False, algorithm_spatial='IP', reference_id=0, fft_size=None, hop_size=None, callbacks=None, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        """
        Args:
            normalize <str>: 'power': power based normalization, or 'projection-back': projection back based normalization.
            threshold <float>: threshold for condition number when computing (WU)^{-1}.
        """
        super().__init__(n_basis=n_basis, partitioning=partitioning, normalize=False, algorithm_spatial=algorithm_spatial, reference_id=reference_id, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps, threshold=threshold)

        if fft_size is None:
            raise ValueError("Specify `fft_size`.")
        
        if hop_size is None:
            hop_size = fft_size // 2
        
        self.fft_size, self.hop_size = fft_size, hop_size

        assert self.algorithm_spatial == 'IP', "Supports only IP-based spatial update."

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
                
                self.demix_filter = None
            else:
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
                    
                    self.demix_filter = None
                else:
                    for callback in self.callbacks:
                        callback(self)
        
        reference_id = self.reference_id

        if self.algorithm_spatial == 'ISS':
            # In `update_once()`, demix_filter isn't updated
            # because we don't have to compute demixing filter explicitly by AuxIVA-ISS.
            X, Y = self.input, self.estimation
            self.demix_filter = self.compute_demix_filter(Y, X)
        else:
            X, W = input, self.demix_filter
            Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[..., np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def __repr__(self):
        s = "Consistent-GaussILRMA("
        s += "n_basis={n_basis}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalize={normalize}"
        s += ", algorithm_spatial={algorithm_spatial}"
        s += ")"

        return s.format(**self.__dict__)
    
    def update_once(self):
        y = istft(self.estimation, fft_size=self.fft_size, hop_size=self.hop_size)
        self.estimation = stft(y, fft_size=self.fft_size, hop_size=self.hop_size)

        self.update_source_model()
        self.update_spatial_model()

        if self.algorithm_spatial == 'ISS':
            X, Y = self.input, self.estimation
            W = self.compute_demix_filter(Y, X)
        else:
            X, W = self.input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
        
        T = self.basis

        if self.partitioning:
            raise NotImplementedError("Not support 'projection-back' based normalization for partitioninig function. Choose 'power' based normalization.")
        scale = projection_back(Y, reference=X[self.reference_id])
        transposed_scale = scale.transpose(1, 0) # (n_sources, n_bins) -> (n_bins, n_sources)
        W = W * transposed_scale[..., np.newaxis] # (n_bins, n_sources, n_channels)
        Y = self.separate(X, demix_filter=W)
        T = T * np.abs(scale[..., np.newaxis])**2

        self.estimation = Y
        self.basis = T

        if self.demix_filter is not None:
            self.demix_filter = W

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

def _test_gauss_ilrma(algorithm_spatial, n_basis=10, domain=2, partitioning=False):
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

    n_sources, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 2048, 1024
    mixture = stft(mixed_signal, fft_size=fft_size, hop_size=hop_size)

    # ILRMA
    n_channels = len(titles)
    iteration = 50

    ilrma = GaussILRMA(n_basis=n_basis, domain=domain, partitioning=partitioning, algorithm_spatial=algorithm_spatial)
    print(ilrma)
    estimation = ilrma(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_channels):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/ILRMA/GaussILRMA-{}/partitioning{}/mixture-{}_estimated-iter{}-{}.wav".format(algorithm_spatial, int(partitioning), sr, iteration, idx), signal=_estimated_signal, sr=sr)
    
    plt.figure()
    plt.plot(ilrma.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/ILRMA/GaussILRMA-{}/partitioning{}/loss.png'.format(algorithm_spatial, int(partitioning)), bbox_inches='tight')
    plt.close()

def _test_t_ilrma(algorithm_spatial, n_basis=10, partitioning=False):
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

    n_sources, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 2048, 1024
    mixture = stft(mixed_signal, fft_size=fft_size, hop_size=hop_size)

    # t-ILRMA
    n_channels = len(titles)
    iteration = 50
    nu = 1000
    ilrma = tILRMA(n_basis=n_basis, nu=nu, partitioning=partitioning, algorithm_spatial=algorithm_spatial)
    print(ilrma)
    estimation = ilrma(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_channels):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/ILRMA/tILRMA-{}/partitioning{}/mixture-{}_estimated-iter{}-{}.wav".format(algorithm_spatial, int(partitioning), sr, iteration, idx), signal=_estimated_signal, sr=sr)
    
    plt.figure()
    plt.plot(ilrma.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/ILRMA/tILRMA-{}/partitioning{}/loss.png'.format(algorithm_spatial, int(partitioning)), bbox_inches='tight')
    plt.close()

def _test_consistent_ilrma(n_basis=10, partitioning=False):
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

    n_sources, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 2048, 1024
    mixture = stft(mixed_signal, fft_size=fft_size, hop_size=hop_size)

    # ILRMA
    n_channels = len(titles)
    iteration = 50
    
    ilrma = ConsistentGaussILRMA(n_basis=n_basis, partitioning=partitioning, fft_size=fft_size, hop_size=hop_size)
    estimation = ilrma(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_channels):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/ILRMA/ConsistentGaussILRMA/partitioning{}/mixture-{}_estimated-iter{}-{}.wav".format(int(partitioning), sr, iteration, idx), signal=_estimated_signal, sr=sr)
    
    plt.figure()
    plt.plot(ilrma.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/ILRMA/ConsistentGaussILRMA/partitioning{}/loss.png'.format(int(partitioning)), bbox_inches='tight')
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

    plt.rcParams['figure.dpi'] = 200

    os.makedirs("data/multi-channel", exist_ok=True)
    os.makedirs("data/ILRMA/GaussILRMA-IP/partitioning0", exist_ok=True)
    os.makedirs("data/ILRMA/GaussILRMA-IP/partitioning1", exist_ok=True)
    os.makedirs("data/ILRMA/GaussILRMA-ISS/partitioning0", exist_ok=True)
    os.makedirs("data/ILRMA/GaussILRMA-ISS/partitioning1", exist_ok=True)
    os.makedirs("data/ILRMA/tILRMA-IP/partitioning0", exist_ok=True)
    os.makedirs("data/ILRMA/tILRMA-IP/partitioning1", exist_ok=True)
    os.makedirs("data/ILRMA/tILRMA-ISS/partitioning0", exist_ok=True)
    os.makedirs("data/ILRMA/tILRMA-ISS/partitioning1", exist_ok=True)
    os.makedirs("data/ILRMA/ConsistentGaussILRMA/partitioning0", exist_ok=True)
    os.makedirs("data/ILRMA/ConsistentGaussILRMA/partitioning1", exist_ok=True)

    """
    Use multichannel room impulse response database.
    Download database from "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/"
    """

    _test_conv()

    print("="*10, "Gauss-ILRMA (IP)", "="*10)
    print("-"*10, "without partitioning function", "-"*10)
    _test_gauss_ilrma(algorithm_spatial='IP', n_basis=2, partitioning=False)
    print()

    print("-"*10, "with partitioning function", "-"*10)
    _test_gauss_ilrma(algorithm_spatial='IP', n_basis=5, partitioning=True)
    print()

    print("="*10, "Gauss-ILRMA (ISS)", "="*10)
    print("-"*10, "without partitioning function", "-"*10)
    _test_gauss_ilrma(algorithm_spatial='ISS', n_basis=2, partitioning=False)
    print()

    print("-"*10, "with partitioning function", "-"*10)
    _test_gauss_ilrma(algorithm_spatial='ISS', n_basis=5, partitioning=True)
    print()

    print("="*10, "t-ILRMA (IP)", "="*10)
    print("-"*10, "without partitioning function", "-"*10)
    _test_t_ilrma(algorithm_spatial='IP', n_basis=2, partitioning=False)
    print()
    # _test_t_ilrma(algorithm_spatial='IP', n_basis=5, partitioning=True)
    print()

    print("="*10, "t-ILRMA (ISS)", "="*10)
    print("-"*10, "without partitioning function", "-"*10)
    # _test_t_ilrma(algorithm_spatial='ISS', n_basis=2, partitioning=False)
    print()
    # _test_t_ilrma(algorithm_spatial='ISS', n_basis=5, partitioning=True)
    print()

    print("="*10, "Consistent-ILRMA", "="*10)
    _test_consistent_ilrma(n_basis=5, partitioning=False)
    print()