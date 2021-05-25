import numpy as np

from algorithm.stft import stft, istft
from algorithm.projection_back import projection_back

EPS=1e-12
THRESHOLD=1e+12

class ILRMAbase:
    """
    Independent Low-rank Matrix Analysis
    """
    def __init__(self, n_bases=10, partitioning=False, normalize=True, callbacks=None, recordable_loss=True, eps=EPS):
        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None
        self.eps = eps
        
        self.n_bases = n_bases
        self.partitioning = partitioning
        self.normalize = normalize

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

        n_bases = self.n_bases

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
        
        X, W = input, self.demix_filter
        output = self.separate(X, demix_filter=W)

        return output
    
    def __repr__(self) -> str:
        s = "ILRMA("
        s += "n_bases={n_bases}"
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
        input = input.transpose(1,0,2)
        estimation = demix_filter @ input
        output = estimation.transpose(1,0,2)

        return output
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' function.")

class GaussILRMA(ILRMAbase):
    """
    Reference: "Determined Blind Source Separation Unifying Independent Vector Analysis and Nonnegative Matrix Factorization"
    See https://ieeexplore.ieee.org/document/7486081
    """
    def __init__(self, n_bases=10, domain=2, partitioning=False, normalize='power', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        """
        Args:
            normalize <str>: 'power': power based normalization, or 'projection-back': projection back based normalization.
            threshold <float>: threshold for condition number when computing (WU)^{-1}.
        """
        super().__init__(n_bases=n_bases, partitioning=partitioning, normalize=normalize, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        assert 1 <= domain <= 2, "1 <= `domain` <= 2 is not satisfied."

        self.domain = domain
        self.reference_id = reference_id
        self.threshold = threshold
    
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
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output

    def __repr__(self) -> str:
        s = "Gauss-ILRMA("
        s += "n_bases={n_bases}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalize={normalize}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        eps = self.eps

        self.update_source_model()
        self.update_space_model()

        X, W = self.input, self.demix_filter
        T = self.base

        Y = self.separate(X, demix_filter=W)
        self.estimation = Y
        
        if self.normalize:
            if self.normalize == 'power':
                P = np.abs(Y)**2
                aux = np.sqrt(P.mean(axis=(1,2))) # (n_sources,)
                aux[aux < eps] = eps

                # Normalize
                W = W / aux[np.newaxis,:,np.newaxis]
                Y = Y / aux[:,np.newaxis,np.newaxis]

                if self.partitioning:
                    Z = self.latent
                    
                    Zaux = Z / (aux[:,np.newaxis]**2) # (n_sources, n_bases)
                    Zauxsum = np.sum(Zaux, axis=0) # (n_bases,)
                    T = T * Zauxsum # (n_bins, n_bases)
                    Z = Zaux / Zauxsum # (n_sources, n_bases)
                    self.latent = Z
                else:
                    T = T / (aux[:,np.newaxis,np.newaxis]**2)
            elif self.normalize == 'projection-back':
                if self.partitioning:
                    raise NotImplementedError("Not support 'projection-back' based normalization for partitioninig function. Choose 'power' based normalization.")
                domain = self.domain
                scale = projection_back(Y, reference=X[self.reference_id])
                Y = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
                transposed_scale = scale.transpose(1,0) # (n_sources, n_bins) -> (n_bins, n_sources)
                W = W * transposed_scale[...,np.newaxis] # (n_bins, n_sources, n_channels)
                T = T * np.abs(scale[...,np.newaxis])**domain
            else:
                raise ValueError("Not support normalization based on {}. Choose 'power' or 'projection-back'".format(self.normalize))

            self.demix_filter = W
            self.estimation = Y
            self.base = T
    
    def update_source_model(self):
        domain = self.domain
        eps = self.eps

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        P = np.abs(Y)**2
        
        if self.partitioning:
            assert domain == 2, "Not support domain = {}".format(domain)

            Z = self.latent # (n_sources, n_bases)
            T, V = self.base, self.activation

            TV = T[:,:,np.newaxis] * V[np.newaxis,:,:] # (n_bins, n_bases, n_frames)
            ZT = Z[:,np.newaxis,:] * T[np.newaxis,:,:] # (n_sources, n_bins, n_bases)
            ZTV = ZT @ V[np.newaxis,:,:] # (n_sources, n_bins, n_frames)
            ZTV[ZTV < eps] = eps
            division, ZTV_inverse = P / (ZTV**2), 1 / ZTV # (n_sources, n_bins, n_frames)
            numerator = np.sum(division[:,:,np.newaxis,:,] * TV, axis=(1,3)) # (n_sources, n_bases)
            denominator = np.sum(ZTV_inverse[:,:,np.newaxis,:,] * TV, axis=(1,3)) # (n_sources, n_bases)
            denominator[denominator < eps] = eps
            Z = np.sqrt(numerator / denominator) # (n_sources, n_bases)
            Z = Z / Z.sum(axis=0) # (n_sources, n_bases)

            # Update bases
            ZT = Z[:,np.newaxis,:] * T[np.newaxis,:,:] # (n_sources, n_bins, n_bases)
            ZTV = ZT @ V[np.newaxis,:,:] # (n_sources, n_bins, n_frames)
            ZTV[ZTV < eps] = eps
            division, ZTV_inverse = P / (ZTV**2), 1 / ZTV # (n_sources, n_bins, n_frames)
            ZV = Z[:,:,np.newaxis] * V[np.newaxis,:,:] # (n_sources, n_bases, n_frames)
            numerator = np.sum(division[:,:,np.newaxis,:] * ZV[:,np.newaxis,:,:], axis=(0,3)) # (n_bins, n_bases)
            denominator = np.sum(ZTV_inverse[:,:,np.newaxis,:] * ZV[:,np.newaxis,:,:], axis=(0,3)) # (n_bins, n_bases)
            denominator[denominator < eps] = eps
            T = T * np.sqrt(numerator / denominator) # (n_bins, n_bases)

            # Update activations
            ZT = Z[:,np.newaxis,:] * T[np.newaxis,:,:] # (n_sources, n_bins, n_bases)
            ZTV = ZT @ V[np.newaxis,:,:] # (n_sources, n_bins, n_frames)
            ZTV[ZTV < eps] = eps
            division, ZTV_inverse = P / (ZTV**2), 1 / ZTV # (n_sources, n_bins, n_frames)
            ZT = Z[:,np.newaxis,:] * T[np.newaxis,:,:] # (n_sources, n_bins, n_bases)
            numerator = np.sum(division[:,:,np.newaxis,:] * ZT[:,:,:,np.newaxis], axis=(0,1)) # (n_bases, n_frames)
            denominator = np.sum(ZTV_inverse[:,:,np.newaxis,:] * ZT[:,:,:,np.newaxis], axis=(0,1)) # (n_bases, n_frames)
            denominator[denominator < eps] = eps
            V = V * np.sqrt(numerator / denominator) # (n_bins, n_bases)

            self.latent = Z
            self.base, self.activation = T, V
        else:
            T, V = self.base, self.activation

            # Update bases
            V_transpose = V.transpose(0,2,1)
            TV = T @ V
            TV[TV < eps] = eps
            division, TV_inverse = P / (TV**((domain + 2) / domain)), 1 / TV
            TVV = TV_inverse @ V_transpose
            TVV[TVV < eps] = eps
            T = T * (division @ V_transpose / TVV)**(domain / (domain + 2))
            
            # Update activations
            T_transpose = T.transpose(0,2,1)
            TV = T @ V
            TV[TV < eps] = eps
            division, TV_inverse = P / (TV**((domain + 2) / domain)), 1 / TV
            TTV = T_transpose @ TV_inverse
            TTV[TTV < eps] = eps
            V = V * (T_transpose @ division / TTV)**(domain / (domain + 2))

            self.base, self.activation = T, V

    def update_space_model(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins

        domain = self.domain
        eps, threshold = self.eps, self.threshold

        X, W = self.input, self.demix_filter
        
        if self.partitioning:
            assert domain == 2, "Not support domain = {}".format(domain)

            Z = self.latent
            T, V = self.base, self.activation
            ZTV = np.sum(Z[:,np.newaxis,:,np.newaxis] * T[:,:,np.newaxis] * V[np.newaxis,:,:], axis=2) # (n_sources, n_bins, n_frames)
            R = ZTV[...,np.newaxis, np.newaxis] # (n_sources, n_bins, n_frames, 1, 1)
        else:
            T, V = self.base, self.activation
            TV = T @ V
            R = (TV[...,np.newaxis, np.newaxis])**(2 / domain) # (n_sources, n_bins, n_frames, 1, 1)
        
        X = X.transpose(1,2,0) # (n_bins, n_frames, n_channels)
        X = X[...,np.newaxis]
        X_Hermite = X.transpose(0,1,3,2).conj()
        XX = X @ X_Hermite # (n_bins, n_frames, n_channels, n_channels)
        R[R < eps] = eps
        U = XX / R
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
            # if condition number is too big, `denominator[denominator < eps] = eps` may diverge of cost function.
            W[:,source_idx,:] = w_n_Hermite

        self.demix_filter = W

    def compute_negative_loglikelihood(self):
        n_frames = self.n_frames
        domain = self.domain
        eps = self.eps

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)

        if self.partitioning:
            Z = self.latent
            T, V = self.base, self.activation
            ZTV = np.sum(Z[:,np.newaxis,:,np.newaxis] * T[:,:,np.newaxis] * V[np.newaxis,:,:], axis=2) # (n_sources, n_bins, n_frames)
            R = ZTV**(2 / domain)
        else:
            T, V = self.base, self.activation
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
    def __init__(self, n_bases=10, beta=1, domain=2, partitioning=False, normalize='power', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        """
        Args:
            beta: shape parameter. beta = 1: Laplace distribution, beta = 2: Gaussian distribution.
            normalize <str>: 'power': power based normalization, or 'projection-back': projection back based normalization.
            threshold <float>: threshold for condition number when computing (WU)^{-1}.
        """
        super().__init__(n_bases=n_bases, partitioning=partitioning, normalize=normalize, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.beta = beta
        self.domain = domain
        self.reference_id = reference_id

        raise NotImplementedError("Implement GGD-ILRMA")
    
    def __repr__(self) -> str:
        s = "GGD-ILRMA("
        s += "n_bases={n_bases}"
        s += ", beta={beta}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalize={normalize}"
        s += ")"

        return s.format(**self.__dict__)

class tILRMA(ILRMAbase):
    """
    Reference: "Independent low-rank matrix analysis based on complex student's t-distribution for blind audio source separation"
    See: https://ieeexplore.ieee.org/document/8168129
    """
    def __init__(self, n_bases=10, nu=1, domain=2, partitioning=False, normalize='power', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        """
        Args:
            nu: degree of freedom. nu = 1: Cauchy distribution, nu -> infty: Gaussian distribution.
            normalize <str>: 'power': power based normalization, or 'projection-back': projection back based normalization.
            threshold <float>: threshold for condition number when computing (WU)^{-1}.
        """
        super().__init__(n_bases=n_bases, partitioning=partitioning, normalize=normalize, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.nu = nu
        self.domain = domain
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
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def __repr__(self) -> str:
        s = "t-ILRMA("
        s += "n_bases={n_bases}"
        s += ", nu={nu}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalize={normalize}"
        s += ")"

        return s.format(**self.__dict__)
    
    def update_once(self):
        eps = self.eps

        self.update_source_model()
        self.update_space_model()

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        self.estimation = Y
        
        if self.normalize:
            if self.normalize == 'power':
                P = np.abs(Y)**2
                aux = np.sqrt(P.mean(axis=(1,2))) # (n_sources,)
                aux[aux < eps] = eps

                # Normalize
                W = W / aux[np.newaxis,:,np.newaxis]
                Y = Y / aux[:,np.newaxis,np.newaxis]

                if self.partitioning:
                    Z = self.latent
                    T = self.base
                    Zaux = Z / (aux[:,np.newaxis]**2) # (n_sources, n_bases)
                    Zauxsum = np.sum(Zaux, axis=0) # (n_bases,)
                    T = T * Zauxsum # (n_bins, n_bases)
                    Z = Zaux / Zauxsum # (n_sources, n_bases)
                    self.latent = Z
                    self.base = T
                else:
                    T = self.base
                    T = T / (aux[:,np.newaxis,np.newaxis]**2)
                    self.base = T
            else:
                raise ValueError("Not support normalization based on {}. Choose 'power' or 'projection-back'".format(self.normalize))
            
            self.demix_filter = W
            self.estimation = Y

    def update_source_model(self):
        nu = self.nu
        domain = self.domain
        eps = self.eps

        assert domain == 2, "Only domain = 2 is supported."

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        P = np.abs(Y)**2
        
        if self.partitioning:
            raise NotImplementedError("Only support when `partitioning=False` ")
            Z = self.latent # (n_sources, n_bases)
            T, V = self.base, self.activation

            TV = T[:,:,np.newaxis] * V[np.newaxis,:,:] # (n_bins, n_bases, n_frames)
            ZT = Z[:,np.newaxis,:] * T[np.newaxis,:,:] # (n_sources, n_bins, n_bases)
            ZTV = ZT @ V[np.newaxis,:,:] # (n_sources, n_bins, n_frames)
            ZTV[ZTV < eps] = eps
            division, ZTV_inverse = P / (ZTV**2), 1 / ZTV # (n_sources, n_bins, n_frames)
            numerator = np.sum(division[:,:,np.newaxis,:,] * TV, axis=(1,3)) # (n_sources, n_bases)
            denominator = np.sum(ZTV_inverse[:,:,np.newaxis,:,] * TV, axis=(1,3)) # (n_sources, n_bases)
            denominator[denominator < eps] = eps
            Z = np.sqrt(numerator / denominator) # (n_sources, n_bases)
            Zsum = Z.sum(axis=0)
            Z = Z / Zsum # (n_sources, n_bases)

            # Update bases
            ZT = Z[:,np.newaxis,:] * T[np.newaxis,:,:] # (n_sources, n_bins, n_bases)
            ZTV = ZT @ V[np.newaxis,:,:] # (n_sources, n_bins, n_frames)
            ZTV[ZTV < eps] = eps
            division, ZTV_inverse = P / (ZTV**2), 1 / ZTV # (n_sources, n_bins, n_frames)
            ZV = Z[:,:,np.newaxis] * V[np.newaxis,:,:] # (n_sources, n_bases, n_frames)
            numerator = np.sum(division[:,:,np.newaxis,:] * ZV[:,np.newaxis,:,:], axis=(0,3)) # (n_bins, n_bases)
            denominator = np.sum(ZTV_inverse[:,:,np.newaxis,:] * ZV[:,np.newaxis,:,:], axis=(0,3)) # (n_bins, n_bases)
            denominator[denominator < eps] = eps
            T = T * np.sqrt(numerator / denominator) # (n_bins, n_bases)

            # Update activations
            ZT = Z[:,np.newaxis,:] * T[np.newaxis,:,:] # (n_sources, n_bins, n_bases)
            ZTV = ZT @ V[np.newaxis,:,:] # (n_sources, n_bins, n_frames)
            ZTV[ZTV < eps] = eps
            division, ZTV_inverse = P / (ZTV**2), 1 / ZTV # (n_sources, n_bins, n_frames)
            ZT = Z[:,np.newaxis,:] * T[np.newaxis,:,:] # (n_sources, n_bins, n_bases)
            numerator = np.sum(division[:,:,np.newaxis,:] * ZT[:,:,:,np.newaxis], axis=(0,1)) # (n_bases, n_frames)
            denominator = np.sum(ZTV_inverse[:,:,np.newaxis,:] * ZT[:,:,:,np.newaxis], axis=(0,1)) # (n_bases, n_frames)
            denominator[denominator < eps] = eps
            V = V * np.sqrt(numerator / denominator) # (n_bins, n_bases)

            self.latent = Z
            self.base, self.activation = T, V
        else:
            T, V = self.base, self.activation # (n_sources, n_bins, n_bases), (n_sources, n_bases, n_frames)

            # Update bases
            V_transpose = V.transpose(0, 2, 1) # (n_sources, n_frames, n_bases)
            TV = T @ V # (n_sources, n_bins, n_frames)
            TV[TV < eps] = eps
            harmonic = 1 / (2 / ((2 + nu) * TV) + nu / ((2 + nu) * P))
            division, TV_inverse = harmonic / (TV**2), 1 / TV # (n_sources, n_bins, n_frames), (n_sources, n_bins, n_frames)
            TVV = TV_inverse @ V_transpose # (n_sources, n_bins, n_bases)
            TVV[TVV < eps] = eps
            T = T * np.sqrt(division @ V_transpose / TVV) # (n_sources, n_bins, n_bases)

            # Update activations
            T_transpose = T.transpose(0, 2, 1) # (n_sources, n_bases, n_bins)
            TV = T @ V # (n_sources, n_bins, n_frames)
            TV[TV < eps] = eps
            harmonic = 1 / (2 / ((2 + nu) * TV) + nu / ((2 + nu) * P))
            division, TV_inverse = harmonic / (TV**2), 1 / TV # (n_sources, n_bins, n_frames)
            TTV = T_transpose @ TV_inverse # (n_sources, n_bases, n_frames)
            TTV[TTV < eps] = eps
            V = V * np.sqrt(T_transpose @ division / TTV) # (n_sources, n_bases, n_frames)

            self.base, self.activation = T, V
    
    def update_space_model(self):
        n_sources = self.n_sources
        nu = self.nu
        eps = self.eps

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)
        
        if self.partitioning:
            Z = self.latent
            T, V = self.base, self.activation
            ZTV = np.sum(Z[:,np.newaxis,:,np.newaxis] * T[:,:,np.newaxis] * V[np.newaxis,:,:], axis=2) # (n_sources, n_bins, n_frames)
            R = ZTV[...,np.newaxis, np.newaxis] # (n_sources, n_bins, n_frames, 1, 1)
        else:
            T, V = self.base, self.activation
            TV = T @ V
            R = TV[...,np.newaxis, np.newaxis] # (n_sources, n_bins, n_frames, 1, 1)
        
        R[R < eps] = eps
        Xi = (nu * R + 2 * P[..., np.newaxis, np.newaxis]) / (nu + 2)

        X = X.transpose(1,2,0) # (n_bins, n_frames, n_channels)
        X = X[...,np.newaxis]
        X_Hermite = X.transpose(0,1,3,2).conj()
        XX = X @ X_Hermite # (n_bins, n_frames, n_channels, n_channels)
        U = XX / Xi
        U = U.mean(axis=2) # (n_sources, n_bins, n_channels, n_channels)

        for source_idx in range(n_sources):
            # W: (n_bins, n_sources, n_channels), U: (n_sources, n_bins, n_channels, n_channels)
            U_n = U[source_idx] # (n_bins, n_channels, n_channels)
            WU = W @ U_n # (n_bins, n_sources, n_channels)
            WU_inverse = np.linalg.inv(WU) # (n_bins, n_sources, n_channels)
            w = WU_inverse[...,source_idx] # (n_bins, n_channels)
            wUw = w[:,np.newaxis,:].conj() @ U_n @ w[:,:,np.newaxis]
            denominator = np.sqrt(wUw[...,0])
            denominator[denominator < eps] = eps
            W[:, source_idx, :] = w.conj() / denominator

        self.demix_filter = W

    def compute_negative_loglikelihood(self):
        n_frames = self.n_frames
        nu = self.nu
        eps = self.eps

        W = self.demix_filter
        Y = self.estimation

        P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)

        if self.partitioning:
            Z = self.latent
            T, V = self.base, self.activation
            R = np.sum(Z[:,np.newaxis,:,np.newaxis] * T[:,:,np.newaxis] * V[np.newaxis,:,:], axis=2) # (n_sources, n_bins, n_frames)
        else:
            T, V = self.base, self.activation
            R = T @ V # (n_sources, n_bins, n_frames)
        
        R[R < eps] = eps
        loss = np.sum((1 + nu / 2) * np.log(1 + (2 / nu) * (P / R)) + np.log(R)) - 2 * n_frames * np.sum(np.log(np.abs(np.linalg.det(W))))

        return loss

class KLILRMA(ILRMAbase):
    """
    Reference: "Independent Low-Rank Matrix Analysis Based on Generalized Kullback-Leibler Divergence"
    """
    def __init__(self, n_bases=10, partitioning=False, normalize='power', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        super().__init__(n_bases=n_bases, partitioning=partitioning, normalize=normalize, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.reference_id = reference_id

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
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def __repr__(self) -> str:
        s = "KL-ILRMA("
        s += "n_bases={n_bases}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalize={normalize}"
        s += ")"

        return s.format(**self.__dict__)

class RegularizedILRMA(ILRMAbase):
    """
    Reference: "Blind source separation based on independent low-rank matrix analysis with sparse regularization for time-series activity"
    See https://ieeexplore.ieee.org/document/7486081
    """
    def __init__(self, n_bases=10, partitioning=False, normalize='power', reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        """
        Args:
            normalize <str>
        """
        super().__init__(n_bases=n_bases, partitioning=partitioning, normalize=normalize, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.reference_id = reference_id

        raise NotImplementedError("Implement Regularized ILRMA")

class ConsistentGaussILRMA(GaussILRMA):
    """
    Reference: "Consistent independent low-rank matrix analysis for determined blind source separation"
    See https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-020-00704-4
    """
    def __init__(self, n_bases=10, partitioning=False, reference_id=0, fft_size=None, hop_size=None, callbacks=None, recordable_loss=True, eps=EPS, threshold=THRESHOLD):
        """
        Args:
            normalize <str>: 'power': power based normalization, or 'projection-back': projection back based normalization.
            threshold <float>: threshold for condition number when computing (WU)^{-1}.
        """
        super().__init__(n_bases=n_bases, partitioning=partitioning, normalize=False, reference_id=reference_id, threshold=threshold, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        if fft_size is None:
            raise ValueError("Specify `fft_size`.")
        
        if hop_size is None:
            hop_size = fft_size // 2
        
        self.fft_size, self.hop_size = fft_size, hop_size

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
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def __repr__(self) -> str:
        s = "Consistent-GaussILRMA("
        s += "n_bases={n_bases}"
        s += ", domain={domain}"
        s += ", partitioning={partitioning}"
        s += ", normalize={normalize}"
        s += ")"

        return s.format(**self.__dict__)
    
    def update_once(self):
        y = istft(self.estimation, fft_size=self.fft_size, hop_size=self.hop_size)
        self.estimation = stft(y, fft_size=self.fft_size, hop_size=self.hop_size)

        self.update_source_model()
        self.update_space_model()

        X, W = self.input, self.demix_filter
        T = self.base
        Y = self.separate(X, demix_filter=W)

        if self.partitioning:
            raise NotImplementedError("Not support 'projection-back' based normalization for partitioninig function. Choose 'power' based normalization.")
        scale = projection_back(Y, reference=X[self.reference_id])
        transposed_scale = scale.transpose(1,0) # (n_sources, n_bins) -> (n_bins, n_sources)
        W = W * transposed_scale[...,np.newaxis] # (n_bins, n_sources, n_channels)
        Y = self.separate(X, demix_filter=W)
        T = T * np.abs(scale[...,np.newaxis])**2

        self.demix_filter = W
        self.estimation = Y
        self.base = T

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

def _test(method, n_bases=10, domain=2, partitioning=False):
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

    if method == 'Gauss':
        ilrma = GaussILRMA(n_bases=n_bases, domain=domain, partitioning=partitioning)
    elif method == 't':
        nu = 1000
        ilrma = tILRMA(n_bases=n_bases, nu=nu, partitioning=partitioning)
    else:
        raise ValueError("Not support {}-ILRMA.".format(method))
    
    print(ilrma)
    estimation = ilrma(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_channels):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/ILRMA/{}ILRMA/partitioning{}/mixture-{}_estimated-iter{}-{}.wav".format(method, int(partitioning), sr, iteration, idx), signal=_estimated_signal, sr=sr)
    
    plt.figure()
    plt.plot(ilrma.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/ILRMA/{}ILRMA/partitioning{}/loss.png'.format(method, int(partitioning)), bbox_inches='tight')
    plt.close()

def _test_consistent_ilrma(n_bases=10, partitioning=False):
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
    
    ilrma = ConsistentGaussILRMA(n_bases=n_bases, partitioning=partitioning, fft_size=fft_size, hop_size=hop_size)
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
    os.makedirs("data/ILRMA/GaussILRMA/partitioning0", exist_ok=True)
    os.makedirs("data/ILRMA/GaussILRMA/partitioning1", exist_ok=True)
    os.makedirs("data/ILRMA/tILRMA/partitioning0", exist_ok=True)
    os.makedirs("data/ILRMA/tILRMA/partitioning1", exist_ok=True)
    os.makedirs("data/ILRMA/ConsistentGaussILRMA/partitioning0", exist_ok=True)
    os.makedirs("data/ILRMA/ConsistentGaussILRMA/partitioning1", exist_ok=True)

    """
    Use multichannel room impulse response database.
    Download database from "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/"
    """

    _test_conv()

    print("="*10, "Gauss-ILRMA", "="*10)
    print("-"*10, "without partitioning function", "-"*10)
    _test(method='Gauss', n_bases=2, partitioning=False)
    print()

    print("-"*10, "with partitioning function", "-"*10)
    _test(method='Gauss', n_bases=5, partitioning=True)
    print()

    print("="*10, "t-ILRMA", "="*10)
    print("-"*10, "without partitioning function", "-"*10)
    _test(method='t', n_bases=2, partitioning=False)
    print()
    _test(method='t', n_bases=5, partitioning=True)

    print("="*10, "Consistent-ILRMA", "="*10)
    _test_consistent_ilrma(n_bases=5, partitioning=False)
    print()