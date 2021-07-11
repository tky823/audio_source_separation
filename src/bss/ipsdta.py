import numpy as np

from algorithm.projection_back import projection_back

EPS = 1e-12

__kwargs_ikeshita_ipsdta__ = {
    'n_blocks': 1024
}

class IPSDTAbase:
    """
    Independent Positive Semi-Definite Tensor Analysis
    """
    def __init__(self, n_basis=10, callbacks=None, reference_id=0, recordable_loss=True, eps=EPS):
        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None
        self.reference_id = reference_id
        self.eps = eps
        
        self.n_basis = n_basis

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

        if not hasattr(self, 'basis'):
            U = 0.5 * np.random.rand(n_sources, n_basis, n_bins, n_bins) + 0.5j * np.random.rand(n_sources, n_basis, n_bins, n_bins) # should be positive semi-definite
            U = U.swapaxes(-2, -1).conj() @ U
            self.basis = _to_Hermite(U, axis1=3, axis2=4)
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

        if self.recordable_loss and len(self.loss) == 0:
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
        self.estimation = output

        return output
    
    def __repr__(self):
        s = "IPSDTA("
        s += "n_basis={n_basis}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        raise NotImplementedError("Implement 'update_once' method.")
    
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
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement `compute_negative_loglikelihood` method.")

class GaussIPSDTA(IPSDTAbase):
    """
    """
    def __init__(self, n_basis=10, callbacks=None, reference_id=0, author='Ikeshita', recordable_loss=True, eps=EPS, **kwargs):
        """
        Args:
        """
        super().__init__(n_basis=n_basis, callbacks=callbacks, reference_id=reference_id, recordable_loss=recordable_loss, eps=eps)

        self.author = author
        
        if author.lower() == 'ikeshita':
            if set(kwargs) - set(__kwargs_ikeshita_ipsdta__) != set():
                raise ValueError("Invalid keywords.")
            for key in __kwargs_ikeshita_ipsdta__.keys():
                setattr(self, key, __kwargs_ikeshita_ipsdta__[key])
            for key in kwargs.keys():
                setattr(self, key, kwargs[key])
        else:
            raise ValueError("Not support {}'s IPSDTA".format(author))
    
    def __call__(self, input, iteration=100, **kwargs):
        """
        Args:
            input (n_channels, n_bins, n_frames)
        Returns:
            output (n_channels, n_bins, n_frames)
        """
        self.input = input

        self._reset(**kwargs)

        if self.recordable_loss and len(self.loss) == 0:
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
            Y = self.separate(X, demix_filter=W)

        reference_id = self.reference_id
        
        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[..., np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def _reset(self, **kwargs):
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        
        if self.author.lower() == 'ikeshita':
            self._reset_ikeshita(**kwargs)
    
    def _reset_ikeshita(self, **kwargs):
        n_basis = self.n_basis
        n_blocks = self.n_blocks

        X = self.input

        n_channels, n_bins, n_frames = X.shape
        n_sources = n_channels # n_channels == n_sources

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        if not hasattr(self, 'demix_filter'):
            W_Hermite = np.eye(n_sources, n_channels, dtype=np.complex128)
            self.demix_filter = np.tile(W_Hermite, reps=(n_bins, 1, 1))
        else:
            W_Hermite = self.demix_filter.copy()
            self.demix_filter = W_Hermite
        
        self.estimation = self.separate(X, demix_filter=W_Hermite)

        n_neighbors = n_bins  // n_blocks
        n_paddings = (n_blocks - n_bins % n_blocks) % n_neighbors

        self.n_blocks, self.n_neighbors = n_blocks, n_neighbors
        self.n_paddings = n_paddings

        if not hasattr(self, 'basis'):
            if n_paddings > 0:
                U = 0.5 * np.random.rand(n_sources, n_basis, n_blocks + 1, n_neighbors, n_neighbors) + 0.5j * np.random.rand(n_sources, n_basis, n_blocks + 1, n_neighbors, n_neighbors)
            else:
                U = 0.5 * np.random.rand(n_sources, n_basis, n_blocks, n_neighbors, n_neighbors) + 0.5j * np.random.rand(n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            
            U = U.swapaxes(-2, -1).conj() @ U
            self.basis = _to_Hermite(U, axis1=3, axis2=4)
        else:
            self.basis = self.basis.copy()
        if not hasattr(self, 'activation'):
            self.activation = 0.9 * np.random.rand(n_sources, n_basis, n_frames) + 0.1
        else:
            self.activation = self.activation.copy()
        if not hasattr(self, 'fixed_point'):
            self.fixed_point = 0.5 * np.random.rand(n_sources, n_bins) + 0.5j * np.random.rand(n_sources, n_bins)
        else:
            self.fixed_point = self.fixed_point.copy()

    def __repr__(self):
        s = "Gauss-IPSDTA("
        s += "n_basis={n_basis}"
        if self.author.lower() == 'ikeshita':
            s += ", n_blocks={n_blocks}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        self.update_source_model()
        self.update_spatial_model()
    
    def update_source_model(self):
        if self.author.lower() == 'ikeshita':
            self.update_source_model_em()
        else:
            raise NotImplementedError("Not support {}'s IPSDTA.".format(self.author))
    
    def update_spatial_model(self):
        if self.author.lower() == 'ikeshita':
            self.update_spatial_model_fixed_point()
        else:
            raise NotImplementedError("Not support {}'s IPSDTA.".format(self.author))
    
    def update_source_model_em(self):
        self.update_basis_em()
        self.update_activation_em()
        # TODO: normalize
    
    def update_basis_em(self):
        n_bins, n_frames = self.n_bins, self.n_frames
        n_sources = self.n_sources
        n_basis = self.n_basis
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_paddings = self.n_paddings
        eps = self.eps

        X, W_Hermite = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W_Hermite) # (n_sources, n_bins, n_frames)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)

        U, V = self.basis, self.activation # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors), (n_sources, n_basis, n_frames)
        
        if n_paddings > 0:
            Y_low, Y_high = np.split(Y, [n_bins - (n_neighbors - n_paddings)], axis=2) # (n_sources, n_frames, n_bins - (n_neighbors - n_paddings)), (n_sources, n_frames, n_paddings)
            Y_low = Y_low.reshape(n_sources, n_frames, n_blocks, n_neighbors, 1) # (n_sources, n_frames, n_blocks, n_neighbors, 1)
            Y_high = Y_high.reshape(n_sources, n_frames, 1, n_neighbors - n_paddings, 1) # (n_sources, n_frames, 1, n_neighbors - n_paddings, 1)
            
            R_basis = U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = np.sum(R_basis, axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)

            R_basis_low, R_basis_high = np.split(R_basis, [n_blocks], axis=3) # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors), (n_sources, n_basis, n_frames, 1, n_neighbors, n_neighbors)
            R_low, R_high = np.split(R, [n_blocks], axis=2) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors), (n_sources, n_frames, 1, n_neighbors, n_neighbors)
            R_basis_high = R_basis_high[..., :-n_paddings, :-n_paddings] # (n_sources, n_basis, n_frames, 1, n_paddings, n_paddings)
            R_high = R_high[..., :-n_paddings, :-n_paddings] # (n_sources, n_frames, 1, n_neighbors - n_paddings, n_neighbors - n_paddings)
            R_low, R_high = _to_Hermite(R_low, axis1=3, axis2=4), _to_Hermite(R_high, axis1=3, axis2=4)

            inv_R_low, inv_R_high = np.linalg.inv(R_low + eps * np.eye(n_neighbors)), np.linalg.inv(R_high + eps * np.eye(n_neighbors - n_paddings)) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors), (n_sources, n_frames, 1, n_neighbors - n_paddings, n_neighbors - n_paddings)
            RR_low = R_basis_low @ inv_R_low[:, np.newaxis, :, :, :, :]
            RR_high = R_basis_high @ inv_R_high[:, np.newaxis, :, :, :, :]
            y_hat_low = RR_low @ Y_low[:, np.newaxis, :, :, :, :] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, 1)
            y_hat_high = RR_high @ Y_high[:, np.newaxis, :, :, :, :] # (n_sources, n_basis, n_frames, 1, n_neighbors - n_paddings, 1)

            R_hat_low = R_basis_low @ (np.eye(n_neighbors) - RR_low.swapaxes(-2, -1).conj()) # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            R_hat_high = R_basis_high @ (np.eye(n_neighbors - n_paddings) - RR_high.swapaxes(-2, -1).conj()) # (n_sources, n_basis, n_frames, 1, n_neighbors - n_paddings, n_neighbors - n_paddings)
            R_hat_low, R_hat_high = _to_Hermite(R_hat_low), _to_Hermite(R_hat_high)

            Phi_low = y_hat_low * y_hat_low.swapaxes(-2, -1).conj() + R_hat_low
            Phi_high = y_hat_high * y_hat_high.swapaxes(-2, -1).conj() + R_hat_high
            Phi_low, Phi_high = _to_Hermite(Phi_low), _to_Hermite(Phi_high) # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors), (n_sources, n_basis, n_frames, 1, n_neighbors - n_paddings, n_neighbors - n_paddings)
            
            V[V < eps] = eps
            U_low = np.mean(Phi_low[:, :, :, :, :, :] / V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=2) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            U_high = np.mean(Phi_high[:, :, :, :, :, :] / V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=2) # (n_sources, n_basis, 1, n_neighbors - n_paddings, n_neighbors - n_paddings)
            U_low, U_high = _to_Hermite(U_low, axis1=3, axis2=4), _to_Hermite(U_high, axis1=3, axis2=4)
            
            U = np.tile(np.eye(n_neighbors, dtype=np.complex128), reps=(n_sources, n_basis, n_blocks + 1, 1, 1))
            U[:, :, :n_blocks, :, :] = U_low
            U[:, :, n_blocks:, :-n_paddings, :-n_paddings] = U_high
        else:
            Y = Y.reshape(n_sources, n_frames, n_blocks, n_neighbors, 1) # (n_sources, n_frames, n_blocks, n_neighbors, 1)
            
            R_basis = U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = np.sum(R_basis, axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)

            inv_R = np.linalg.inv(R) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            RR = R_basis @ inv_R[:, np.newaxis, :, :, :, :]
            y_hat = RR @ Y[:, np.newaxis, :, :, :, :] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, 1)
            
            R_hat = R_basis @ (np.eye(n_neighbors) - RR.swapaxes(-2, -1).conj()) # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            R_hat = _to_Hermite(R_hat)

            Phi = y_hat * y_hat.swapaxes(-2, -1).conj() + R_hat
            Phi = _to_Hermite(Phi) # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            
            V[V < eps] = eps # here
            U = np.mean(Phi[:, :, :, :, :, :] / V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=2) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            U = _to_Hermite(U, axis1=3, axis2=4)
        
        self.basis, self.activation = U, V
    
    def update_activation_em(self):
        n_bins, n_frames = self.n_bins, self.n_frames
        n_sources = self.n_sources
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_paddings = self.n_paddings
        eps = self.eps

        X, W_Hermite = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W_Hermite) # (n_sources, n_bins, n_frames)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)

        U, V = self.basis, self.activation # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors), (n_sources, n_basis, n_frames)
        
        if n_paddings > 0:
            Y_low, Y_high = np.split(Y, [n_bins - (n_neighbors - n_paddings)], axis=2) # (n_sources, n_frames, n_bins - (n_neighbors - n_paddings)), (n_sources, n_frames, n_paddings)
            Y_low = Y_low.reshape(n_sources, n_frames, n_blocks, n_neighbors, 1) # (n_sources, n_frames, n_blocks, n_neighbors, 1)
            Y_high = Y_high.reshape(n_sources, n_frames, 1, n_neighbors - n_paddings, 1) # (n_sources, n_frames, 1, n_neighbors - n_paddings, 1)

            R_basis = U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = np.sum(R_basis, axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)

            R_basis_low, R_basis_high = np.split(R_basis, [n_blocks], axis=3) # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors), (n_sources, n_basis, n_frames, 1, n_neighbors, n_neighbors)
            R_low, R_high = np.split(R, [n_blocks], axis=2) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors), (n_sources, n_frames, 1, n_neighbors, n_neighbors)
            R_basis_high = R_basis_high[..., :-n_paddings, :-n_paddings] # (n_sources, n_basis, n_frames, 1, n_neighbors - n_paddings, n_neighbors - n_paddings)
            R_high = R_high[..., :-n_paddings, :-n_paddings] # (n_sources, n_frames, 1, n_neighbors - n_paddings, n_neighbors - n_paddings)

            inv_R_low, inv_R_high = np.linalg.inv(R_low + eps * np.eye(n_neighbors)), np.linalg.inv(R_high + eps * np.eye(n_neighbors - n_paddings)) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors), (n_sources, n_frames, 1, n_neighbors - n_paddings, n_neighbors - n_paddings)
            RR_low = R_basis_low @ inv_R_low[:, np.newaxis, :, :, :, :]
            RR_high = R_basis_high @ inv_R_high[:, np.newaxis, :, :, :, :]
            y_hat_low = RR_low @ Y_low[:, np.newaxis, :, :, :, :] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, 1)
            y_hat_high = RR_high @ Y_high[:, np.newaxis, :, :, :, :] # (n_sources, n_basis, n_frames, 1, n_neighbors - n_paddings, 1)

            R_hat_low = R_basis_low @ (np.eye(n_neighbors) - RR_low.swapaxes(-2, -1).conj()) # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            R_hat_high = R_basis_high @ (np.eye(n_neighbors - n_paddings) - RR_high.swapaxes(-2, -1).conj()) # (n_sources, n_basis, n_frames, 1, n_neighbors - n_paddings, n_neighbors - n_paddings)
            R_hat_low, R_hat_high = _to_Hermite(R_hat_low), _to_Hermite(R_hat_high)

            Phi_low = y_hat_low * y_hat_low.swapaxes(-2, -1).conj() + R_hat_low
            Phi_high = y_hat_high * y_hat_high.swapaxes(-2, -1).conj() + R_hat_high
            Phi_low, Phi_high = _to_Hermite(Phi_low), _to_Hermite(Phi_high) # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors), (n_sources, n_basis, n_frames, 1, n_neighbors - n_paddings, n_neighbors - n_paddings)
            
            U_low, U_high = np.split(U, [n_blocks], axis=2) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors), (n_sources, n_basis, 1, n_neighbors, n_neighbors)
            U_high = U_high[..., :-n_paddings, :-n_paddings]
            inv_U_low, inv_U_high = np.linalg.inv(U_low + eps * np.eye(n_neighbors)), np.linalg.inv(U_high + eps * np.eye(n_neighbors - n_paddings)) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors), (n_sources, n_basis, 1, n_neighbors, n_neighbors)
            UPhi_low, UPhi_high = inv_U_low[:, :, np.newaxis, :, :, :] @ Phi_low, inv_U_high[:, :, np.newaxis, :, :, :] @ Phi_high
            trace_low = np.trace(UPhi_low, axis1=-2, axis2=-1).real
            trace_high = np.trace(UPhi_high, axis1=-2, axis2=-1).real
            trace = np.concatenate([trace_low, trace_high], axis=3)
        else:
            Y = Y.reshape(n_sources, n_frames, n_blocks, n_neighbors, 1) # (n_sources, n_frames, n_blocks, n_neighbors, 1)
            
            # Update activation
            R_basis = U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = np.sum(R_basis, axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)

            inv_R = np.linalg.inv(R) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            RR = R_basis @ inv_R[:, np.newaxis, :, :, :, :]

            y_hat = RR @ Y[:, np.newaxis, :, :, :, :] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, 1)
            
            R_hat = R_basis @ (np.eye(n_neighbors) - RR.swapaxes(-2, -1).conj()) # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            R_hat = _to_Hermite(R_hat)

            Phi = y_hat * y_hat.swapaxes(-2, -1).conj() + R_hat
            Phi = _to_Hermite(Phi) # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            
            inv_U = np.linalg.inv(U) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            trace = np.trace(inv_U[:, :, np.newaxis, :, :, :] @ Phi, axis1=-2, axis2=-1).real
        
        trace[trace < 0] = 0
        trace = np.sum(trace, axis=3) # (n_sources, n_basis, n_frames)
        V = trace / n_bins
        
        self.basis, self.activation = U, V
    
    def update_spatial_model_fixed_point(self):
        n_bins, n_frames = self.n_bins, self.n_frames
        n_sources, n_channels = self.n_sources, self.n_channels
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_paddings = self.n_paddings
        eps = self.eps

        X, W_Hermite = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W_Hermite) # (n_sources, n_bins, n_frames)
        X = X.transpose(0, 2, 1) # (n_channels, n_frames, n_bins)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)
        A = np.linalg.inv(W_Hermite) # (n_bins, n_channels, n_sources)

        U, V = self.basis, self.activation # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors), (n_sources, n_basis, n_frames)
        R = np.sum(U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
        R = _to_Hermite(R, axis1=3, axis2=4)

        Lambda = self.fixed_point # (n_sources, n_bins)

        if n_paddings > 0:
            X_low, X_high = np.split(X, [n_bins - (n_neighbors - n_paddings)], axis=2) # (n_channels, n_frames, n_bins - (n_neighbors - n_paddings)), (n_channels, n_frames, n_paddings)
            X_low, X_high = X_low.reshape(n_channels, n_frames, n_blocks, n_neighbors), X_high.reshape(n_channels, n_frames, 1, n_neighbors - n_paddings)
            X_low, X_high = X_low.transpose(1, 2, 3, 0).reshape(n_frames, n_blocks, n_neighbors * n_channels), X_high.transpose(1, 2, 3, 0).reshape(n_frames, 1, (n_neighbors - n_paddings) * n_channels)

            # Compute G
            XX_low = X_low[:, :, :, np.newaxis] * X_low[:, :, np.newaxis, :].conj() # (n_frames, n_blocks, n_neighbors * n_channels, n_neighbors * n_channels)
            XX_high = X_high[:, :, :, np.newaxis] * X_high[:, :, np.newaxis, :].conj() # (n_frames, 1, (n_neighbors - n_paddings) * n_channels, (n_neighbors - n_paddings) * n_channels)
            XX_low = XX_low.reshape(n_frames, n_blocks, n_neighbors, n_channels, n_neighbors, n_channels)
            XX_high = XX_high.reshape(n_frames, 1, n_neighbors - n_paddings, n_channels, n_neighbors - n_paddings, n_channels)
            XX_low, XX_high = XX_low.transpose(0, 1, 2, 4, 3, 5), XX_high.transpose(0, 1, 2, 4, 3, 5) # (n_frames, n_blocks, n_neighbors, n_neighbors, n_channels, n_channels), (n_frames, 1, n_neighbors - n_paddings, n_neighbors - n_paddings, n_channels, n_channels)

            R_low, R_high = np.split(R, [n_blocks], axis=2) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors), (n_sources, n_frames, 1, n_neighbors, n_neighbors)
            R_high = R_high[..., :-n_paddings, :-n_paddings] # (n_sources, n_frames, 1, n_neighbors - n_paddings, n_neighbors - n_paddings)
            inv_R_low, inv_R_high = np.linalg.inv(R_low.conj() + eps * np.eye(n_neighbors)), np.linalg.inv(R_high.conj() + eps * np.eye(n_neighbors - n_paddings)) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors), (n_sources, n_frames, 1, n_neighbors - n_paddings, n_neighbors - n_paddings)
            
            G_low = np.mean(XX_low * inv_R_low[:, :, :, :, :, np.newaxis, np.newaxis], axis=1) # (n_sources, n_blocks, n_neighbors, n_neighbors, n_channels, n_channels)
            G_high = np.mean(XX_high * inv_R_high[:, :, :, :, :, np.newaxis, np.newaxis], axis=1) # (n_sources, 1, n_neighbors - n_paddings, n_neighbors - n_paddings, n_channels, n_channels)
            G_low, G_high = G_low.transpose(0, 1, 2, 4, 3, 5), G_high.transpose(0, 1, 2, 4, 3, 5) # (n_sources, n_blocks, n_neighbors, n_channels, n_neighbors, n_channels), (n_sources, 1, n_neighbors - n_paddings, n_channels, n_neighbors - n_paddings, n_channels)
            G_low, G_high = G_low.reshape(n_sources, n_blocks, n_neighbors * n_channels, n_neighbors * n_channels), G_high.reshape(n_sources, 1, (n_neighbors - n_paddings) * n_channels, (n_neighbors - n_paddings) * n_channels)
            G_low, G_high = _to_Hermite(G_low), _to_Hermite(G_high)
            inv_G_low, inv_G_high = np.linalg.inv(G_low + eps * np.eye(n_neighbors * n_channels)), np.linalg.inv(G_high + eps * np.eye((n_neighbors - n_paddings) * n_channels)) # (n_sources, n_blocks, n_neighbors * n_channels, n_neighbors * n_channels), (n_sources, 1, (n_neighbors - n_paddings) * n_channels, (n_neighbors - n_paddings) * n_channels)

            # Compute B
            A_low, A_high = np.split(A, [n_bins - (n_neighbors - n_paddings)], axis=0) # (n_bins - (n_neighbors - n_paddings), n_channels, n_sources), (n_neighbors - n_paddings, n_channels, n_sources)
            A_low, A_high = A_low.transpose(2, 0, 1), A_high.transpose(2, 0, 1) # (n_sources, n_bins - (n_neighbors - n_paddings), n_channels), (n_sources, n_neighbors - n_paddings, n_channels)
            A_low, A_high = A_low.reshape(n_sources, n_blocks, n_neighbors, n_channels), A_high.reshape(n_sources, 1, n_neighbors - n_paddings, n_channels)

            inv_G_low = inv_G_low.reshape(n_sources, n_blocks, n_neighbors, n_channels, n_neighbors, n_channels) # (n_sources, n_blocks, n_neighbors * n_channels, n_neighbors * n_channels)
            inv_G_high = inv_G_high.reshape(n_sources, 1, n_neighbors - n_paddings, n_channels, n_neighbors - n_paddings, n_channels) # (n_sources, n_blocks, n_neighbors * n_channels, n_neighbors * n_channels)
            inv_G_low, inv_G_high = inv_G_low.transpose(0, 1, 2, 4, 3, 5), inv_G_high.transpose(0, 1, 2, 4, 3, 5) # (n_sources, n_blocks, n_neighbors, n_neighbors, n_channels, n_channels), (n_sources, 1, n_neighbors - n_paddings, n_neighbors - n_paddings, n_channels, n_channels)

            B_low = A_low[:, :, :, np.newaxis, np.newaxis, :].conj() @ inv_G_low @ A_low[:, :, np.newaxis, :, :, np.newaxis] # (n_sources, n_blocks, n_neighbors, n_neighbors, 1, 1)
            B_high = A_high[:, :, :, np.newaxis, np.newaxis, :].conj() @ inv_G_high @ A_high[:, :, np.newaxis, :, :, np.newaxis] # (n_sources, n_blocks, n_neighbors - n_paddings, n_neighbors - n_paddings, 1, 1)
            B_low, B_high = B_low.squeeze(axis=(-2, -1)), B_high.squeeze(axis=(-2, -1)) # (n_sources, n_blocks, n_neighbors, n_neighbors), (n_sources, n_blocks, n_neighbors - n_paddings, n_neighbors - n_paddings)
 
            # Update Lambda
            Lambda_low, Lambda_high = np.split(Lambda, [n_bins - (n_neighbors - n_paddings)], axis=1)
            Lambda_low = Lambda_low.reshape(n_sources, n_blocks, n_neighbors, 1)
            Lambda_high = Lambda_high.reshape(n_sources, 1, n_neighbors - n_paddings, 1)
            Lambda_low, Lambda_high = B_low.swapaxes(2, 3) @ Lambda_low.conj(), B_high.swapaxes(2, 3) @ Lambda_high.conj()
            Lambda_low[np.abs(Lambda_low) < eps], Lambda_high[np.abs(Lambda_high) < eps] = eps, eps
            Lambda_low, Lambda_high = 1 / Lambda_low, 1 / Lambda_high
            Lambda_low, Lambda_high = Lambda_low.squeeze(axis=3), Lambda_high.squeeze(axis=3) # (n_sources, n_blocks, n_neighbors), (n_sources, 1, n_neighbors - n_paddings)

            GL_low, GL_high = inv_G_low * Lambda_low[:, :, np.newaxis, :, np.newaxis, np.newaxis], inv_G_high * Lambda_high[:, :, np.newaxis, :, np.newaxis, np.newaxis]
            GL_low, GL_high = GL_low.transpose(0, 1, 2, 4, 3, 5), GL_high.transpose(0, 1, 2, 4, 3, 5) # (n_sources, n_blocks, n_neighbors, n_channels, n_neighbors, n_channels), (n_sources, 1, n_neighbors - n_paddings, n_channels, n_neighbors - n_paddings, n_channels)
            GL_low, GL_high = GL_low.reshape(n_sources, n_blocks, n_neighbors * n_channels, n_neighbors * n_channels), GL_high.reshape(n_sources, 1, (n_neighbors - n_paddings) * n_channels, (n_neighbors - n_paddings) * n_channels)

            Lambda_low, Lambda_high = Lambda_low.reshape(n_sources, n_bins - (n_neighbors - n_paddings)), Lambda_high.reshape(n_sources, n_neighbors - n_paddings)
            Lambda = np.concatenate([Lambda_low, Lambda_high], axis=1)

            A_low, A_high = A_low.reshape(n_sources, n_blocks, n_neighbors * n_channels), A_high.reshape(n_sources, 1, (n_neighbors - n_paddings) * n_channels)
            W_Hermite_low = np.sum(GL_low * A_low[:, :, np.newaxis, :], axis=3) # (n_sources, n_blocks, n_neighbors * n_channels)
            W_Hermite_high = np.sum(GL_high * A_high[:, :, np.newaxis, :], axis=3) # (n_sources, 1, (n_neighbors - n_paddings) * n_channels)
            W_Hermite_low, W_Hermite_high = W_Hermite_low.reshape(n_sources, n_bins - (n_neighbors - n_paddings), n_channels), W_Hermite_high.reshape(n_sources, n_neighbors - n_paddings, n_channels)
            W_Hermite = np.concatenate([W_Hermite_low, W_Hermite_high], axis=1) # (n_sources, n_bins, n_channels)
            W_Hermite = W_Hermite.transpose(1, 0, 2).conj()
        else:
            X = X.reshape(n_channels, n_frames, n_blocks, n_neighbors)
            X = X.transpose(1, 2, 3, 0).reshape(n_frames, n_blocks, n_neighbors * n_channels)

            # Compute G
            XX = X[:, :, :, np.newaxis] * X[:, :, np.newaxis, :].conj() # (n_frames, n_blocks, n_neighbors * n_channels, n_neighbors * n_channels)
            XX = XX.reshape(n_frames, n_blocks, n_neighbors, n_channels, n_neighbors, n_channels)
            XX = XX.transpose(0, 1, 2, 4, 3, 5) # (n_frames, n_blocks, n_neighbors, n_neighbors, n_channels, n_channels)

            inv_R = np.linalg.inv(R.conj() + eps * np.eye(n_neighbors)) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            
            G = np.mean(XX * inv_R[:, :, :, :, :, np.newaxis, np.newaxis], axis=1) # (n_sources, n_blocks, n_neighbors, n_neighbors, n_channels, n_channels)
            G = G.transpose(0, 1, 2, 4, 3, 5) # (n_sources, n_blocks, n_neighbors, n_channels, n_neighbors, n_channels)
            G = G.reshape(n_sources, n_blocks, n_neighbors * n_channels, n_neighbors * n_channels)
            G = _to_Hermite(G)
            inv_G = np.linalg.inv(G + eps * np.eye(n_neighbors * n_channels)) # (n_sources, n_blocks, n_neighbors * n_channels, n_neighbors * n_channels)

            # Compute B
            A = A.transpose(2, 0, 1) # (n_sources, n_bins, n_channels)
            A = A.reshape(n_sources, n_blocks, n_neighbors, n_channels)

            inv_G = inv_G.reshape(n_sources, n_blocks, n_neighbors, n_channels, n_neighbors, n_channels) # (n_sources, n_blocks, n_neighbors * n_channels, n_neighbors * n_channels)
            inv_G = inv_G.transpose(0, 1, 2, 4, 3, 5) # (n_sources, n_blocks, n_neighbors, n_neighbors, n_channels, n_channels)

            B = A[:, :, :, np.newaxis, np.newaxis, :].conj() @ inv_G @ A[:, :, np.newaxis, :, :, np.newaxis] # (n_sources, n_blocks, n_neighbors, n_neighbors, 1, 1)
            B = B.squeeze(axis=(-2, -1)) # (n_sources, n_blocks, n_neighbors, n_neighbors)
 
            # Update Lambda
            Lambda = Lambda.reshape(n_sources, n_blocks, n_neighbors, 1)
            Lambda = B.swapaxes(2, 3) @ Lambda.conj()
            Lambda[np.abs(Lambda) < eps] = eps
            Lambda = 1 / Lambda
            Lambda = Lambda.squeeze(axis=3) # (n_sources, n_blocks, n_neighbors)

            GL = inv_G * Lambda[:, :, np.newaxis, :, np.newaxis, np.newaxis]
            GL = GL.transpose(0, 1, 2, 4, 3, 5) # (n_sources, n_blocks, n_neighbors, n_channels, n_neighbors, n_channels)
            GL = GL.reshape(n_sources, n_blocks, n_neighbors * n_channels, n_neighbors * n_channels)

            Lambda = Lambda.reshape(n_sources, n_bins)

            A = A.reshape(n_sources, n_blocks, n_neighbors * n_channels)
            W_Hermite = np.sum(GL * A[:, :, np.newaxis, :], axis=3) # (n_sources, n_blocks, n_neighbors * n_channels)
            W_Hermite = W_Hermite.reshape(n_sources, n_bins, n_channels) # (n_sources, n_bins, n_channels)
            W_Hermite = W_Hermite.transpose(1, 0, 2).conj()

        self.demix_filter = W_Hermite
        self.fixed_point = Lambda

    def compute_negative_loglikelihood(self):
        loss = self.compute_negative_loglikelihood_ikeshita()

        return loss

    def compute_negative_loglikelihood_ikeshita(self):
        n_bins, n_frames = self.n_bins, self.n_frames
        n_sources = self.n_sources
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_paddings = self.n_paddings
        eps = self.eps

        X, W_Hermite = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W_Hermite) # (n_sources, n_bins, n_frames)
        
        U, V = self.basis, self.activation # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors), (n_sources, n_basis, n_frames)
        R = np.sum(U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)

        if n_paddings > 0:
            Y_low, Y_high = np.split(Y, [n_bins - (n_neighbors - n_paddings)], axis=2) # (n_sources, n_frames, n_bins - (n_neighbors - n_paddings)), (n_sources, n_frames, n_paddings)
            R_low, R_high = np.split(R, [n_blocks], axis=2) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors), (n_sources, n_frames, 1, n_neighbors, n_neighbors)
            R_high = R_high[..., :-n_paddings, :-n_paddings] # (n_sources, n_frames, 1, n_paddings, n_paddings)
            R_low, R_high = _to_Hermite(R_low, axis1=3, axis2=4), _to_Hermite(R_high, axis1=3, axis2=4)

            Y_low = Y_low.reshape(n_sources, n_frames, n_blocks, n_neighbors, 1) # (n_sources, n_frames, n_blocks, n_neighbors, 1)
            Y_high = Y_high.reshape(n_sources, n_frames, 1, n_neighbors - n_paddings, 1) # (n_sources, n_frames, 1, n_neighbors - n_paddings, 1)

            inv_R_low, inv_R_high = np.linalg.inv(R_low + eps * np.eye(n_neighbors)), np.linalg.inv(R_high + eps * np.eye(n_neighbors - n_paddings)) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors), # (n_sources, n_frames, 1, n_neighbors - n_paddings, n_neighbors - n_paddings)
            Ry_low = inv_R_low @ Y_low # (n_sources, n_frames, n_blocks - 1, n_neighbors, 1)
            Ry_high = inv_R_high @ Y_high # (n_sources, n_frames, 1, n_neighbors - n_paddings, 1)
            Ry_low = Ry_low.reshape(n_sources, n_frames, n_blocks * n_neighbors)
            Ry_high = Ry_high.reshape(n_sources, n_frames, n_neighbors - n_paddings)
            Ry = np.concatenate([Ry_low, Ry_high], axis=2) # (n_sources, n_frames, n_bins)

            det_low, det_high = np.linalg.det(R_low).real, np.linalg.det(R_high).real # (n_sources, n_frames, n_blocks), # (n_sources, n_frames, 1)
            det = np.concatenate([det_low, det_high], axis=2) # (n_sources, n_frames, n_blocks + 1)
        else:
            R = _to_Hermite(R, axis1=3, axis2=4)
            Y = Y.reshape(n_sources, n_frames, n_blocks, n_neighbors, 1) # (n_sources, n_frames, n_blocks, n_neighbors, 1)
            
            inv_R = np.linalg.inv(R + eps * np.eye(n_neighbors)) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            Ry = inv_R @ Y # (n_sources, n_frames, n_blocks, n_neighbors, 1)
            Ry = Ry.reshape(n_sources, n_frames, n_blocks * n_neighbors)

            det = np.linalg.det(R).real # (n_sources, n_frames, n_blocks)

            Y = Y.reshape(n_sources, n_frames, n_blocks * n_neighbors)

        det[det < eps] = eps
        logdet = np.sum(np.log(det), axis=2) # (n_sources, n_frames)

        Y_Hermite = Y.conj()
        yRy = np.sum(Y_Hermite * Ry, axis=2).real # (n_sources, n_frames)
        loss = np.sum(yRy + logdet) - 2 * n_frames * np.sum(np.log(np.abs(np.linalg.det(W_Hermite))))

        return loss

class tIPSDTA(IPSDTAbase):
    """
    """
    def __init__(self, n_basis=10, callbacks=None, reference_id=0, author='Kondo', recordable_loss=True, eps=EPS, **kwargs):
        """
        Args:
        """
        super().__init__(n_basis=n_basis, callbacks=callbacks, reference_id=reference_id, recordable_loss=recordable_loss, eps=eps)

        self.author = author

        raise NotImplementedError("In progress...")

def _to_Hermite(X, axis1=-2, axis2=-1):
    X = (X + X.swapaxes(axis1, axis2).conj()) / 2
    return X

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

def _test_gauss_ipsdta(n_basis=10):
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

    # IPSDTA
    n_channels = len(titles)
    iteration = 50

    ipsdta = GaussIPSDTA(n_basis=n_basis)
    print(ipsdta)
    estimation = ipsdta(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_channels):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/IPSDTA/GaussIPSDTA/partitioning0/mixture-{}_estimated-iter{}-{}.wav".format(sr, iteration, idx), signal=_estimated_signal, sr=sr)
    
    plt.figure()
    plt.plot(ipsdta.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/IPSDTA/GaussIPSDTA/partitioning0/loss.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.io import loadmat

    from utils.utils_audio import read_wav, write_wav
    from transform.stft import stft, istft

    plt.rcParams['figure.dpi'] = 200

    os.makedirs("data/multi-channel", exist_ok=True)
    os.makedirs("data/ILRMA/GaussIPSDTA/partitioning0", exist_ok=True)

    """
    Use multichannel room impulse response database.
    Download database from "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/"
    """

    _test_conv()

    print("="*10, "Gauss-IPSDTA", "="*10)
    _test_gauss_ipsdta(n_basis=2)
    print()