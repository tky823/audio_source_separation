import numpy as np

from utils.utils_linalg import to_PSD
from algorithm.projection_back import projection_back

EPS = 1e-12

__authors_ipsdta__ = [
    'ikeshita', 'kondo'
]

__kwargs_ikeshita_ipsdta__ = {
    'n_blocks': 1024,
    'spatial_iteration': 1
}

__kwargs_kondo_ipsdta__ = {
    'n_blocks': 1024,
    'spatial_iteration': 10
}

class IPSDTAbase:
    """
    Independent Positive Semi-Definite Tensor Analysis
    """
    def __init__(self, n_basis=10, normalize=True, callbacks=None, reference_id=0, recordable_loss=True, eps=EPS):
        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None
        self.reference_id = reference_id
        self.eps = eps
        
        self.n_basis = n_basis
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
            U = to_PSD(U, axis1=3, axis2=4)
            self.basis = U.transpose(0, 2, 3, 1) # (n_sources, n_bins, n_bins, n_basis)
        else:
            self.basis = self.basis.copy()
        if not hasattr(self, 'activation'):
            self.activation = np.random.rand(n_sources, n_basis, n_frames)
        else:
            self.activation = self.activation.copy()
        
        if self.normalize:
            U, V = self.basis, self.activation # (n_sources, n_bins, n_bins, n_basis), (n_sources, n_basis, n_frames)
            trace = np.trace(U, axis1=1, axis2=2).real # (n_sources, n_basis)
            U = U / trace[:, np.newaxis, np.newaxis, :]
            V = V * trace[:, :, np.newaxis]

            self.basis, self.activation = U, V
    
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
        s += ", normalize={normalize}"
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
        Gauss Independent Positive Semi-Definite Tensor Analysis
        References:
            "Independent Positive Semidefinite Tensor Analysisin Blind Source Separation"
            "Convergence-Guaranteed Independent Positive Semidefinite Tensor Analysis Based on Student's T Distribution"
        See:
            https://ieeexplore.ieee.org/document/8553546
            https://ieeexplore.ieee.org/document/9054150
    """
    def __init__(self, n_basis=10, spatial_iteration=None, normalize=True, callbacks=None, reference_id=0, author='Kondo', recordable_loss=True, eps=EPS, **kwargs):
        """
        Args:
            n_basis <int>: Number of basis matrices
            callbacks <callable> or <list<callable>>:
            reference_id <int>: Reference microphone index
            author <str>: 'Ikeshita' or 'Kondo'
        """
        super().__init__(n_basis=n_basis, normalize=normalize, callbacks=callbacks, reference_id=reference_id, recordable_loss=recordable_loss, eps=eps)

        self.spatial_iteration = spatial_iteration
        self.author = author

        if author.lower() in __authors_ipsdta__:
            if author.lower() == 'ikeshita':
                if set(kwargs) - set(__kwargs_ikeshita_ipsdta__) != set():
                    raise ValueError("Invalid keywords.")
                for key in __kwargs_ikeshita_ipsdta__.keys():
                    setattr(self, key, __kwargs_ikeshita_ipsdta__[key])
                self.algorithm_source = 'em'
                self.algorithm_spatial = 'fixed-point'
            elif author.lower() == 'kondo':
                if set(kwargs) - set(__kwargs_kondo_ipsdta__) != set():
                    raise ValueError("Invalid keywords.")
                for key in __kwargs_kondo_ipsdta__.keys():
                    setattr(self, key, __kwargs_kondo_ipsdta__[key])
                self.algorithm_source = 'mm'
                self.algorithm_spatial = 'vcd'
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

        if self.author.lower() in __authors_ipsdta__:
            self._reset_block_diagonal(**kwargs)
    
    def _reset_block_diagonal(self, **kwargs):
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
        n_remains = n_bins % n_blocks

        self.n_blocks, self.n_neighbors = n_blocks, n_neighbors
        self.n_remains = n_remains

        if not hasattr(self, 'basis'):
            if n_remains > 0:
                eye_low = np.eye(n_neighbors, dtype=np.complex128)
                eye_high = np.eye(n_neighbors + 1, dtype=np.complex128)
                eye_low = np.tile(eye_low, reps=(n_sources, n_basis, n_blocks - n_remains, 1, 1))
                eye_high = np.tile(eye_high, reps=(n_sources, n_basis, n_remains, 1, 1))
                U_low, U_high = np.random.rand(n_sources, n_basis, n_blocks - n_remains, n_neighbors), np.random.rand(n_sources, n_basis, n_remains, n_neighbors + 1)
                U_low, U_high = U_low[:, :, :, :, np.newaxis] * eye_low, U_high[:, :, :, :, np.newaxis] * eye_high
                U = U_low.transpose(0, 2, 3, 4, 1), U_high.transpose(0, 2, 3, 4, 1)
            else:
                eye = np.eye(n_neighbors, dtype=np.complex128)
                eye = np.tile(eye, reps=(n_sources, n_basis, n_blocks, 1, 1))
                U = np.random.rand(n_sources, n_basis, n_blocks, n_neighbors)
                U = U[:, :, :, :, np.newaxis] * eye
                U = U.transpose(0, 2, 3, 4, 1)
            self.basis = U
        else:
            if n_remains > 0:
                U_low, U_high = self.basis
                U_low, U_high = U_low.copy(), U_high.copy()
                U = U_low, U_high
            else:
                U = self.basis
                U = U.copy()
            self.basis = U
        
        if not hasattr(self, 'activation'):
            self.activation = np.random.rand(n_sources, n_basis, n_frames)
        else:
            self.activation = self.activation.copy()

        if self.normalize:
            self.normalize_psdtf()
        
        if self.algorithm_spatial == 'fixed-point':
            if not hasattr(self, 'fixed_point'):
                self.fixed_point = np.ones((n_sources, n_bins), dtype=np.complex128)
            else:
                self.fixed_point = self.fixed_point.copy()

    def __repr__(self):
        s = "Gauss-IPSDTA("
        s += "n_basis={n_basis}"
        s += ", normalize={normalize}"
        s += ", algorithm(source)={algorithm_source}"
        s += ", algorithm(spatial)={algorithm_spatial}"
        if self.author.lower() in __authors_ipsdta__:
            s += ", n_blocks={n_blocks}"
        s += ", author={author}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        self.update_source_model()

        for spatial_idx in range(self.spatial_iteration):
            self.update_spatial_model()
    
    def update_source_model(self):
        algorithm_source = self.algorithm_source

        if algorithm_source == 'em':
            self.update_source_model_em()
        elif algorithm_source == 'mm':
            self.update_source_model_mm()
        else:
            raise NotImplementedError("Not support {}'s IPSDTA.".format(self.author))
        
        if self.normalize:
            self.normalize_psdtf()
     
    def update_spatial_model(self):
        algorithm_spatial = self.algorithm_spatial

        if algorithm_spatial == 'fixed-point':
            self.update_spatial_model_fixed_point()
        elif algorithm_spatial == 'vcd':
            self.update_spatial_model_vcd()
        else:
            raise NotImplementedError("Not support {}-based spatial model updates.".format(algorithm_spatial))
    
    def update_source_model_em(self):
        self.update_basis_em()
        self.update_activation_em()
    
    def update_source_model_mm(self):
        self.update_basis_mm()
        self.update_activation_mm()

    def update_basis_em(self):
        n_frames = self.n_frames
        n_sources = self.n_sources
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_remains = self.n_remains
        eps = self.eps

        X, W_Hermite = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W_Hermite) # (n_sources, n_bins, n_frames)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)

        U, V = self.basis, self.activation # (n_sources, n_basis, n_frames)
        
        if n_remains > 0:
            U_low, U_high = U
            U_low, U_high = U_low.transpose(0, 4, 1, 2, 3), U_high.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_basis, n_remains, n_neighbors + n_remains, n_neighbors + n_remains)
            Y_low, Y_high = np.split(Y, [(n_blocks - n_remains) * n_neighbors], axis=2) # (n_sources, n_frames, (n_blocks - n_remains) * n_neighbors), (n_sources, n_frames, n_remains * (n_neighbors + 1))
            Y_low = Y_low.reshape(n_sources, n_frames, n_blocks - n_remains, n_neighbors, 1)
            Y_high = Y_high.reshape(n_sources, n_frames, n_remains, n_neighbors + 1, 1)
            
            R_basis_low = U_low[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            R_basis_high = U_high[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_low, R_high = np.sum(R_basis_low, axis=1), np.sum(R_basis_high, axis=1) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_low, R_high = to_PSD(R_low, axis1=3, axis2=4, eps=eps), to_PSD(R_high, axis1=3, axis2=4, eps=eps)

            inv_R_low, inv_R_high = np.linalg.inv(R_low), np.linalg.inv(R_high) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            RR_low = R_basis_low @ inv_R_low[:, np.newaxis, :, :, :, :]
            RR_high = R_basis_high @ inv_R_high[:, np.newaxis, :, :, :, :]
            y_hat_low = RR_low @ Y_low[:, np.newaxis, :, :, :, :] # (n_sources, n_basis, n_frames, n_blocks - n_remains, n_neighbors, 1)
            y_hat_high = RR_high @ Y_high[:, np.newaxis, :, :, :, :] # (n_sources, n_basis, n_frames, n_remains, n_neighbors + 1, 1)

            R_hat_low = R_basis_low @ (np.eye(n_neighbors) - RR_low.swapaxes(-2, -1).conj()) # (n_sources, n_basis, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            R_hat_high = R_basis_high @ (np.eye(n_neighbors + 1) - RR_high.swapaxes(-2, -1).conj()) # (n_sources, n_basis, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_hat_low, R_hat_high = to_PSD(R_hat_low, eps=eps), to_PSD(R_hat_high, eps=eps)

            Phi_low = y_hat_low * y_hat_low.swapaxes(-2, -1).conj() + R_hat_low
            Phi_high = y_hat_high * y_hat_high.swapaxes(-2, -1).conj() + R_hat_high
            Phi_low, Phi_high = to_PSD(Phi_low, eps=eps), to_PSD(Phi_high, eps=eps) # (n_sources, n_basis, n_frames, n_blocks - 1, n_neighbors, n_neighbors), (n_sources, n_basis, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            
            V[V < eps] = eps
            U_low = np.mean(Phi_low[:, :, :, :, :, :] / V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=2) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors)
            U_high = np.mean(Phi_high[:, :, :, :, :, :] / V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=2) # (n_sources, n_basis, n_remains, n_neighbors + 1, n_neighbors + 1)
            U_low, U_high = to_PSD(U_low, axis1=3, axis2=4, eps=eps), to_PSD(U_high, axis1=3, axis2=4, eps=eps)
            U = U_low.transpose(0, 2, 3, 4, 1), U_high.transpose(0, 2, 3, 4, 1)
        else:
            U = U.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            Y = Y.reshape(n_sources, n_frames, n_blocks, n_neighbors, 1)

            R_basis = U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = np.sum(R_basis, axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = to_PSD(R, axis1=3, axis2=4, eps=eps)

            inv_R = np.linalg.inv(R) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            RR = R_basis @ inv_R[:, np.newaxis, :, :, :, :]
            y_hat = RR @ Y[:, np.newaxis, :, :, :, :] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, 1)

            R_hat = R_basis @ (np.eye(n_neighbors) - RR.swapaxes(-2, -1).conj()) # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            R_hat = to_PSD(R_hat, eps=eps)

            Phi = y_hat * y_hat.swapaxes(-2, -1).conj() + R_hat
            Phi = to_PSD(Phi, eps=eps) # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)

            V[V < eps] = eps
            U = np.mean(Phi[:, :, :, :, :, :] / V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=2) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            U = to_PSD(U, axis1=3, axis2=4, eps=eps)
            U = U.transpose(0, 2, 3, 4, 1)
        
        self.basis, self.activation = U, V

    def update_activation_em(self):
        n_bins, n_frames = self.n_bins, self.n_frames
        n_sources = self.n_sources
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_remains = self.n_remains
        eps = self.eps

        X, W_Hermite = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W_Hermite) # (n_sources, n_bins, n_frames)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)

        U, V = self.basis, self.activation # _, (n_sources, n_basis, n_frames)
        
        if n_remains > 0:
            U_low, U_high = U
            U_low, U_high = U_low.transpose(0, 4, 1, 2, 3), U_high.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_basis, n_remains, n_neighbors + 1, n_neighbors + 1)
            Y_low, Y_high = np.split(Y, [(n_blocks - n_remains) * n_neighbors], axis=2) # (n_sources, n_frames, (n_blocks - n_remains) * n_neighbors), (n_sources, n_frames, n_remains * (n_neighbors + 1))
            Y_low = Y_low.reshape(n_sources, n_frames, n_blocks - n_remains, n_neighbors, 1)
            Y_high = Y_high.reshape(n_sources, n_frames, n_remains, n_neighbors + 1, 1)
            
            R_basis_low = U_low[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            R_basis_high = U_high[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_low, R_high = np.sum(R_basis_low, axis=1), np.sum(R_basis_high, axis=1) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_low, R_high = to_PSD(R_low, axis1=3, axis2=4, eps=eps), to_PSD(R_high, axis1=3, axis2=4, eps=eps)

            inv_R_low, inv_R_high = np.linalg.inv(R_low), np.linalg.inv(R_high) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_frames, 1, n_neighbors + 1, n_neighbors + 1)
            RR_low = R_basis_low @ inv_R_low[:, np.newaxis, :, :, :, :]
            RR_high = R_basis_high @ inv_R_high[:, np.newaxis, :, :, :, :]
            y_hat_low = RR_low @ Y_low[:, np.newaxis, :, :, :, :] # (n_sources, n_basis, n_frames, n_blocks - n_remains, n_neighbors, 1)
            y_hat_high = RR_high @ Y_high[:, np.newaxis, :, :, :, :] # (n_sources, n_basis, n_frames, n_remains, n_neighbors + 1, 1)

            R_hat_low = R_basis_low @ (np.eye(n_neighbors) - RR_low.swapaxes(-2, -1).conj()) # (n_sources, n_basis, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            R_hat_high = R_basis_high @ (np.eye(n_neighbors + 1) - RR_high.swapaxes(-2, -1).conj()) # (n_sources, n_basis, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_hat_low, R_hat_high = to_PSD(R_hat_low, eps=eps), to_PSD(R_hat_high, eps=eps)

            Phi_low = y_hat_low * y_hat_low.swapaxes(-2, -1).conj() + R_hat_low
            Phi_high = y_hat_high * y_hat_high.swapaxes(-2, -1).conj() + R_hat_high
            Phi_low, Phi_high = to_PSD(Phi_low, eps=eps), to_PSD(Phi_high, eps=eps) # (n_sources, n_basis, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_basis, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            
            inv_U_low, inv_U_high = np.linalg.inv(U_low), np.linalg.inv(U_high) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_basis, n_remains, n_neighbors + 1, n_neighbors + 1)
            UPhi_low, UPhi_high = inv_U_low[:, :, np.newaxis, :, :, :] @ Phi_low, inv_U_high[:, :, np.newaxis, :, :, :] @ Phi_high
            trace_low = np.trace(UPhi_low, axis1=-2, axis2=-1).real
            trace_high = np.trace(UPhi_high, axis1=-2, axis2=-1).real
            trace = np.concatenate([trace_low, trace_high], axis=3)

            U = U_low.transpose(0, 2, 3, 4, 1), U_high.transpose(0, 2, 3, 4, 1)
        else:
            U = U.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            Y = Y.reshape(n_sources, n_frames, n_blocks, n_neighbors, 1)

            R_basis = U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = np.sum(R_basis, axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = to_PSD(R, axis1=3, axis2=4, eps=eps)

            inv_R = np.linalg.inv(R) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            RR = R_basis @ inv_R[:, np.newaxis, :, :, :, :]
            y_hat = RR @ Y[:, np.newaxis, :, :, :, :] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, 1)
            
            R_hat = R_basis @ (np.eye(n_neighbors) - RR.swapaxes(-2, -1).conj()) # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            R_hat = to_PSD(R_hat, eps=eps)

            Phi = y_hat * y_hat.swapaxes(-2, -1).conj() + R_hat
            Phi = to_PSD(Phi, eps=eps) # (n_sources, n_basis, n_frames, n_blocks - 1, n_neighbors, n_neighbors), (n_sources, n_basis, n_frames, 1, n_neighbors + n_remains, n_neighbors + n_remains)
            
            inv_U = np.linalg.inv(U) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            UPhi = inv_U[:, :, np.newaxis, :, :, :] @ Phi
            trace = np.trace(UPhi, axis1=-2, axis2=-1).real

            U = U.transpose(0, 2, 3, 4, 1)
        
        trace[trace < 0] = 0
        trace = np.sum(trace, axis=3) # (n_sources, n_basis, n_frames)
        V = trace / n_bins
        
        self.basis, self.activation = U, V

    def update_basis_mm(self):
        n_sources = self.n_sources
        n_frames = self.n_frames
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_remains = self.n_remains
        eps = self.eps

        X, W_Hermite = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W_Hermite) # (n_sources, n_bins, n_frames)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)

        U, V = self.basis, self.activation # _, (n_sources, n_basis, n_frames)

        if n_remains > 0:
            U_low, U_high = U
            U_low, U_high = U_low.transpose(0, 4, 1, 2, 3), U_high.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_basis, n_remains, n_neighbors + 1, n_neighbors + 1)
            y_low, y_high = np.split(Y, [(n_blocks - n_remains)* n_neighbors], axis=2) # (n_sources, n_frames, (n_blocks - n_remains) * n_neighbors), (n_sources, n_frames, n_remains * (n_neighbors + 1))
            y_low, y_high = y_low.reshape(n_sources, n_frames, n_blocks - n_remains, n_neighbors, 1), y_high.reshape(n_sources, n_frames, n_remains, n_neighbors + 1, 1)

            R_basis_low = U_low[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            R_basis_high = U_high[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_low, R_high = np.sum(R_basis_low, axis=1), np.sum(R_basis_high, axis=1) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_low, R_high = to_PSD(R_low, axis1=3, axis2=4, eps=eps), to_PSD(R_high, axis1=3, axis2=4, eps=eps)
            inv_R_low, inv_R_high = np.linalg.inv(R_low), np.linalg.inv(R_high)
            inv_R_low, inv_R_high = to_PSD(inv_R_low, axis1=3, axis2=4, eps=eps), to_PSD(inv_R_high, axis1=3, axis2=4, eps=eps)
            
            yy_low = y_low @ y_low.transpose(0, 1, 2, 4, 3).conj() + eps * np.eye(n_neighbors)
            yy_high = y_high @ y_high.transpose(0, 1, 2, 4, 3).conj() + eps * np.eye(n_neighbors + 1)
            RyyR_low = inv_R_low @ yy_low @ inv_R_low # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            RyyR_high = inv_R_high @ yy_high @ inv_R_high # (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            S_low = np.sum(V[:, :, :, np.newaxis, np.newaxis, np.newaxis] * RyyR_low[:, np.newaxis, :, :, :, :], axis=2) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors)
            S_high = np.sum(V[:, :, :, np.newaxis, np.newaxis, np.newaxis] * RyyR_high[:, np.newaxis, :, :, :, :], axis=2) # (n_sources, n_basis, n_remains, n_neighbors + 1, n_neighbors + 1)
            T_low = np.sum(V[:, :, :, np.newaxis, np.newaxis, np.newaxis] * inv_R_low[:, np.newaxis, :, :, :, :], axis=2) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors)
            T_high = np.sum(V[:, :, :, np.newaxis, np.newaxis, np.newaxis] * inv_R_high[:, np.newaxis, :, :, :, :], axis=2) # (n_sources, n_basis, n_remains, n_neighbors + 1, n_neighbors + 1)

            # compute S_low^(1/2) and S_high^(1/2)
            eigval, eigvec = np.linalg.eigh(S_low)
            eigval[eigval < 0] = 0
            eigval = np.sqrt(eigval)
            eigval = eigval[..., np.newaxis] * np.eye(n_neighbors)
            sqrt_S_low = eigvec @ eigval @ np.linalg.inv(eigvec)
            sqrt_S_low = to_PSD(sqrt_S_low, eps=eps)

            eigval, eigvec = np.linalg.eigh(S_high)
            eigval[eigval < 0] = 0
            eigval = np.sqrt(eigval)
            eigval = eigval[..., np.newaxis] * np.eye(n_neighbors + 1)
            sqrt_S_high = eigvec @ eigval @ np.linalg.inv(eigvec)
            sqrt_S_high = to_PSD(sqrt_S_high, eps=eps)

            # compute (S^(1/2)TUTS^(1/2))^(-1/2)
            STUTS_low, STUTS_high = sqrt_S_low @ U_low @ T_low @ U_low @ sqrt_S_low, sqrt_S_high @ U_high @ T_high @ U_high @ sqrt_S_high
            STUTS_low, STUTS_high = to_PSD(STUTS_low, eps=eps), to_PSD(STUTS_high, eps=eps)

            eigval, eigvec = np.linalg.eigh(STUTS_low)
            eigval[eigval < 0] = 0
            eigval = np.sqrt(eigval)
            eigval = eigval[..., np.newaxis] * np.eye(n_neighbors)
            sqrt_STUTS_low = eigvec @ eigval @ np.linalg.inv(eigvec)
            sqrt_STUTS_low = to_PSD(sqrt_STUTS_low, eps=eps)
            inv_STUTS_low = np.linalg.inv(sqrt_STUTS_low)
            inv_STUTS_low = to_PSD(inv_STUTS_low, eps=eps)

            eigval, eigvec = np.linalg.eigh(STUTS_high)
            eigval[eigval < 0] = 0
            eigval = np.sqrt(eigval)
            eigval = eigval[..., np.newaxis] * np.eye(n_neighbors + 1)
            sqrt_STUTS_high = eigvec @ eigval @ np.linalg.inv(eigvec)
            sqrt_STUTS_high = to_PSD(sqrt_STUTS_high, eps=eps)
            inv_STUTS_high = np.linalg.inv(sqrt_STUTS_high)
            inv_STUTS_high = to_PSD(inv_STUTS_high, eps=eps)

            U_low, U_high = U_low @ sqrt_S_low @ inv_STUTS_low @ sqrt_S_low @ U_low, U_high @ sqrt_S_high @ inv_STUTS_high @ sqrt_S_high @ U_high
            U_low, U_high = to_PSD(U_low, eps=eps), to_PSD(U_high, eps=eps)
            U = U_low.transpose(0, 2, 3, 4, 1), U_high.transpose(0, 2, 3, 4, 1)
        else:
            U = U.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            y = Y.reshape(n_sources, n_frames, n_blocks, n_neighbors, 1)

            R_basis = U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = np.sum(R_basis, axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = to_PSD(R, axis1=3, axis2=4, eps=eps)
            inv_R = np.linalg.inv(R) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            inv_R = to_PSD(inv_R, axis1=3, axis2=4, eps=eps)

            yy = y @ y.transpose(0, 1, 2, 4, 3).conj() + eps * np.eye(n_neighbors)
            RyyR = inv_R @ yy @ inv_R # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            S = np.sum(V[:, :, :, np.newaxis, np.newaxis, np.newaxis] * RyyR[:, np.newaxis, :, :, :, :], axis=2) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            T = np.sum(V[:, :, :, np.newaxis, np.newaxis, np.newaxis] * inv_R[:, np.newaxis, :, :, :, :], axis=2) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)

            # compute S^(1/2)
            eigval, eigvec = np.linalg.eigh(S)
            eigval[eigval < 0] = 0
            eigval = np.sqrt(eigval)
            eigval = eigval[..., np.newaxis] * np.eye(n_neighbors)
            sqrt_S = eigvec @ eigval @ np.linalg.inv(eigvec)
            sqrt_S = to_PSD(sqrt_S, eps=eps)

            # compute (S^(1/2)TUTS^(1/2))^(-1/2)
            STUTS = sqrt_S @ U @ T @ U @ sqrt_S
            STUTS = to_PSD(STUTS, eps=eps)
            eigval, eigvec = np.linalg.eigh(STUTS)
            eigval[eigval < 0] = 0
            eigval = np.sqrt(eigval)
            eigval = eigval[..., np.newaxis] * np.eye(n_neighbors)
            sqrt_STUTS = eigvec @ eigval @ np.linalg.inv(eigvec)
            sqrt_STUTS = to_PSD(sqrt_STUTS, eps=eps)
            inv_STUTS = np.linalg.inv(sqrt_STUTS)
            inv_STUTS = to_PSD(inv_STUTS, eps=eps)
            U = U @ sqrt_S @ inv_STUTS @ sqrt_S @ U
            U = to_PSD(U, eps=eps)
            U = U.transpose(0, 2, 3, 4, 1)
        
        self.basis, self.activation = U, V

    def update_activation_mm(self):
        n_sources = self.n_sources
        n_frames = self.n_frames
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_remains = self.n_remains
        eps = self.eps

        X, W_Hermite = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W_Hermite) # (n_sources, n_bins, n_frames)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)

        U, V = self.basis, self.activation # _, (n_sources, n_basis, n_frames)

        if n_remains > 0:
            U_low, U_high = U
            U_low, U_high = U_low.transpose(0, 4, 1, 2, 3), U_high.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_basis, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_low = np.sum(U_low[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            R_high = np.sum(U_high[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, 1, n_neighbors + 1, n_neighbors + 1)
            R_low, R_high = to_PSD(R_low, eps=eps), to_PSD(R_high, eps=eps)
            y_low, y_high = np.split(Y, [(n_blocks - n_remains)* n_neighbors], axis=2) # (n_sources, n_frames, (n_blocks - n_remains) * n_neighbors), (n_sources, n_frames, n_remains * (n_neighbors + 1))
            y_low, y_high = y_low.reshape(n_sources, n_frames, n_blocks - n_remains, n_neighbors), y_high.reshape(n_sources, n_frames, n_remains, n_neighbors + 1)

            yy_low = y_low[:, :, :, :, np.newaxis] * y_low[:, :, :, np.newaxis, :].conj() + eps * np.eye(n_neighbors) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            yy_high = y_high[:, :, :, :, np.newaxis] * y_high[:, :, :, np.newaxis, :].conj() + eps * np.eye(n_neighbors + 1) # (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            yy_low, yy_high = to_PSD(yy_low, eps=eps), to_PSD(yy_high, eps=eps)

            inv_R_low, inv_R_high = np.linalg.inv(R_low), np.linalg.inv(R_high) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            inv_R_low, inv_R_high = to_PSD(inv_R_low, eps=eps), to_PSD(inv_R_high, eps=eps)
            Ryy_low, Ryy_high = inv_R_low @ yy_low, inv_R_high @ yy_high # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            RU_low = inv_R_low[:, np.newaxis, :, :, :, :] @ U_low[:, :, np.newaxis, :, :, :] # (n_sources, n_basis, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            RU_high = inv_R_high[:, np.newaxis, :, :, :, :] @ U_high[:, :, np.newaxis, :, :, :] # (n_sources, n_basis, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)

            numerator_low = np.trace(RU_low @ Ryy_low[:, np.newaxis, :, :, :, :], axis1=-2, axis2=-1).real # (n_sources, n_basis, n_frames, n_blocks - n_remains)
            numerator_high = np.trace(RU_high @ Ryy_high[:, np.newaxis, :, :, :, :], axis1=-2, axis2=-1).real # (n_sources, n_basis, n_frames, n_remains)
            numerator = np.concatenate([numerator_low, numerator_high], axis=3) # (n_sources, n_basis, n_frames, n_blocks)
            denominator_low = np.trace(RU_low, axis1=-2, axis2=-1).real # (n_sources, n_basis, n_frames, n_blocks - n_remains)
            denominator_high = np.trace(RU_high, axis1=-2, axis2=-1).real # (n_sources, n_basis, n_frames, n_remains)
            denominator = np.concatenate([denominator_low, denominator_high], axis=3) # (n_sources, n_basis, n_frames, n_blocks)

            U = U_low.transpose(0, 2, 3, 4, 1), U_high.transpose(0, 2, 3, 4, 1) # (n_sources, n_blocks - n_remains, n_neighbors, n_neighbors, n_basis), (n_sources, n_remains, n_neighbors + 1, n_neighbors + 1, n_basis)
        else:
            U = U.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            R = np.sum(U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = to_PSD(R, eps=eps) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            y = Y.reshape(n_sources, n_frames, n_blocks, n_neighbors) # (n_sources, n_frames, n_blocks, n_neighbors)

            yy = y[:, :, :, :, np.newaxis] * y[:, :, :, np.newaxis, :].conj() + eps * np.eye(n_neighbors) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            yy = to_PSD(yy, eps=eps)

            inv_R = np.linalg.inv(R) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            inv_R = to_PSD(inv_R, eps=eps)
            Ryy = inv_R @ yy # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            RU = inv_R[:, np.newaxis, :, :, :, :] @ U[:, :, np.newaxis, :, :, :] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            numerator = np.trace(RU @ Ryy[:, np.newaxis, :, :, :, :], axis1=-2, axis2=-1).real # (n_sources, n_basis, n_frames, n_blocks)
            denominator = np.trace(RU, axis1=-2, axis2=-1).real # (n_sources, n_basis, n_frames, n_blocks)

            U = U.transpose(0, 2, 3, 4, 1)
        
        numerator, denominator = np.sum(numerator, axis=3), np.sum(denominator, axis=3) # (n_sources, n_basis, n_frames), (n_sources, n_basis, n_frames)
        numerator[numerator < 0] = 0
        denominator[denominator < eps] = eps
        V = V * np.sqrt(numerator / denominator) # (n_sources, n_basis, n_frames)

        self.basis, self.activation = U, V

    def update_spatial_model_fixed_point(self):
        n_frames = self.n_frames
        n_sources, n_channels = self.n_sources, self.n_channels
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_remains = self.n_remains
        eps = self.eps

        X, W_Hermite = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W_Hermite) # (n_sources, n_bins, n_frames)
        X = X.transpose(0, 2, 1) # (n_channels, n_frames, n_bins)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)

        U, V = self.basis, self.activation # _, (n_sources, n_basis, n_frames)
        Lambda = self.fixed_point # (n_sources, n_bins)

        if n_remains > 0:
            U_low, U_high = U
            U_low, U_high = U_low.transpose(0, 4, 1, 2, 3), U_high.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_basis, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_low = np.sum(U_low[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            R_high = np.sum(U_high[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_low, R_high = to_PSD(R_low, axis1=3, axis2=4), to_PSD(R_high, axis1=3, axis2=4)

            X_low, X_high = np.split(X, [(n_blocks - n_remains) * n_neighbors], axis=2) # (n_channels, n_frames, (n_blocks - n_remains) * n_neighbors), (n_channels, n_frames, n_remains * (n_neighbors + 1))
            X_low, X_high = X_low.reshape(n_channels, n_frames, n_blocks - n_remains, n_neighbors), X_high.reshape(n_channels, n_frames, n_remains, n_neighbors + 1)
            X_low, X_high = X_low.transpose(1, 2, 3, 0).reshape(n_frames, n_blocks - n_remains, n_neighbors * n_channels), X_high.transpose(1, 2, 3, 0).reshape(n_frames, n_remains, (n_neighbors + 1) * n_channels)

            # Compute G
            XX_low = X_low[:, :, :, np.newaxis] * X_low[:, :, np.newaxis, :].conj() # (n_frames, n_blocks - n_remains, n_neighbors * n_channels, n_neighbors * n_channels)
            XX_high = X_high[:, :, :, np.newaxis] * X_high[:, :, np.newaxis, :].conj() # (n_frames, n_remains, (n_neighbors + 1) * n_channels, (n_neighbors + 1) * n_channels)
            XX_low = XX_low.reshape(n_frames, n_blocks - n_remains, n_neighbors, n_channels, n_neighbors, n_channels)
            XX_high = XX_high.reshape(n_frames, n_remains, n_neighbors + 1, n_channels, n_neighbors + 1, n_channels)
            XX_low, XX_high = XX_low.transpose(0, 1, 2, 4, 3, 5), XX_high.transpose(0, 1, 2, 4, 3, 5) # (n_frames, n_blocks - n_remains, n_neighbors, n_neighbors, n_channels, n_channels), (n_frames, n_remains, n_neighbors + 1, n_neighbors + 1, n_channels, n_channels)

            inv_R_low, inv_R_high = np.linalg.inv(R_low.conj() + eps * np.eye(n_neighbors)), np.linalg.inv(R_high.conj() + eps * np.eye(n_neighbors + 1)) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)

            G_low = np.mean(XX_low * inv_R_low[:, :, :, :, :, np.newaxis, np.newaxis], axis=1) # (n_sources, n_blocks - n_remains, n_neighbors, n_neighbors, n_channels, n_channels)
            G_high = np.mean(XX_high * inv_R_high[:, :, :, :, :, np.newaxis, np.newaxis], axis=1) # (n_sources, n_remains, n_neighbors + 1, n_neighbors + 1, n_channels, n_channels)
            G_low, G_high = G_low.transpose(0, 1, 2, 4, 3, 5), G_high.transpose(0, 1, 2, 4, 3, 5) # (n_sources, n_blocks - n_remains, n_neighbors, n_channels, n_neighbors, n_channels), (n_sources, n_remains, n_neighbors + 1, n_channels, n_neighbors + 1, n_channels)
            G_low, G_high = G_low.reshape(n_sources, n_blocks - n_remains, n_neighbors * n_channels, n_neighbors * n_channels), G_high.reshape(n_sources, n_remains, (n_neighbors + 1) * n_channels, (n_neighbors + 1) * n_channels)
            G_low, G_high = to_PSD(G_low), to_PSD(G_high)
            
            inv_G_low, inv_G_high = np.linalg.inv(G_low), np.linalg.inv(G_high) # (n_sources, n_blocks - n_remains, n_neighbors * n_channels, n_neighbors * n_channels), (n_sources, n_remains, (n_neighbors + 1) * n_channels, (n_neighbors + 1) * n_channels)
            inv_G_low_Hermite, inv_G_high_Hermite = inv_G_low.transpose(0, 1, 3, 2).conj(), inv_G_high.transpose(0, 1, 3, 2).conj() # (n_sources, n_blocks - n_remains, n_neighbors * n_channels, n_neighbors * n_channels), (n_sources, n_remains, (n_neighbors + 1) * n_channels, (n_neighbors + 1) * n_channels)
            inv_G_low_Hermite, inv_G_high_Hermite = inv_G_low_Hermite.reshape(n_sources, n_blocks - n_remains, n_neighbors, n_channels, n_neighbors, n_channels), inv_G_high_Hermite.reshape(n_sources, n_remains, n_neighbors + 1, n_channels, n_neighbors + 1, n_channels)
            inv_G_low_Hermite, inv_G_high_Hermite = inv_G_low_Hermite.transpose(0, 1, 2, 4, 3, 5), inv_G_high_Hermite.transpose(0, 1, 2, 4, 3, 5) # (n_sources, n_blocks - n_remains, n_neighbors, n_neighbors, n_channels, n_channels), (n_sources, n_remains, n_neighbors + 1, n_neighbors + 1, n_channels, n_channels)

            A = np.linalg.inv(W_Hermite) # (n_bins, n_channels, n_sources)
            A = A.transpose(2, 0, 1) # (n_sources, n_bins, n_channels)
            A_low, A_high = np.split(A, [(n_blocks - n_remains) * n_neighbors], axis=1) # (n_sources, (n_blocks - n_remains) * n_neighbors, n_channels), (n_sources, n_remains * (n_neighbors + 1), n_channels)
            A_low, A_high = A_low.reshape(n_sources, n_blocks - n_remains, n_neighbors, n_channels), A_high.reshape(n_sources, n_remains, n_neighbors + 1, n_channels)
            B_low = A_low[:, :, :, np.newaxis, np.newaxis, :].conj() @ inv_G_low_Hermite @ A_low[:, :, np.newaxis, :, :, np.newaxis] # (n_sources, n_blocks - n_remains, n_neighbors, n_neighbors, 1, 1)
            B_high = A_high[:, :, :, np.newaxis, np.newaxis, :].conj() @ inv_G_high_Hermite @ A_high[:, :, np.newaxis, :, :, np.newaxis] # (n_sources, n_remains, n_neighbors + 1, n_neighbors + 1, 1, 1)
            B_low, B_high = B_low.squeeze(axis=(4, 5)), B_high.squeeze(axis=(4, 5)) # (n_sources, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_remains, n_neighbors + 1, n_neighbors + 1)

            # Update Lambda
            Lambda_low, Lambda_high = np.split(Lambda, [(n_blocks - n_remains) * n_neighbors], axis=1)
            Lambda_low, Lambda_high = Lambda_low.reshape(n_sources, n_blocks - n_remains, n_neighbors, 1), Lambda_high.reshape(n_sources, n_remains, n_neighbors + 1, 1)

            denominator_low, denominator_high = B_low.swapaxes(2, 3) @ Lambda_low.conj(), B_high.swapaxes(2, 3) @ Lambda_high.conj()
            denominator_low, denominator_high = denominator_low.squeeze(axis=3), denominator_high.squeeze(axis=3) # (n_sources, n_blocks - n_remains, n_neighbors), (n_sources, n_remains, n_neighbors + 1)
            denominator_low[np.abs(denominator_low) < eps], denominator_high[np.abs(denominator_high) < eps] = eps, eps
            Lambda_low, Lambda_high = 1 / denominator_low, 1 / denominator_high # (n_sources, n_blocks - n_remains, n_neighbors), (n_sources, n_remains, n_neighbors + 1)

            inv_G_low, inv_G_high = inv_G_low.reshape(n_sources, n_blocks - n_remains, n_neighbors, n_channels, n_neighbors, n_channels), inv_G_high.reshape(n_sources, n_remains, n_neighbors + 1, n_channels, n_neighbors + 1, n_channels)
            GL_low, GL_high = inv_G_low * Lambda_low[:, :, np.newaxis, np.newaxis, :, np.newaxis], inv_G_high * Lambda_high[:, :, np.newaxis, np.newaxis, :, np.newaxis] # (n_sources, n_blocks - n_remains, n_neighbors, n_channels, n_neighbors, n_channels), (n_sources, n_remains, n_neighbors + 1, n_channels, n_neighbors + 1, n_channels)
            GL_low, GL_high = GL_low.reshape(n_sources, n_blocks - n_remains, n_neighbors * n_channels, n_neighbors * n_channels), GL_high.reshape(n_sources, n_remains, (n_neighbors + 1) * n_channels, (n_neighbors + 1) * n_channels)

            Lambda_low, Lambda_high = Lambda_low.reshape(n_sources, (n_blocks - n_remains) * n_neighbors), Lambda_high.reshape(n_sources, n_remains * (n_neighbors + 1))
            Lambda = np.concatenate([Lambda_low, Lambda_high], axis=1) # (n_sources, n_bins)
            
            A_low, A_high = A_low.reshape(n_sources, n_blocks - n_remains, n_neighbors * n_channels, 1), A_high.reshape(n_sources, n_remains, (n_neighbors + 1) * n_channels, 1)
            W_low, W_high = np.squeeze(GL_low @ A_low, axis=3), np.squeeze(GL_high @ A_high, axis=3) # (n_sources, n_blocks - n_remains, n_neighbors * n_channels), (n_sources, n_remains, (n_neighbors + 1) * n_channels)
            W_low, W_high = W_low.reshape(n_sources, (n_blocks - n_remains) * n_neighbors, n_channels), W_high.reshape(n_sources, n_remains * (n_neighbors + 1), n_channels)
            W = np.concatenate([W_low, W_high], axis=1) # (n_sources, n_bins, n_channels)
            W = W.transpose(1, 0, 2)
            W_Hermite = W.conj() # (n_bins, n_sources, n_channels)
        else:
            U = U.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            R = np.sum(U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = to_PSD(R, axis1=3, axis2=4)

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
            G = to_PSD(G)
            
            inv_G = np.linalg.inv(G) # (n_sources, n_blocks, n_neighbors * n_channels, n_neighbors * n_channels)
            inv_G_Hermite = inv_G.transpose(0, 1, 3, 2).conj() # (n_sources, n_blocks, n_neighbors * n_channels, n_neighbors * n_channels)
            inv_G_Hermite = inv_G_Hermite.reshape(n_sources, n_blocks, n_neighbors, n_channels, n_neighbors, n_channels)
            inv_G_Hermite = inv_G_Hermite.transpose(0, 1, 2, 4, 3, 5) # (n_sources, n_blocks, n_neighbors, n_neighbors, n_channels, n_channels)

            A = np.linalg.inv(W_Hermite) # (n_bins, n_channels, n_sources)
            A = A.transpose(2, 0, 1) # (n_sources, n_bins, n_channels)
            A = A.reshape(n_sources, n_blocks, n_neighbors, n_channels)
            B = A[:, :, :, np.newaxis, np.newaxis, :].conj() @ inv_G_Hermite @ A[:, :, np.newaxis, :, :, np.newaxis] # (n_sources, n_blocks, n_neighbors, n_neighbors, 1, 1)
            B = B.squeeze(axis=(4, 5)) # (n_sources, n_blocks, n_neighbors, n_neighbors)

            # Update Lambda
            Lambda = Lambda.reshape(n_sources, n_blocks, n_neighbors, 1)

            denominator = B.swapaxes(2, 3) @ Lambda.conj()
            denominator = denominator.squeeze(axis=3) # (n_sources, n_blocks, n_neighbors)
            denominator[np.abs(denominator) < eps] = eps
            Lambda = 1 / denominator # (n_sources, n_blocks, n_neighbors)

            inv_G = inv_G.reshape(n_sources, n_blocks, n_neighbors, n_channels, n_neighbors, n_channels)
            GL = inv_G * Lambda[:, :, np.newaxis, np.newaxis, :, np.newaxis] # (n_sources, n_blocks, n_neighbors, n_channels, n_neighbors, n_channels)
            GL = GL.reshape(n_sources, n_blocks, n_neighbors * n_channels, n_neighbors * n_channels)

            Lambda = Lambda.reshape(n_sources, n_blocks * n_neighbors)
                        
            A = A.reshape(n_sources, n_blocks, n_neighbors * n_channels, 1)
            W = np.squeeze(GL @ A, axis=3) # (n_sources, n_blocks, n_neighbors * n_channels)
            W = W.reshape(n_sources, n_blocks * n_neighbors, n_channels)
            W = W.transpose(1, 0, 2)
            W_Hermite = W.conj() # (n_bins, n_sources, n_channels)

        self.demix_filter = W_Hermite
        self.fixed_point = Lambda

    def update_spatial_model_vcd(self):
        n_bins, n_frames = self.n_bins, self.n_frames
        n_sources, n_channels = self.n_sources, self.n_channels
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_remains = self.n_remains
        eps = self.eps

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W) # (n_sources, n_bins, n_frames)
        X = X.transpose(1, 2, 0) # (n_bins, n_frames, n_channels)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)

        U, V = self.basis, self.activation # _, (n_sources, n_basis, n_frames)

        if n_remains > 0:
            W_low, W_high = np.split(W, [(n_blocks - n_remains) * n_neighbors], axis=0)
            W_low, W_high = W_low.reshape(n_blocks - n_remains, n_neighbors, n_sources, n_channels), W_high.reshape(n_remains, n_neighbors + 1, n_sources, n_channels)
            X_low, X_high = np.split(X, [(n_blocks - n_remains) * n_neighbors], axis=0)
            X_low, X_high = X_low.reshape(n_blocks - n_remains, n_neighbors, n_frames, n_channels), X_high.reshape(n_remains, n_neighbors + 1, n_frames, n_channels)
            U_low, U_high = U
            U_low, U_high = U_low.transpose(0, 4, 1, 2, 3), U_high.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors), (n_sources, n_basis, n_remains, n_neighbors + 1)

            XX_low = X_low[:, :, :, :, np.newaxis] * X_low[:, :, :, np.newaxis, :].conj() # (n_blocks - n_remains, n_neighbors, n_frames, n_channels, n_channels)
            XX_high = X_high[:, :, :, :, np.newaxis] * X_high[:, :, :, np.newaxis, :].conj() # (n_remains, n_neighbors + 1, n_frames, n_channels, n_channels)
            XX_low, XX_high = to_PSD(XX_low, eps=eps), to_PSD(XX_high, eps=eps)

            mask_low, mask_high = 1 - np.eye(n_neighbors), 1 - np.eye(n_neighbors + 1) # (n_neighbors, n_neighbors), (n_neighbors + 1, n_neighbors + 1)
            E_low = np.tile(np.eye(n_sources, n_channels), reps=(n_blocks - n_remains, 1, 1)) # (n_neighbors, n_sources, n_channels)
            E_high = np.tile(np.eye(n_sources, n_channels), reps=(n_remains, 1, 1)) # (n_neighbors + 1, n_sources, n_channels)

            for source_idx in range(n_sources):
                e_n_low, e_n_high = E_low[:, source_idx, :], E_high[:, source_idx, :] # (n_blocks - n_remains, n_channels), (n_remains, n_channels)

                U_n_low = U_low[source_idx, :, :, :, :] # (n_basis, n_blocks - n_remains, n_neighbors', n_neighbors)
                U_n_high = U_high[source_idx, :, :, :, :] # (n_basis, n_remains, n_neighbors' + 1, n_neighbors + 1)
                V_n = V[source_idx, :] # (n_basis, n_frames)
                R_n_low = np.sum(U_n_low[:, np.newaxis, :, :, :] * V_n[:, :, np.newaxis, np.newaxis, np.newaxis], axis=0) # (n_frames, n_blocks - n_remains, n_neighbors', n_neighbors)
                R_n_high = np.sum(U_n_high[:, np.newaxis, :, :, :] * V_n[:, :, np.newaxis, np.newaxis, np.newaxis], axis=0) # (n_frames, n_remains, n_neighbors' + 1, n_neighbors + 1)
                R_n_low, R_n_high = to_PSD(R_n_low, axis1=2, axis2=3, eps=eps), to_PSD(R_n_high, axis1=2, axis2=3, eps=eps)
                
                inv_R_n_low, inv_R_n_high = np.linalg.inv(R_n_low), np.linalg.inv(R_n_high) # (n_frames, n_blocks - n_remains, n_neighbors', n_neighbors), (n_frames, n_remains, n_neighbors' + 1, n_neighbors + 1)
                inv_R_n_low, inv_R_n_high = to_PSD(inv_R_n_low, eps=eps), to_PSD(inv_R_n_high, eps=eps)
                inv_R_n_low, inv_R_n_high = inv_R_n_low.transpose(1, 3, 2, 0), inv_R_n_high.transpose(1, 3, 2, 0) # (n_blocks - n_remains, n_neighbors, n_neighbors', n_frames), (n_remains, n_neighbors + 1, n_neighbors' + 1, n_frames)
                inv_R_ii_n_low, inv_R_ii_n_high = np.diagonal(inv_R_n_low, axis1=1, axis2=2).real, np.diagonal(inv_R_n_high, axis1=1, axis2=2).real # (n_blocks - n_remains, n_frames, n_neighbors), (n_remains, n_frames, n_neighbors + 1)
                inv_R_ii_n_low, inv_R_ii_n_high = inv_R_ii_n_low.transpose(0, 2, 1), inv_R_ii_n_high.transpose(0, 2, 1) # (n_blocks - n_remains, n_neighbors, n_frames), (n_remains, n_neighbors + 1, n_frames)
                
                for neighbor_idx in range(n_neighbors):
                    w_n_low = W_low[:, :, source_idx, :].conj() # (n_blocks - n_remains, n_neighbors', n_channels)
                    Xw_n_low = np.sum(X_low.conj() * w_n_low[:, :, np.newaxis, :], axis=3) # (n_blocks - n_remains, n_neighbors', n_frames)

                    Q_in_low = inv_R_ii_n_low[:, neighbor_idx, :, np.newaxis, np.newaxis] * XX_low[:, neighbor_idx, :, :, :] # (n_blocks - n_remains, n_frames, n_channels, n_channels)
                    Q_in_low = Q_in_low.mean(axis=1) # (n_blocks - n_remains, n_channels, n_channels)
                    Q_in_low = to_PSD(Q_in_low, eps=eps)

                    XXw_n_low = X_low[:, neighbor_idx, np.newaxis, :, :] * Xw_n_low[:, :, :, np.newaxis] # (n_blocks - n_remains, n_neighbors', n_frames, n_channels)
                    RXXw_n_low = np.mean(inv_R_n_low[:, neighbor_idx, :, :, np.newaxis] * XXw_n_low, axis=2) # (n_blocks - n_remains, n_neighbors', n_channels)
                    gamma_in_low = np.sum(mask_low[np.newaxis, neighbor_idx, :, np.newaxis] * RXXw_n_low, axis=1) # (n_blocks - n_remains, n_channels)

                    WQ_in_low = W_low[:, neighbor_idx, :, :] @ Q_in_low # (n_blocks - n_remains, n_sources, n_channels)
                    zeta_in_low = np.linalg.solve(WQ_in_low, e_n_low) # (n_blocks - n_remains, n_channels)
                    zeta_hat_in_low = np.linalg.solve(Q_in_low, gamma_in_low) # (n_blocks - n_remains, n_channels)
                    eta_in_low = np.squeeze(zeta_in_low[:, np.newaxis, :].conj() @ Q_in_low @ zeta_in_low[:, :, np.newaxis], axis=(1, 2)) # (n_blocks - n_remains,)
                    eta_hat_in_low = np.squeeze(zeta_in_low[:, np.newaxis, :].conj() @ Q_in_low @ zeta_hat_in_low[:, :, np.newaxis], axis=(1, 2)) # (n_blocks - n_remains,)
                    
                    eta_in_low[np.abs(eta_in_low) < eps] = eps
                    condition_if_low = np.abs(eta_hat_in_low) < eps
                    eta_hat_in_low[condition_if_low] = eps
                    weight_low = (eta_hat_in_low / (2 * eta_in_low)) * (1 - np.sqrt(1 + 4 * eta_in_low / (np.abs(eta_hat_in_low)**2)))
                    weight_if_low = 1 / np.sqrt(eta_in_low)
                    weight_low[condition_if_low] = weight_if_low[condition_if_low]
                    w_in_low = weight_low[:, np.newaxis] * zeta_in_low - zeta_hat_in_low
                    W_low[:, neighbor_idx, source_idx, :] = w_in_low.conj()

                for neighbor_idx in range(n_neighbors + 1):
                    w_n_high = W_high[:, :, source_idx, :].conj() # (n_remains, n_neighbors' + 1, n_channels)
                    Xw_n_high = np.sum(X_high.conj() * w_n_high[:, :, np.newaxis, :], axis=3) # (n_remains, n_neighbors' + 1, n_frames)

                    Q_in_high = inv_R_ii_n_high[:, neighbor_idx, :, np.newaxis, np.newaxis] * XX_high[:, neighbor_idx, :, :, :] # (n_remains, n_frames, n_channels, n_channels)
                    Q_in_high = Q_in_high.mean(axis=1) # (n_remains, n_channels, n_channels)
                    Q_in_high = to_PSD(Q_in_high, eps=eps)

                    XXw_n_high = X_high[:, neighbor_idx, np.newaxis, :, :] * Xw_n_high[:, :, :, np.newaxis] # (n_remains, n_neighbors' + 1, n_frames, n_channels)
                    RXXw_n_high = np.mean(inv_R_n_high[:, neighbor_idx, :, :, np.newaxis] * XXw_n_high, axis=2) # (n_remains, n_neighbors' + 1, n_channels)
                    gamma_in_high = np.sum(mask_high[np.newaxis, neighbor_idx, :, np.newaxis] * RXXw_n_high, axis=1) # (n_remains, n_channels)

                    WQ_in_high = W_high[:, neighbor_idx, :, :] @ Q_in_high # (n_blocks - n_remains, n_sources, n_channels), (n_remains, n_sources, n_channels)
                    zeta_in_high = np.linalg.solve(WQ_in_high, e_n_high) # (n_remains, n_channels)
                    zeta_hat_in_high = np.linalg.solve(Q_in_high, gamma_in_high) # (n_remains, n_channels)
                    eta_in_high = np.squeeze(zeta_in_high[:, np.newaxis, :].conj() @ Q_in_high @ zeta_in_high[:, :, np.newaxis], axis=(1, 2)) # (n_remains,)
                    eta_hat_in_high = np.squeeze(zeta_in_high[:, np.newaxis, :].conj() @ Q_in_high @ zeta_hat_in_high[:, :, np.newaxis], axis=(1, 2)) # (n_remains,)

                    eta_in_high[np.abs(eta_in_high) < eps] = eps
                    condition_if_high = np.abs(eta_hat_in_high) < eps
                    eta_hat_in_high[condition_if_high] = eps
                    weight_high = (eta_hat_in_high / (2 * eta_in_high)) * (1 - np.sqrt(1 + 4 * eta_in_high / (np.abs(eta_hat_in_high)**2)))
                    weight_if_high = 1 / np.sqrt(eta_in_high)
                    weight_high[condition_if_high] = weight_if_high[condition_if_high]
                    w_in_high = weight_high[:, np.newaxis] * zeta_in_high - zeta_hat_in_high
                    W_high[:, neighbor_idx, source_idx, :] = w_in_high.conj()
            
            W_low, W_high = W_low.reshape((n_blocks - n_remains) * n_neighbors, n_sources, n_channels), W_high.reshape(n_remains * (n_neighbors + 1), n_sources, n_channels)
            W = np.concatenate([W_low, W_high], axis=0) # (n_bins, n_sources, n_channels)
        else:
            W = W.reshape(n_blocks, n_neighbors, n_sources, n_channels)
            X = X.reshape(n_blocks, n_neighbors, n_frames, n_channels)
            U = U.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks, n_neighbors)

            XX = X[:, :, :, :, np.newaxis] * X[:, :, :, np.newaxis, :].conj() # (n_blocks, n_neighbors, n_frames, n_channels, n_channels)
            XX = to_PSD(XX, eps=eps)

            mask = 1 - np.eye(n_neighbors) # (n_neighbors, n_neighbors)
            E = np.tile(np.eye(n_sources, n_channels), reps=(n_blocks, 1, 1)) # (n_blocks, n_neighbors, n_sources, n_channels)

            for source_idx in range(n_sources):
                e_n = E[:, source_idx, :] # (n_blocks, n_channels)

                U_n, V_n = U[source_idx, :, :, :, :], V[source_idx, :] # (n_basis, n_blocks, n_neighbors', n_neighbors), (n_basis, n_frames)
                R_n = np.sum(U_n[:, np.newaxis, :, :, :] * V_n[:, :, np.newaxis, np.newaxis, np.newaxis], axis=0) # (n_frames, n_blocks, n_neighbors', n_neighbors)
                R_n = to_PSD(R_n, axis1=2, axis2=3, eps=eps)

                inv_R_n = np.linalg.inv(R_n) # (n_frames, n_blocks, n_neighbors', n_neighbors)
                inv_R_n = to_PSD(inv_R_n, eps=eps)
                inv_R_n = inv_R_n.transpose(1, 3, 2, 0) # (n_blocks, n_neighbors, n_neighbors', n_frames)
                inv_R_ii_n = np.diagonal(inv_R_n, axis1=1, axis2=2).real # (n_blocks, n_frames, n_neighbors)
                inv_R_ii_n = inv_R_ii_n.transpose(0, 2, 1) # (n_blocks, n_neighbors, n_frames)

                for neighbor_idx in range(n_neighbors):
                    w_n = W[:, :, source_idx, :].conj() # (n_blocks, n_neighbors', n_channels)
                    Xw_n = np.sum(X.conj() * w_n[:, :, np.newaxis, :], axis=3) # (n_blocks, n_neighbors', n_frames)

                    Q_in = inv_R_ii_n[:, neighbor_idx, :, np.newaxis, np.newaxis] * XX[:, neighbor_idx, :, :, :] # (n_blocks, n_frames, n_channels, n_channels)
                    Q_in = Q_in.mean(axis=1) # (n_blocks, n_channels, n_channels)
                    Q_in = to_PSD(Q_in, eps=eps)

                    XXw_n = X[:, neighbor_idx, np.newaxis, :, :] * Xw_n[:, :, :, np.newaxis] # (n_blocks, n_neighbors', n_frames, n_channels)
                    RXXw_n = np.mean(inv_R_n[:, neighbor_idx, :, :, np.newaxis] * XXw_n, axis=2) # (n_blocks, n_neighbors', n_channels)
                    gamma_in = np.sum(mask[np.newaxis, neighbor_idx, :, np.newaxis] * RXXw_n, axis=1) # (n_blocks, n_channels)

                    WQ_in = W[:, neighbor_idx, :, :] @ Q_in # (n_blocks, n_sources, n_channels)
                    zeta_in = np.linalg.solve(WQ_in, e_n) # (n_blocks, n_channels)
                    zeta_hat_in = np.linalg.solve(Q_in, gamma_in) # (n_blocks, n_channels)
                    eta_in = np.squeeze(zeta_in[:, np.newaxis, :].conj() @ Q_in @ zeta_in[:, :, np.newaxis], axis=(1, 2)) # (n_blocks,)
                    eta_hat_in = np.squeeze(zeta_in[:, np.newaxis, :].conj() @ Q_in @ zeta_hat_in[:, :, np.newaxis], axis=(1, 2)) # (n_blocks,)

                    eta_in[np.abs(eta_in) < eps] = eps
                    condition_if = np.abs(eta_hat_in) < eps
                    eta_hat_in[condition_if] = eps
                    weight = (eta_hat_in / (2 * eta_in)) * (1 - np.sqrt(1 + 4 * eta_in / (np.abs(eta_hat_in)**2)))
                    weight_if = 1 / np.sqrt(eta_in)
                    weight[condition_if] = weight_if[condition_if]
                    w_in = weight[:, np.newaxis] * zeta_in - zeta_hat_in
                    W[:, neighbor_idx, source_idx, :] = w_in.conj()
            
            W = W.reshape(n_bins, n_sources, n_channels)
        
        self.demix_filter = W

    def normalize_psdtf(self):
        if self.author.lower() in __authors_ipsdta__:
            self.normalize_psdtf_block_diagonal()
        else:
            raise NotImplementedError("Not support {}'s IPSDTA.".format(self.author))

    def normalize_psdtf_block_diagonal(self):
        n_remains = self.n_remains

        U, V = self.basis, self.activation # _, (n_sources, n_basis, n_frames)

        if n_remains > 0:
            U_low, U_high = U
            U_low, U_high = U_low.transpose(0, 4, 1, 2, 3), U_high.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks - 1, n_neighbors, n_neighbors), (n_sources, n_basis, 1, n_neighbors + n_remains, n_neighbors + n_remains)
            trace_low, trace_high = np.trace(U_low, axis1=3, axis2=4).real, np.trace(U_high, axis1=3, axis2=4).real
            trace = np.concatenate([trace_low, trace_high], axis=2) # (n_sources, n_basis, n_blocks)
            trace = trace.sum(axis=2) # (n_sources, n_basis)
            U_low, U_high = U_low / trace[:, :, np.newaxis, np.newaxis, np.newaxis], U_high / trace[:, :, np.newaxis, np.newaxis, np.newaxis]
            U = U_low.transpose(0, 2, 3, 4, 1), U_high.transpose(0, 2, 3, 4, 1)
            V = V * trace[:, :, np.newaxis]
        else:
            U = U.transpose(0, 4, 1, 2, 3)
            trace = np.trace(U, axis1=3, axis2=4).real
            trace = trace.sum(axis=2) # (n_sources, n_basis)
            U = U / trace[:, :, np.newaxis, np.newaxis, np.newaxis]
            U = U.transpose(0, 2, 3, 4, 1)
            V = V * trace[:, :, np.newaxis]

        self.basis, self.activation = U, V

    def compute_negative_loglikelihood(self):
        if self.author.lower() in __authors_ipsdta__:
            loss = self.compute_negative_loglikelihood_block_diagonal()
        else:
            raise ValueError("Not support {}'s IPSDTA.".format(self.author))

        return loss

    def compute_negative_loglikelihood_block_diagonal(self):
        n_frames = self.n_frames
        n_sources = self.n_sources
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_remains = self.n_remains
        eps = self.eps

        X, W_Hermite = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W_Hermite) # (n_sources, n_bins, n_frames)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)
        U, V = self.basis, self.activation

        if n_remains > 0:
            U_low, U_high = U
            U_low, U_high = U_low.transpose(0, 4, 1, 2, 3), U_high.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_basis, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_low = np.sum(U_low[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            R_high = np.sum(U_high[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, 1, n_neighbors + 1, n_neighbors + 1)
            y_low, y_high = np.split(Y, [(n_blocks - n_remains)* n_neighbors], axis=2) # (n_sources, n_frames, (n_blocks - n_remains) * n_neighbors), (n_sources, n_frames, n_remains * (n_neighbors + 1))
            
            R_low, R_high = to_PSD(R_low, axis1=3, axis2=4, eps=eps), to_PSD(R_high, axis1=3, axis2=4, eps=eps)
            y_low = y_low.reshape(n_sources, n_frames, n_blocks - n_remains, n_neighbors, 1) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, 1)
            y_high = y_high.reshape(n_sources, n_frames, n_remains, n_neighbors + 1, 1) # (n_sources, n_frames, 1, n_neighbors + 1, 1)

            inv_R_low, inv_R_high = np.linalg.inv(R_low), np.linalg.inv(R_high) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1s, n_neighbors + n_remains)
            inv_R_low, inv_R_high = to_PSD(inv_R_low, eps=eps), to_PSD(inv_R_high, eps=eps)
            Ry_low = inv_R_low @ y_low # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, 1)
            Ry_high = inv_R_high @ y_high # (n_sources, n_frames, n_remains, n_neighbors + 1, 1)
            Ry_low = Ry_low.reshape(n_sources, n_frames, (n_blocks - n_remains) * n_neighbors)
            Ry_high = Ry_high.reshape(n_sources, n_frames, n_remains * (n_neighbors + 1))
            Ry = np.concatenate([Ry_low, Ry_high], axis=2) # (n_sources, n_frames, n_bins)

            eigvals_low, eigvals_high = np.linalg.eigvalsh(R_low), np.linalg.eigvalsh(R_high) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1)
            eigvals_low[eigvals_low < eps], eigvals_high[eigvals_high < eps] = eps, eps
            logdet_low, logdet_high = np.log(eigvals_low), np.log(eigvals_high) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1)
            logdet_low, logdet_high = logdet_low.sum(axis=3), logdet_high.sum(axis=3) # (n_sources, n_frames, n_blocks - n_remains), (n_sources, n_frames, n_remains)
            logdet_R = np.concatenate([logdet_low, logdet_high], axis=2)# (n_sources, n_frames, n_blocks)
        else:
            U = U.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            R = np.sum(U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            y = Y # (n_sources, n_frames, n_blocks * n_neighbors)

            R = to_PSD(R, axis1=3, axis2=4, eps=eps)
            y = y.reshape(n_sources, n_frames, n_blocks, n_neighbors, 1) # (n_sources, n_frames, n_blocks, n_neighbors, 1)

            inv_R = np.linalg.inv(R) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            inv_R = to_PSD(inv_R, eps=eps)
            Ry = inv_R @ y # (n_sources, n_frames, n_blocks, n_neighbors, 1)
            Ry = Ry.reshape(n_sources, n_frames, n_blocks * n_neighbors)

            eigvals = np.linalg.eigvalsh(R) # (n_sources, n_frames, n_blocks, n_neighbors)
            eigvals[eigvals < eps] = eps
            logdet_R = np.log(eigvals) # (n_sources, n_frames, n_blocks, n_neighbors)
            logdet_R = logdet_R.sum(axis=3) # (n_sources, n_frames, n_blocks)
        
        logdet_R = logdet_R.sum(axis=2) # (n_sources, n_frames)
        
        eigvals = np.linalg.eigvals(W_Hermite) # (n_bins, n_sources)
        eigvals = np.abs(eigvals)
        eigvals[eigvals < eps] = eps
        logdet_W = np.log(eigvals)
        logdet_W = logdet_W.sum(axis=1) # (n_bins,)

        y_Hermite = Y.conj()
        yRy = np.sum(y_Hermite * Ry, axis=2).real # (n_sources, n_frames)
        loss = np.sum(yRy + logdet_R) - 2 * n_frames * logdet_W.sum()

        return loss

class tIPSDTA(IPSDTAbase):
    """
        Independent Positive Semi-Definite Tensor Analysis Based on Student's t distribution
        Reference: "Convergence-Guaranteed Independent Positive Semidefinite Tensor Analysis Based on Student's T Distribution"
        See https://ieeexplore.ieee.org/document/9054150
    """
    def __init__(self, n_basis=10, nu=1, spatial_iteration=None, normalize=True, callbacks=None, reference_id=0, author='Kondo', recordable_loss=True, eps=EPS, **kwargs):
        """
        Args:
            nu <float>: Degree of freedom
            author <str>: 'Kondo'
        """
        super().__init__(n_basis=n_basis, normalize=normalize, callbacks=callbacks, reference_id=reference_id, recordable_loss=recordable_loss, eps=eps)

        self.nu = nu
        self.spatial_iteration = spatial_iteration
        self.author = author

        if author.lower() == 'kondo':
            if set(kwargs) - set(__kwargs_kondo_ipsdta__) != set():
                raise ValueError("Invalid keywords.")
            for key in __kwargs_kondo_ipsdta__.keys():
                setattr(self, key, __kwargs_kondo_ipsdta__[key])
            self.algorithm_source = 'mm'
            self.algorithm_spatial = 'vcd'
        else:
            raise ValueError("Not support {}'s IPSDTA".format(author))
        
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

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
        
        if self.author.lower() in __authors_ipsdta__:
            self._reset_block_diagonal(**kwargs)
    
    def _reset_block_diagonal(self, **kwargs):
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
        n_remains = n_bins % n_blocks

        self.n_blocks, self.n_neighbors = n_blocks, n_neighbors
        self.n_remains = n_remains

        if not hasattr(self, 'basis'):
            if n_remains > 0:
                eye_low = np.eye(n_neighbors, dtype=np.complex128)
                eye_high = np.eye(n_neighbors + 1, dtype=np.complex128)
                eye_low = np.tile(eye_low, reps=(n_sources, n_basis, n_blocks - n_remains, 1, 1))
                eye_high = np.tile(eye_high, reps=(n_sources, n_basis, n_remains, 1, 1))
                U_low, U_high = np.random.rand(n_sources, n_basis, n_blocks - n_remains, n_neighbors), np.random.rand(n_sources, n_basis, n_remains, n_neighbors + 1)
                U_low, U_high = U_low[:, :, :, :, np.newaxis] * eye_low, U_high[:, :, :, :, np.newaxis] * eye_high
                U = U_low.transpose(0, 2, 3, 4, 1), U_high.transpose(0, 2, 3, 4, 1)
            else:
                eye = np.eye(n_neighbors, dtype=np.complex128)
                eye = np.tile(eye, reps=(n_sources, n_basis, n_blocks, 1, 1))
                U = np.random.rand(n_sources, n_basis, n_blocks, n_neighbors)
                U = U[:, :, :, :, np.newaxis] * eye
                U = U.transpose(0, 2, 3, 4, 1)
            self.basis = U
        else:
            if n_remains > 0:
                U_low, U_high = self.basis
                U_low, U_high = U_low.copy(), U_high.copy()
                U = U_low, U_high
            else:
                U = self.basis
                U = U.copy()
            self.basis = U
        
        if not hasattr(self, 'activation'):
            self.activation = np.random.rand(n_sources, n_basis, n_frames)
        else:
            self.activation = self.activation.copy()

        if self.normalize:
            self.normalize_psdtf()

    def __repr__(self):
        s = "t-IPSDTA("
        s += "n_basis={n_basis}"
        s += ", nu={nu}"
        s += ", normalize={normalize}"
        s += ", algorithm(source)={algorithm_source}"
        s += ", algorithm(spatial)={algorithm_spatial}"
        if self.author.lower() in __authors_ipsdta__:
            s += ", n_blocks={n_blocks}"
        s += ", author={author}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        self.update_source_model()

        for spatial_idx in range(self.spatial_iteration):
            self.update_spatial_model()
    
    def update_source_model(self):
        algorithm_source = self.algorithm_source

        if algorithm_source == 'mm':
            self.update_source_model_mm()
        else:
            raise NotImplementedError("Not support {}'s IPSDTA.".format(self.author))
        
        if self.normalize:
            self.normalize_psdtf()
    
    def update_spatial_model(self):
        algorithm_spatial = self.algorithm_spatial

        if algorithm_spatial == 'vcd':
            self.update_spatial_model_vcd()
        else:
            raise NotImplementedError("Not support {}'s IPSDTA.".format(self.author))

    def update_source_model_mm(self):
        self.update_basis_mm()
        self.update_activation_mm()

    def update_basis_mm(self):
        n_sources = self.n_sources
        n_bins, n_frames = self.n_bins, self.n_frames
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_remains = self.n_remains

        nu = self.nu
        eps = self.eps

        X, W_Hermite = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W_Hermite) # (n_sources, n_bins, n_frames)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)

        U, V = self.basis, self.activation # _, (n_sources, n_basis, n_frames)

        if n_remains > 0:
            U_low, U_high = U
            U_low, U_high = U_low.transpose(0, 4, 1, 2, 3), U_high.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_basis, n_remains, n_neighbors + 1, n_neighbors + 1)
            y_low, y_high = np.split(Y, [(n_blocks - n_remains)* n_neighbors], axis=2) # (n_sources, n_frames, (n_blocks - n_remains) * n_neighbors), (n_sources, n_frames, n_remains * (n_neighbors + 1))
            y_low, y_high = y_low.reshape(n_sources, n_frames, n_blocks - n_remains, n_neighbors, 1), y_high.reshape(n_sources, n_frames, n_remains, n_neighbors + 1, 1)

            R_basis_low = U_low[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            R_basis_high = U_high[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_low, R_high = np.sum(R_basis_low, axis=1), np.sum(R_basis_high, axis=1) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_low, R_high = to_PSD(R_low, axis1=3, axis2=4, eps=eps), to_PSD(R_high, axis1=3, axis2=4, eps=eps)
            inv_R_low, inv_R_high = np.linalg.inv(R_low), np.linalg.inv(R_high)
            inv_R_low, inv_R_high = to_PSD(inv_R_low, axis1=3, axis2=4, eps=eps), to_PSD(inv_R_high, axis1=3, axis2=4, eps=eps)

            Ry_low, Ry_high = inv_R_low @ y_low, inv_R_high @ y_high # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, 1), (n_sources, n_frames, n_remains, n_neighbors + 1, 1)
            yRy_low, yRy_high = np.sum(y_low.conj() * Ry_low, axis=(2, 3, 4)).real, np.sum(y_high.conj() * Ry_high, axis=(2, 3, 4)).real
            yRy = yRy_low + yRy_high # (n_sources, n_frames)
            pi = (nu + 2 * n_bins) / (nu + 2 * yRy) # (n_sources, n_frames)
            
            yy_low = y_low @ y_low.transpose(0, 1, 2, 4, 3).conj() + eps * np.eye(n_neighbors)
            yy_high = y_high @ y_high.transpose(0, 1, 2, 4, 3).conj() + eps * np.eye(n_neighbors + 1)
            RyyR_low = inv_R_low @ yy_low @ inv_R_low # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            RyyR_high = inv_R_high @ yy_high @ inv_R_high # (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            piRyyR_low = pi[:, :, np.newaxis, np.newaxis, np.newaxis] * RyyR_low
            piRyyR_high = pi[:, :, np.newaxis, np.newaxis, np.newaxis] * RyyR_high
            S_low = np.sum(V[:, :, :, np.newaxis, np.newaxis, np.newaxis] * piRyyR_low[:, np.newaxis, :, :, :, :], axis=2) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors)
            S_high = np.sum(V[:, :, :, np.newaxis, np.newaxis, np.newaxis] * piRyyR_high[:, np.newaxis, :, :, :, :], axis=2) # (n_sources, n_basis, n_remains, n_neighbors + 1, n_neighbors + 1)
            T_low = np.sum(V[:, :, :, np.newaxis, np.newaxis, np.newaxis] * inv_R_low[:, np.newaxis, :, :, :, :], axis=2) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors)
            T_high = np.sum(V[:, :, :, np.newaxis, np.newaxis, np.newaxis] * inv_R_high[:, np.newaxis, :, :, :, :], axis=2) # (n_sources, n_basis, n_remains, n_neighbors + 1, n_neighbors + 1)

            # compute S_low^(1/2) and S_high^(1/2)
            eigval, eigvec = np.linalg.eigh(S_low)
            eigval[eigval < 0] = 0
            eigval = np.sqrt(eigval)
            eigval = eigval[..., np.newaxis] * np.eye(n_neighbors)
            sqrt_S_low = eigvec @ eigval @ np.linalg.inv(eigvec)
            sqrt_S_low = to_PSD(sqrt_S_low, eps=eps)

            eigval, eigvec = np.linalg.eigh(S_high)
            eigval[eigval < 0] = 0
            eigval = np.sqrt(eigval)
            eigval = eigval[..., np.newaxis] * np.eye(n_neighbors + 1)
            sqrt_S_high = eigvec @ eigval @ np.linalg.inv(eigvec)
            sqrt_S_high = to_PSD(sqrt_S_high, eps=eps)

            # compute (S^(1/2)TUTS^(1/2))^(-1/2)
            STUTS_low, STUTS_high = sqrt_S_low @ U_low @ T_low @ U_low @ sqrt_S_low, sqrt_S_high @ U_high @ T_high @ U_high @ sqrt_S_high
            STUTS_low, STUTS_high = to_PSD(STUTS_low, eps=eps), to_PSD(STUTS_high, eps=eps)

            eigval, eigvec = np.linalg.eigh(STUTS_low)
            eigval[eigval < 0] = 0
            eigval = np.sqrt(eigval)
            eigval = eigval[..., np.newaxis] * np.eye(n_neighbors)
            sqrt_STUTS_low = eigvec @ eigval @ np.linalg.inv(eigvec)
            sqrt_STUTS_low = to_PSD(sqrt_STUTS_low, eps=eps)
            inv_STUTS_low = np.linalg.inv(sqrt_STUTS_low)
            inv_STUTS_low = to_PSD(inv_STUTS_low, eps=eps)

            eigval, eigvec = np.linalg.eigh(STUTS_high)
            eigval[eigval < 0] = 0
            eigval = np.sqrt(eigval)
            eigval = eigval[..., np.newaxis] * np.eye(n_neighbors + 1)
            sqrt_STUTS_high = eigvec @ eigval @ np.linalg.inv(eigvec)
            sqrt_STUTS_high = to_PSD(sqrt_STUTS_high, eps=eps)
            inv_STUTS_high = np.linalg.inv(sqrt_STUTS_high)
            inv_STUTS_high = to_PSD(inv_STUTS_high, eps=eps)

            U_low, U_high = U_low @ sqrt_S_low @ inv_STUTS_low @ sqrt_S_low @ U_low, U_high @ sqrt_S_high @ inv_STUTS_high @ sqrt_S_high @ U_high
            U_low, U_high = to_PSD(U_low, eps=eps), to_PSD(U_high, eps=eps)
            U = U_low.transpose(0, 2, 3, 4, 1), U_high.transpose(0, 2, 3, 4, 1)
        else:
            U = U.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            y = Y.reshape(n_sources, n_frames, n_blocks, n_neighbors, 1)

            R_basis = U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = np.sum(R_basis, axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = to_PSD(R, axis1=3, axis2=4, eps=eps)
            inv_R = np.linalg.inv(R) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            inv_R = to_PSD(inv_R, axis1=3, axis2=4, eps=eps)

            Ry = inv_R @ y # (n_sources, n_frames, n_blocks, n_neighbors, 1)
            yRy = np.sum(y.conj() * Ry, axis=(2, 3, 4)).real # (n_sources, n_frames)
            pi = (nu + 2 * n_bins) / (nu + 2 * yRy) # (n_sources, n_frames)

            yy = y @ y.transpose(0, 1, 2, 4, 3).conj() + eps * np.eye(n_neighbors)
            RyyR = inv_R @ yy @ inv_R # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            piRyyR = pi[:, :, np.newaxis, np.newaxis, np.newaxis] * RyyR
            S = np.sum(V[:, :, :, np.newaxis, np.newaxis, np.newaxis] * piRyyR[:, np.newaxis, :, :, :, :], axis=2) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            T = np.sum(V[:, :, :, np.newaxis, np.newaxis, np.newaxis] * inv_R[:, np.newaxis, :, :, :, :], axis=2) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)

            # compute S^(1/2)
            eigval, eigvec = np.linalg.eigh(S)
            eigval[eigval < 0] = 0
            eigval = np.sqrt(eigval)
            eigval = eigval[..., np.newaxis] * np.eye(n_neighbors)
            sqrt_S = eigvec @ eigval @ np.linalg.inv(eigvec)
            sqrt_S = to_PSD(sqrt_S, eps=eps)

            # compute (S^(1/2)TUTS^(1/2))^(-1/2)
            STUTS = sqrt_S @ U @ T @ U @ sqrt_S
            STUTS = to_PSD(STUTS, eps=eps)
            eigval, eigvec = np.linalg.eigh(STUTS)
            eigval[eigval < 0] = 0
            eigval = np.sqrt(eigval)
            eigval = eigval[..., np.newaxis] * np.eye(n_neighbors)
            sqrt_STUTS = eigvec @ eigval @ np.linalg.inv(eigvec)
            sqrt_STUTS = to_PSD(sqrt_STUTS, eps=eps)
            inv_STUTS = np.linalg.inv(sqrt_STUTS)
            inv_STUTS = to_PSD(inv_STUTS, eps=eps)
            U = U @ sqrt_S @ inv_STUTS @ sqrt_S @ U
            U = to_PSD(U, eps=eps)
            U = U.transpose(0, 2, 3, 4, 1)
        
        self.basis, self.activation = U, V

    def update_activation_mm(self):
        n_sources = self.n_sources
        n_bins, n_frames = self.n_bins, self.n_frames
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_remains = self.n_remains

        nu = self.nu
        eps = self.eps

        X, W_Hermite = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W_Hermite) # (n_sources, n_bins, n_frames)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)

        U, V = self.basis, self.activation # _, (n_sources, n_basis, n_frames)

        if n_remains > 0:
            U_low, U_high = U
            U_low, U_high = U_low.transpose(0, 4, 1, 2, 3), U_high.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_basis, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_low = np.sum(U_low[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            R_high = np.sum(U_high[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, 1, n_neighbors + 1, n_neighbors + 1)
            R_low, R_high = to_PSD(R_low, eps=eps), to_PSD(R_high, eps=eps)
            y_low, y_high = np.split(Y, [(n_blocks - n_remains)* n_neighbors], axis=2) # (n_sources, n_frames, (n_blocks - n_remains) * n_neighbors), (n_sources, n_frames, n_remains * (n_neighbors + 1))
            y_low, y_high = y_low.reshape(n_sources, n_frames, n_blocks - n_remains, n_neighbors), y_high.reshape(n_sources, n_frames, n_remains, n_neighbors + 1)

            yy_low = y_low[:, :, :, :, np.newaxis] * y_low[:, :, :, np.newaxis, :].conj() + eps * np.eye(n_neighbors) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            yy_high = y_high[:, :, :, :, np.newaxis] * y_high[:, :, :, np.newaxis, :].conj() + eps * np.eye(n_neighbors + 1) # (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            yy_low, yy_high = to_PSD(yy_low, eps=eps), to_PSD(yy_high, eps=eps)

            inv_R_low, inv_R_high = np.linalg.inv(R_low), np.linalg.inv(R_high) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            inv_R_low, inv_R_high = to_PSD(inv_R_low, eps=eps), to_PSD(inv_R_high, eps=eps)
            Ryy_low, Ryy_high = inv_R_low @ yy_low, inv_R_high @ yy_high # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)
            RU_low = inv_R_low[:, np.newaxis, :, :, :, :] @ U_low[:, :, np.newaxis, :, :, :] # (n_sources, n_basis, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            RU_high = inv_R_high[:, np.newaxis, :, :, :, :] @ U_high[:, :, np.newaxis, :, :, :] # (n_sources, n_basis, n_frames, n_remains, n_neighbors + 1, n_neighbors + 1)

            numerator_low = np.trace(RU_low @ Ryy_low[:, np.newaxis, :, :, :, :], axis1=-2, axis2=-1).real # (n_sources, n_basis, n_frames, n_blocks - n_remains)
            numerator_high = np.trace(RU_high @ Ryy_high[:, np.newaxis, :, :, :, :], axis1=-2, axis2=-1).real # (n_sources, n_basis, n_frames, n_remains)
            numerator = np.concatenate([numerator_low, numerator_high], axis=3) # (n_sources, n_basis, n_frames, n_blocks)
            denominator_low = np.trace(RU_low, axis1=-2, axis2=-1).real # (n_sources, n_basis, n_frames, n_blocks - n_remains)
            denominator_high = np.trace(RU_high, axis1=-2, axis2=-1).real # (n_sources, n_basis, n_frames, n_remains)
            denominator = np.concatenate([denominator_low, denominator_high], axis=3) # (n_sources, n_basis, n_frames, n_blocks)

            Ry_low = np.sum(inv_R_low * y_low[:, :, :, np.newaxis, :], axis=4) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors)
            Ry_high = np.sum(inv_R_high * y_high[:, :, :, np.newaxis, :], axis=4) # (n_sources, n_frames, n_remains, n_neighbors + 1)
            yRy_low, yRy_high = np.sum(y_low.conj() * Ry_low, axis=(2, 3)).real, np.sum(y_high.conj() * Ry_high, axis=(2, 3)).real # (n_sources, n_frames), (n_sources, n_frames)
            yRy = yRy_low + yRy_high

            U = U_low.transpose(0, 2, 3, 4, 1), U_high.transpose(0, 2, 3, 4, 1) # (n_sources, n_blocks - n_remains, n_neighbors, n_neighbors, n_basis), (n_sources, n_remains, n_neighbors + 1, n_neighbors + 1, n_basis)
        else:
            U = U.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            R = np.sum(U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            R = to_PSD(R, eps=eps) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            y = Y.reshape(n_sources, n_frames, n_blocks, n_neighbors) # (n_sources, n_frames, n_blocks, n_neighbors)

            yy = y[:, :, :, :, np.newaxis] * y[:, :, :, np.newaxis, :].conj() + eps * np.eye(n_neighbors) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            yy = to_PSD(yy, eps=eps)

            inv_R = np.linalg.inv(R) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            inv_R = to_PSD(inv_R, eps=eps)
            Ryy = inv_R @ yy # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            RU = inv_R[:, np.newaxis, :, :, :, :] @ U[:, :, np.newaxis, :, :, :] # (n_sources, n_basis, n_frames, n_blocks, n_neighbors, n_neighbors)
            numerator = np.trace(RU @ Ryy[:, np.newaxis, :, :, :, :], axis1=-2, axis2=-1).real # (n_sources, n_basis, n_frames, n_blocks)
            denominator = np.trace(RU, axis1=-2, axis2=-1).real # (n_sources, n_basis, n_frames, n_blocks)

            Ry = np.sum(inv_R * y[:, :, :, np.newaxis, :], axis=4) # (n_sources, n_frames, n_blocks, n_neighbors)
            yRy = np.sum(y.conj() * Ry, axis=(2, 3)).real # (n_sources, n_frames)
            
            U = U.transpose(0, 2, 3, 4, 1)
        
        pi = (nu + 2 * n_bins) / (nu + 2 * yRy) # (n_sources, n_frames)
        numerator, denominator = pi[:, np.newaxis, :] * np.sum(numerator, axis=3), np.sum(denominator, axis=3) # (n_sources, n_basis, n_frames), (n_sources, n_basis, n_frames)
        numerator[numerator < 0] = 0
        denominator[denominator < eps] = eps
        V = V * np.sqrt(numerator / denominator) # (n_sources, n_basis, n_frames)

        self.basis, self.activation = U, V

    def update_spatial_model_vcd(self):
        n_bins, n_frames = self.n_bins, self.n_frames
        n_sources, n_channels = self.n_sources, self.n_channels
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_remains = self.n_remains

        nu = self.nu
        eps = self.eps

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W) # (n_sources, n_bins, n_frames)
        X = X.transpose(1, 2, 0) # (n_bins, n_frames, n_channels)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)

        U, V = self.basis, self.activation # _, (n_sources, n_basis, n_frames)

        if n_remains > 0:
            W_low, W_high = np.split(W, [(n_blocks - n_remains) * n_neighbors], axis=0)
            W_low, W_high = W_low.reshape(n_blocks - n_remains, n_neighbors, n_sources, n_channels), W_high.reshape(n_remains, n_neighbors + 1, n_sources, n_channels)
            X_low, X_high = np.split(X, [(n_blocks - n_remains) * n_neighbors], axis=0)
            X_low, X_high = X_low.reshape(n_blocks - n_remains, n_neighbors, n_frames, n_channels), X_high.reshape(n_remains, n_neighbors + 1, n_frames, n_channels)
            U_low, U_high = U
            U_low, U_high = U_low.transpose(0, 4, 1, 2, 3), U_high.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors), (n_sources, n_basis, n_remains, n_neighbors + 1)

            XX_low = X_low[:, :, :, :, np.newaxis] * X_low[:, :, :, np.newaxis, :].conj() # (n_blocks - n_remains, n_neighbors, n_frames, n_channels, n_channels)
            XX_high = X_high[:, :, :, :, np.newaxis] * X_high[:, :, :, np.newaxis, :].conj() # (n_remains, n_neighbors + 1, n_frames, n_channels, n_channels)
            XX_low, XX_high = to_PSD(XX_low, eps=eps), to_PSD(XX_high, eps=eps)

            mask_low, mask_high = 1 - np.eye(n_neighbors), 1 - np.eye(n_neighbors + 1) # (n_neighbors, n_neighbors), (n_neighbors + 1, n_neighbors + 1)
            E_low = np.tile(np.eye(n_sources, n_channels), reps=(n_blocks - n_remains, 1, 1)) # (n_neighbors, n_sources, n_channels)
            E_high = np.tile(np.eye(n_sources, n_channels), reps=(n_remains, 1, 1)) # (n_neighbors + 1, n_sources, n_channels)

            for source_idx in range(n_sources):
                e_n_low, e_n_high = E_low[:, source_idx, :], E_high[:, source_idx, :] # (n_blocks - n_remains, n_channels), (n_remains, n_channels)

                U_n_low = U_low[source_idx, :, :, :, :] # (n_basis, n_blocks - n_remains, n_neighbors', n_neighbors)
                U_n_high = U_high[source_idx, :, :, :, :] # (n_basis, n_remains, n_neighbors' + 1, n_neighbors + 1)
                V_n = V[source_idx, :] # (n_basis, n_frames)
                R_n_low = np.sum(U_n_low[:, np.newaxis, :, :, :] * V_n[:, :, np.newaxis, np.newaxis, np.newaxis], axis=0) # (n_frames, n_blocks - n_remains, n_neighbors', n_neighbors)
                R_n_high = np.sum(U_n_high[:, np.newaxis, :, :, :] * V_n[:, :, np.newaxis, np.newaxis, np.newaxis], axis=0) # (n_frames, n_remains, n_neighbors' + 1, n_neighbors + 1)
                R_n_low, R_n_high = to_PSD(R_n_low, axis1=2, axis2=3, eps=eps), to_PSD(R_n_high, axis1=2, axis2=3, eps=eps)
                
                inv_R_n_low, inv_R_n_high = np.linalg.inv(R_n_low), np.linalg.inv(R_n_high) # (n_frames, n_blocks - n_remains, n_neighbors', n_neighbors), (n_frames, n_remains, n_neighbors' + 1, n_neighbors + 1)
                inv_R_n_low, inv_R_n_high = to_PSD(inv_R_n_low, eps=eps), to_PSD(inv_R_n_high, eps=eps)
                inv_R_n_low, inv_R_n_high = inv_R_n_low.transpose(1, 3, 2, 0), inv_R_n_high.transpose(1, 3, 2, 0) # (n_blocks - n_remains, n_neighbors, n_neighbors', n_frames), (n_remains, n_neighbors + 1, n_neighbors' + 1, n_frames)
                inv_R_ii_n_low, inv_R_ii_n_high = np.diagonal(inv_R_n_low, axis1=1, axis2=2).real, np.diagonal(inv_R_n_high, axis1=1, axis2=2).real # (n_blocks - n_remains, n_frames, n_neighbors), (n_remains, n_frames, n_neighbors + 1)
                inv_R_ii_n_low, inv_R_ii_n_high = inv_R_ii_n_low.transpose(0, 2, 1), inv_R_ii_n_high.transpose(0, 2, 1) # (n_blocks - n_remains, n_neighbors, n_frames), (n_remains, n_neighbors + 1, n_frames)

                for neighbor_idx in range(n_neighbors):
                    w_n_low, w_n_high = W_low[:, :, source_idx, :].conj(), W_high[:, :, source_idx, :].conj() # (n_blocks - n_remains, n_neighbors', n_channels), (n_remains, n_neighbors' + 1, n_channels)
                    Xw_n_low = np.sum(X_low.conj() * w_n_low[:, :, np.newaxis, :], axis=3) # (n_blocks - n_remains, n_neighbors', n_frames)
                    Xw_n_high = np.sum(X_high.conj() * w_n_high[:, :, np.newaxis, :], axis=3) # (n_remains, n_neighbors' + 1, n_frames)

                    y_n_low, y_n_high = Xw_n_low.conj(), Xw_n_high.conj() # (n_blocks - n_remains, n_neighbors, n_frames), (n_remains, n_neighbors + 1, n_frames)
                    Ry_n_low = np.sum(inv_R_n_low * y_n_low[:, :, np.newaxis, :], axis=1) # (n_blocks - n_remains, n_neighbors, n_frames)
                    Ry_n_high = np.sum(inv_R_n_high * y_n_high[:, :, np.newaxis, :], axis=1) # (n_remains, n_neighbors + 1, n_frames)
                    yRy_n_low = np.sum(y_n_low.conj() * Ry_n_low, axis=(0, 1)).real # (n_frames,)
                    yRy_n_high = np.sum(y_n_high.conj() * Ry_n_high, axis=(0, 1)).real # (n_frames,)
                    yRy_n = yRy_n_low + yRy_n_high # (n_frames,)
                    pi_n = (nu + 2 * n_bins) / (nu + 2 * yRy_n) # (n_frames,)

                    Q_in_low = inv_R_ii_n_low[:, neighbor_idx, :, np.newaxis, np.newaxis] * XX_low[:, neighbor_idx, :, :, :] # (n_blocks - n_remains, n_frames, n_channels, n_channels)
                    Q_in_low = np.mean(pi_n[np.newaxis, :, np.newaxis, np.newaxis] * Q_in_low, axis=1) # (n_blocks - n_remains, n_channels, n_channels)
                    Q_in_low = to_PSD(Q_in_low, eps=eps)

                    XXw_n_low = X_low[:, neighbor_idx, np.newaxis, :, :] * Xw_n_low[:, :, :, np.newaxis] # (n_blocks - n_remains, n_neighbors', n_frames, n_channels)
                    RXXw_n_low = np.mean(pi_n[np.newaxis, np.newaxis, :, np.newaxis] * inv_R_n_low[:, neighbor_idx, :, :, np.newaxis] * XXw_n_low, axis=2) # (n_blocks - n_remains, n_neighbors', n_channels)
                    gamma_in_low = np.sum(mask_low[np.newaxis, neighbor_idx, :, np.newaxis] * RXXw_n_low, axis=1) # (n_blocks - n_remains, n_channels)

                    WQ_in_low = W_low[:, neighbor_idx, :, :] @ Q_in_low # (n_blocks - n_remains, n_sources, n_channels)
                    zeta_in_low = np.linalg.solve(WQ_in_low, e_n_low) # (n_blocks - n_remains, n_channels)
                    zeta_hat_in_low = np.linalg.solve(Q_in_low, gamma_in_low) # (n_blocks - n_remains, n_channels)
                    eta_in_low = np.squeeze(zeta_in_low[:, np.newaxis, :].conj() @ Q_in_low @ zeta_in_low[:, :, np.newaxis], axis=(1, 2)) # (n_blocks - n_remains,)
                    eta_hat_in_low = np.squeeze(zeta_in_low[:, np.newaxis, :].conj() @ Q_in_low @ zeta_hat_in_low[:, :, np.newaxis], axis=(1, 2)) # (n_blocks - n_remains,)
                    
                    eta_in_low[np.abs(eta_in_low) < eps] = eps
                    condition_if_low = np.abs(eta_hat_in_low) < eps
                    eta_hat_in_low[condition_if_low] = eps
                    weight_low = (eta_hat_in_low / (2 * eta_in_low)) * (1 - np.sqrt(1 + 4 * eta_in_low / (np.abs(eta_hat_in_low)**2)))
                    weight_if_low = 1 / np.sqrt(eta_in_low)
                    weight_low[condition_if_low] = weight_if_low[condition_if_low]
                    w_in_low = weight_low[:, np.newaxis] * zeta_in_low - zeta_hat_in_low
                    W_low[:, neighbor_idx, source_idx, :] = w_in_low.conj()

                for neighbor_idx in range(n_neighbors + 1):
                    w_n_low, w_n_high = W_low[:, :, source_idx, :].conj(), W_high[:, :, source_idx, :].conj() # (n_blocks - n_remains, n_neighbors', n_channels), (n_remains, n_neighbors' + 1, n_channels)
                    Xw_n_low = np.sum(X_low.conj() * w_n_low[:, :, np.newaxis, :], axis=3) # (n_blocks - n_remains, n_neighbors', n_frames)
                    Xw_n_high = np.sum(X_high.conj() * w_n_high[:, :, np.newaxis, :], axis=3) # (n_remains, n_neighbors' + 1, n_frames)

                    y_n_low, y_n_high = Xw_n_low.conj(), Xw_n_high.conj() # (n_blocks - n_remains, n_neighbors, n_frames), (n_remains, n_neighbors + 1, n_frames)
                    Ry_n_low = np.sum(inv_R_n_low * y_n_low[:, :, np.newaxis, :], axis=1) # (n_blocks - n_remains, n_neighbors, n_frames)
                    Ry_n_high = np.sum(inv_R_n_high * y_n_high[:, :, np.newaxis, :], axis=1) # (n_remains, n_neighbors + 1, n_frames)
                    yRy_n_low = np.sum(y_n_low.conj() * Ry_n_low, axis=(0, 1)).real # (n_frames,)
                    yRy_n_high = np.sum(y_n_high.conj() * Ry_n_high, axis=(0, 1)).real # (n_frames,)

                    yRy_n = yRy_n_low + yRy_n_high # (n_frames,)
                    pi_n = (nu + 2 * n_bins) / (nu + 2 * yRy_n) # (n_frames,)

                    Q_in_high = inv_R_ii_n_high[:, neighbor_idx, :, np.newaxis, np.newaxis] * XX_high[:, neighbor_idx, :, :, :] # (n_remains, n_frames, n_channels, n_channels)
                    Q_in_high = np.mean(pi_n[np.newaxis, :, np.newaxis, np.newaxis] * Q_in_high, axis=1) # (n_remains, n_channels, n_channels)
                    Q_in_high = to_PSD(Q_in_high, eps=eps)

                    XXw_n_high = X_high[:, neighbor_idx, np.newaxis, :, :] * Xw_n_high[:, :, :, np.newaxis] # (n_remains, n_neighbors' + 1, n_frames, n_channels)
                    RXXw_n_high = np.mean(pi_n[np.newaxis, np.newaxis, :, np.newaxis] * inv_R_n_high[:, neighbor_idx, :, :, np.newaxis] * XXw_n_high, axis=2) # (n_remains, n_neighbors' + 1, n_channels)
                    gamma_in_high = np.sum(mask_high[np.newaxis, neighbor_idx, :, np.newaxis] * RXXw_n_high, axis=1) # (n_remains, n_channels)

                    WQ_in_high = W_high[:, neighbor_idx, :, :] @ Q_in_high # (n_blocks - n_remains, n_sources, n_channels), (n_remains, n_sources, n_channels)
                    zeta_in_high = np.linalg.solve(WQ_in_high, e_n_high) # (n_remains, n_channels)
                    zeta_hat_in_high = np.linalg.solve(Q_in_high, gamma_in_high) # (n_remains, n_channels)
                    eta_in_high = np.squeeze(zeta_in_high[:, np.newaxis, :].conj() @ Q_in_high @ zeta_in_high[:, :, np.newaxis], axis=(1, 2)) # (n_remains,)
                    eta_hat_in_high = np.squeeze(zeta_in_high[:, np.newaxis, :].conj() @ Q_in_high @ zeta_hat_in_high[:, :, np.newaxis], axis=(1, 2)) # (n_remains,)

                    eta_in_high[np.abs(eta_in_high) < eps] = eps
                    condition_if_high = np.abs(eta_hat_in_high) < eps
                    eta_hat_in_high[condition_if_high] = eps
                    weight_high = (eta_hat_in_high / (2 * eta_in_high)) * (1 - np.sqrt(1 + 4 * eta_in_high / (np.abs(eta_hat_in_high)**2)))
                    weight_if_high = 1 / np.sqrt(eta_in_high)
                    weight_high[condition_if_high] = weight_if_high[condition_if_high]
                    w_in_high = weight_high[:, np.newaxis] * zeta_in_high - zeta_hat_in_high
                    W_high[:, neighbor_idx, source_idx, :] = w_in_high.conj()
            
            W_low, W_high = W_low.reshape((n_blocks - n_remains) * n_neighbors, n_sources, n_channels), W_high.reshape(n_remains * (n_neighbors + 1), n_sources, n_channels)
            W = np.concatenate([W_low, W_high], axis=0) # (n_bins, n_sources, n_channels)
        else:
            W = W.reshape(n_blocks, n_neighbors, n_sources, n_channels)
            X = X.reshape(n_blocks, n_neighbors, n_frames, n_channels)
            U = U.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks, n_neighbors)

            XX = X[:, :, :, :, np.newaxis] * X[:, :, :, np.newaxis, :].conj() # (n_blocks, n_neighbors, n_frames, n_channels, n_channels)
            XX = to_PSD(XX, eps=eps)

            mask = 1 - np.eye(n_neighbors) # (n_neighbors, n_neighbors)
            E = np.tile(np.eye(n_sources, n_channels), reps=(n_blocks, 1, 1)) # (n_blocks, n_neighbors, n_sources, n_channels)

            for source_idx in range(n_sources):
                e_n = E[:, source_idx, :] # (n_blocks, n_channels)

                U_n, V_n = U[source_idx, :, :, :, :], V[source_idx, :] # (n_basis, n_blocks, n_neighbors', n_neighbors), (n_basis, n_frames)
                R_n = np.sum(U_n[:, np.newaxis, :, :, :] * V_n[:, :, np.newaxis, np.newaxis, np.newaxis], axis=0) # (n_frames, n_blocks, n_neighbors', n_neighbors)
                R_n = to_PSD(R_n, axis1=2, axis2=3, eps=eps)

                inv_R_n = np.linalg.inv(R_n) # (n_frames, n_blocks, n_neighbors', n_neighbors)
                inv_R_n = to_PSD(inv_R_n, eps=eps)
                inv_R_n = inv_R_n.transpose(1, 3, 2, 0) # (n_blocks, n_neighbors, n_neighbors', n_frames)
                inv_R_ii_n = np.diagonal(inv_R_n, axis1=1, axis2=2).real # (n_blocks, n_frames, n_neighbors)
                inv_R_ii_n = inv_R_ii_n.transpose(0, 2, 1) # (n_blocks, n_neighbors, n_frames)

                for neighbor_idx in range(n_neighbors):
                    w_n = W[:, :, source_idx, :].conj() # (n_blocks, n_neighbors', n_channels)
                    Xw_n = np.sum(X.conj() * w_n[:, :, np.newaxis, :], axis=3) # (n_blocks, n_neighbors', n_frames)

                    y_n = Xw_n.conj() # (n_blocks, n_neighbors, n_frames)
                    Ry_n = np.sum(inv_R_n * y_n[:, :, np.newaxis, :], axis=1) # (n_blocks, n_neighbors, n_frames)
                    yRy_n = np.sum(y_n.conj() * Ry_n, axis=(0, 1)).real # (n_frames,)
                    pi_n = (nu + 2 * n_bins) / (nu + 2 * yRy_n) # (n_frames,)

                    Q_in = inv_R_ii_n[:, neighbor_idx, :, np.newaxis, np.newaxis] * XX[:, neighbor_idx, :, :, :] # (n_blocks, n_frames, n_channels, n_channels)
                    Q_in = np.mean(pi_n[np.newaxis, :, np.newaxis, np.newaxis] * Q_in, axis=1) # (n_remains, n_channels, n_channels)
                    Q_in = to_PSD(Q_in, eps=eps)

                    XXw_n = X[:, neighbor_idx, np.newaxis, :, :] * Xw_n[:, :, :, np.newaxis] # (n_blocks, n_neighbors', n_frames, n_channels)
                    RXXw_n = np.mean(pi_n[np.newaxis, np.newaxis, :, np.newaxis] * inv_R_n[:, neighbor_idx, :, :, np.newaxis] * XXw_n, axis=2) # (n_remains, n_neighbors', n_channels)
                    gamma_in = np.sum(mask[np.newaxis, neighbor_idx, :, np.newaxis] * RXXw_n, axis=1) # (n_blocks, n_channels)

                    WQ_in = W[:, neighbor_idx, :, :] @ Q_in # (n_blocks, n_sources, n_channels)
                    zeta_in = np.linalg.solve(WQ_in, e_n) # (n_blocks, n_channels)
                    zeta_hat_in = np.linalg.solve(Q_in, gamma_in) # (n_blocks, n_channels)
                    eta_in = np.squeeze(zeta_in[:, np.newaxis, :].conj() @ Q_in @ zeta_in[:, :, np.newaxis], axis=(1, 2)) # (n_blocks,)
                    eta_hat_in = np.squeeze(zeta_in[:, np.newaxis, :].conj() @ Q_in @ zeta_hat_in[:, :, np.newaxis], axis=(1, 2)) # (n_blocks,)

                    eta_in[np.abs(eta_in) < eps] = eps
                    condition_if = np.abs(eta_hat_in) < eps
                    eta_hat_in[condition_if] = eps
                    weight = (eta_hat_in / (2 * eta_in)) * (1 - np.sqrt(1 + 4 * eta_in / (np.abs(eta_hat_in)**2)))
                    weight_if = 1 / np.sqrt(eta_in)
                    weight[condition_if] = weight_if[condition_if]
                    w_in = weight[:, np.newaxis] * zeta_in - zeta_hat_in
                    W[:, neighbor_idx, source_idx, :] = w_in.conj()
            
            W = W.reshape(n_bins, n_sources, n_channels)
        
        self.demix_filter = W

    def normalize_psdtf(self):
        if self.author.lower() in __authors_ipsdta__:
            self.normalize_psdtf_block_diagonal()
        else:
            raise NotImplementedError("Not support {}'s IPSDTA.".format(self.author))

    def normalize_psdtf_block_diagonal(self):
        n_remains = self.n_remains

        U, V = self.basis, self.activation # _, (n_sources, n_basis, n_frames)

        if n_remains > 0:
            U_low, U_high = U
            U_low, U_high = U_low.transpose(0, 4, 1, 2, 3), U_high.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks - 1, n_neighbors, n_neighbors), (n_sources, n_basis, 1, n_neighbors + n_remains, n_neighbors + n_remains)
            trace_low, trace_high = np.trace(U_low, axis1=3, axis2=4).real, np.trace(U_high, axis1=3, axis2=4).real
            trace = np.concatenate([trace_low, trace_high], axis=2) # (n_sources, n_basis, n_blocks)
            trace = trace.sum(axis=2) # (n_sources, n_basis)
            U_low, U_high = U_low / trace[:, :, np.newaxis, np.newaxis, np.newaxis], U_high / trace[:, :, np.newaxis, np.newaxis, np.newaxis]
            U = U_low.transpose(0, 2, 3, 4, 1), U_high.transpose(0, 2, 3, 4, 1)
            V = V * trace[:, :, np.newaxis]
        else:
            U = U.transpose(0, 4, 1, 2, 3)
            trace = np.trace(U, axis1=3, axis2=4).real
            trace = trace.sum(axis=2) # (n_sources, n_basis)
            U = U / trace[:, :, np.newaxis, np.newaxis, np.newaxis]
            U = U.transpose(0, 2, 3, 4, 1)
            V = V * trace[:, :, np.newaxis]

        self.basis, self.activation = U, V

    def compute_negative_loglikelihood(self):
        if self.author.lower() in __authors_ipsdta__:
            loss = self.compute_negative_loglikelihood_block_diagonal()
        else:
            raise ValueError("Not support {}'s IPSDTA.".format(self.author))

        return loss

    def compute_negative_loglikelihood_block_diagonal(self):
        n_bins, n_frames = self.n_bins, self.n_frames
        n_sources = self.n_sources
        n_blocks, n_neighbors = self.n_blocks, self.n_neighbors
        n_remains = self.n_remains

        nu = self.nu
        eps = self.eps

        X, W_Hermite = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W_Hermite) # (n_sources, n_bins, n_frames)
        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)
        U, V = self.basis, self.activation

        if n_remains > 0:
            U_low, U_high = U
            U_low, U_high = U_low.transpose(0, 4, 1, 2, 3), U_high.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_basis, n_remains, n_neighbors + 1, n_neighbors + 1)
            R_low = np.sum(U_low[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors)
            R_high = np.sum(U_high[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, 1, n_neighbors + 1, n_neighbors + 1)
            y_low, y_high = np.split(Y, [(n_blocks - n_remains)* n_neighbors], axis=2) # (n_sources, n_frames, (n_blocks - n_remains) * n_neighbors), (n_sources, n_frames, n_remains * (n_neighbors + 1))
            
            R_low, R_high = to_PSD(R_low, axis1=3, axis2=4, eps=eps), to_PSD(R_high, axis1=3, axis2=4, eps=eps)
            y_low = y_low.reshape(n_sources, n_frames, n_blocks - n_remains, n_neighbors, 1) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, 1)
            y_high = y_high.reshape(n_sources, n_frames, n_remains, n_neighbors + 1, 1) # (n_sources, n_frames, 1, n_neighbors + 1, 1)

            inv_R_low, inv_R_high = np.linalg.inv(R_low), np.linalg.inv(R_high) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1s, n_neighbors + n_remains)
            inv_R_low, inv_R_high = to_PSD(inv_R_low, eps=eps), to_PSD(inv_R_high, eps=eps)
            Ry_low = inv_R_low @ y_low # (n_sources, n_frames, n_blocks - n_remains, n_neighbors, 1)
            Ry_high = inv_R_high @ y_high # (n_sources, n_frames, n_remains, n_neighbors + 1, 1)
            Ry_low = Ry_low.reshape(n_sources, n_frames, (n_blocks - n_remains) * n_neighbors)
            Ry_high = Ry_high.reshape(n_sources, n_frames, n_remains * (n_neighbors + 1))
            Ry = np.concatenate([Ry_low, Ry_high], axis=2) # (n_sources, n_frames, n_bins)

            eigvals_low, eigvals_high = np.linalg.eigvalsh(R_low), np.linalg.eigvalsh(R_high) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1)
            eigvals_low[eigvals_low < eps], eigvals_high[eigvals_high < eps] = eps, eps
            logdet_low, logdet_high = np.log(eigvals_low), np.log(eigvals_high) # (n_sources, n_frames, n_blocks - n_remains, n_neighbors), (n_sources, n_frames, n_remains, n_neighbors + 1)
            logdet_low, logdet_high = logdet_low.sum(axis=3), logdet_high.sum(axis=3) # (n_sources, n_frames, n_blocks - n_remains), (n_sources, n_frames, n_remains)
            logdet_R = np.concatenate([logdet_low, logdet_high], axis=2)# (n_sources, n_frames, n_blocks)
        else:
            U = U.transpose(0, 4, 1, 2, 3) # (n_sources, n_basis, n_blocks, n_neighbors, n_neighbors)
            R = np.sum(U[:, :, np.newaxis, :, :, :] * V[:, :, :, np.newaxis, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            y = Y # (n_sources, n_frames, n_blocks * n_neighbors)

            R = to_PSD(R, axis1=3, axis2=4, eps=eps)
            y = y.reshape(n_sources, n_frames, n_blocks, n_neighbors, 1) # (n_sources, n_frames, n_blocks, n_neighbors, 1)

            inv_R = np.linalg.inv(R) # (n_sources, n_frames, n_blocks, n_neighbors, n_neighbors)
            inv_R = to_PSD(inv_R, eps=eps)
            Ry = inv_R @ y # (n_sources, n_frames, n_blocks, n_neighbors, 1)
            Ry = Ry.reshape(n_sources, n_frames, n_blocks * n_neighbors)

            eigvals = np.linalg.eigvalsh(R) # (n_sources, n_frames, n_blocks, n_neighbors)
            eigvals[eigvals < eps] = eps
            logdet_R = np.log(eigvals) # (n_sources, n_frames, n_blocks, n_neighbors)
            logdet_R = logdet_R.sum(axis=3) # (n_sources, n_frames, n_blocks)
        
        logdet_R = logdet_R.sum(axis=2) # (n_sources, n_frames)
        
        eigvals = np.linalg.eigvals(W_Hermite) # (n_bins, n_sources)
        eigvals = np.abs(eigvals)
        eigvals[eigvals < eps] = eps
        logdet_W = np.log(eigvals)
        logdet_W = logdet_W.sum(axis=1) # (n_bins,)

        y_Hermite = Y.conj()
        yRy = np.sum(y_Hermite * Ry, axis=2).real # (n_sources, n_frames)
        loss = logdet_R.sum() + (nu + 2 * n_bins) / 2 * np.sum(np.log(1 + (2 / nu) * yRy)) - 2 * n_frames * logdet_W.sum()

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

def _test_gauss_ipsdta(n_basis=10, author='Ikeshita'):
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
        write_wav("data/IPSDTA/GaussIPSDTA-{}/partitioning0/mixture-{}_estimated-iter{}-{}.wav".format(author, sr, iteration, idx), signal=_estimated_signal, sr=sr)
    
    plt.figure()
    plt.plot(ipsdta.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/IPSDTA/GaussIPSDTA-{}/partitioning0/loss.png'.format(author), bbox_inches='tight')
    plt.close()

def _test_t_ipsdta(n_basis=10):
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

    ipsdta = tIPSDTA(n_basis=n_basis)
    print(ipsdta)
    estimation = ipsdta(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_channels):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/IPSDTA/tIPSDTA/partitioning0/mixture-{}_estimated-iter{}-{}.wav".format(sr, iteration, idx), signal=_estimated_signal, sr=sr)
    
    plt.figure()
    plt.plot(ipsdta.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/IPSDTA/tIPSDTA/partitioning0/loss.png', bbox_inches='tight')
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
    os.makedirs("data/ILRMA/GaussIPSDTA-Ikeshita/partitioning0", exist_ok=True)
    os.makedirs("data/ILRMA/GaussIPSDTA-Kondo/partitioning0", exist_ok=True)
    os.makedirs("data/ILRMA/tIPSDTA/partitioning0", exist_ok=True)

    """
    Use multichannel room impulse response database.
    Download database from "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/"
    """

    _test_conv()

    print("="*10, "Gauss-IPSDTA", "="*10)
    print("-"*10, "Ikeshita's Gauss-IPSDTA", "-"*10)
    _test_gauss_ipsdta(n_basis=2, author='Ikeshita')
    print()

    print("-"*10, "Kondo's Gauss-IPSDTA", "-"*10)
    _test_gauss_ipsdta(n_basis=2, author='Kondo')
    print()

    print("="*10, "t-IPSDTA", "="*10)
    _test_t_ipsdta(n_basis=2)
    print()