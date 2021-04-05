import numpy as np
import torch

from algorithm.stft import stft, istft
from algorithm.projection_back import projection_back

EPS=1e-12
THRESHOLD=1e+12

class IDLMAbase:
    def __init__(self, normalize=True, callback=None, dnn_flooring=1e-5, eps=EPS):
        self.callback = callback
        self.eps = eps
        self.input = None
        self.loss = []

        self.normalize = normalize
        self.dnn_flooring = dnn_flooring

    def _reset(self, dnn=None, **kwargs):
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        X = self.input

        n_channels, n_bins, n_frames = X.shape
        n_sources = n_channels # n_channels == n_sources

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        W = np.eye(n_sources, n_channels, dtype=np.complex128)
        self.demix_filter = np.tile(W, reps=(n_bins, 1, 1))
        self.estimation = self.separate(X, demix_filter=W)

        self.dnn = dnn
        self.dnn_output = np.ones((n_sources, n_bins, n_frames))

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

        for idx in range(iteration):
            self.update_once()

            loss = self.compute_negative_loglikelihood()
            self.loss.append(loss)

            if self.callback is not None:
                self.callback(self)
        
        X, W = input, self.demix_filter
        output = self.separate(X, demix_filter=W)

        return output

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

class GaussIDLMA(IDLMAbase):
    """

    """
    def __init__(self, domain=2, normalize='power', reference_id=0, callback=None, dnn_flooring=1e-5, eps=EPS, threshold=THRESHOLD):
        """
        Args:
            normalize <str>: 'power': power based normalization, or 'projection-back': projection back based normalization.
            threshold <float>: threshold for condition number when computing (WU)^{-1}.
        """
        super().__init__(normalize=normalize, callback=callback, dnn_flooring=dnn_flooring, eps=eps)

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

        for idx in range(iteration):
            self.update_once()

            loss = self.compute_negative_loglikelihood()
            self.loss.append(loss)

            if self.callback is not None:
                self.callback(self)
        
        reference_id = self.reference_id
        X, W = input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output

    def update_once(self, is_source_model_update=True):
        if is_source_model_update:
            self.update_source_model()
        self.update_space_model()

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        self.estimation = Y
        
        if self.normalize:
            if self.normalize == 'projection-back':
                scale = projection_back(Y, reference=X[self.reference_id])
                Y = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
                X = X.transpose(1,0,2) # (n_bins, n_channels, n_frames)
                X_Hermite = X.transpose(0,2,1).conj() # (n_bins, n_frames, n_channels)
                W = Y.transpose(1,0,2) @ X_Hermite @ np.linalg.inv(X @ X_Hermite) # (n_bins, n_sources, n_channels)
            else:
                raise ValueError("Not support normalization based on {}. Choose 'power' or 'projection-back'".format(self.normalize))
        else:
            raise ValueError("Set normalize=True")
        
        self.demix_filter = W
        self.estimation = Y
    
    def update_source_model(self):
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        P = np.abs(Y)**2

        dnn_output = self.estimate_by_dnn(P)
        self.dnn_output = dnn_output

        if self.dnn_flooring:
            self.floor_dnn_output()

    def update_space_model(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins

        domain = self.domain
        eps, threshold = self.eps, self.threshold

        X, W = self.input, self.demix_filter
        R = self.dnn_output[...,np.newaxis, np.newaxis]**(2/domain)
        
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
    
    def estimate_by_dnn(self, input):
        domain = self.domain
        input = input**(domain/2)

        with torch.no_grad():
            input = torch.Tensor(input)
            if next(self.dnn.parameters()).is_cuda:
                input = input.cuda()
            output = self.dnn(input)

        output = output.cpu().numpy()
        output = output**(2/domain)

        return output
    
    def floor_dnn_output(self):
        floor = self.dnn_flooring
        dnn_output = self.dnn_output
        dnn_output = np.maximum(dnn_output, floor)
        self.dnn_output = dnn_output

    def compute_negative_loglikelihood(self):
        n_frames = self.n_frames
        domain = self.domain
        eps = self.eps

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        dnn_output = self.dnn_output
        P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)
        R = dnn_output**(2 / domain)
        R[R < eps] = eps
        loss = np.sum(P / R + np.log(R)) - 2 * n_frames * np.sum(np.log(np.abs(np.linalg.det(W))))

        return loss