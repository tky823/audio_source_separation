import numpy as np

from algorithm.projection_back import projection_back

EPS=1e-12

class IVAbase:
    def __init__(self, eps=EPS):
        self.input = None

        self.eps = eps
    
    def _reset(self):
        assert self.input is not None, "Specify data!"

        X = self.input

        n_channels, n_bins, n_frames = X.shape
        n_sources = n_channels # n_channels == n_sources

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        W = np.eye(n_channels, dtype=np.complex128)
        self.demix_filter = np.tile(W, reps=(n_bins, 1, 1))
        self.estimation = self.separate(X, demix_filter=W)
        
    def __call__(self, input, iteration=100):
        """
        Args:
            input (n_channels, n_bins, n_frames)
        Returns:
            output (n_channels, n_bins, n_frames)
        """
        self.input = input

        self._reset()

        for idx in range(iteration):
            self.update_once()
        
        X, W = input, self.demix_filter
        output = self.separate(X, demix_filter=W)

        return output

    def update_once(self):
        raise NotImplementedError("Implement 'update' function")
    
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

class AuxIVA(IVAbase):
    def __init__(self, eps=EPS):
        super().__init__(eps=eps)