import numpy as np

EPS=1e-12

__metrics__ = ['EUC', 'KL', 'IS']

class MNMFbase:
    def __init__(self, n_bases=10, n_sources=None, callback=None, eps=EPS):
        """
        Args:
            n_bases: number of bases 
        """
        self.callback = callback
        self.eps = eps
        self.input = None
        self.n_bases = n_bases
        self.n_sources = n_sources
        self.loss = []
    
    def _reset(self, **kwargs):
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        n_bases = self.n_bases
        n_sources = self.n_sources

        X = self.input
        n_channels, n_bins, n_frames = X.shape

        if n_sources is None:
            n_sources = n_channels
        self.n_sources, self.n_channels = n_channels, n_sources
        self.n_bins, self.n_frames = n_bins, n_frames

        A = np.random.randn(n_channels, n_sources, dtype=np.complex128)
        self.mix_filter = np.tile(A, reps=(n_bins, 1, 1))
        self.estimation = self.separate(X)

        self.base = np.random.rand(n_sources, n_bins, n_bases)
        self.activation = np.random.rand(n_sources, n_bases, n_frames)
    
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
        covariance_input, covariance_output = self.covariance_input, self.covariance_output # (n_sources, n_bins, n_channels, n_channels), (n_channels, n_bins, n_sources, n_sources)
        estimation = covariance_output @ np.linalg.inv(covariance_input) @ input # (n_channels, n_bins, n_sources, n_sources)
        output = estimation

        return output
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")

class GaussMNMF(MNMFbase):
    """
    Reference: "Multichannel Nonnegative Matrix Factorization in Convolutive Mixtures for Audio Source Separation"
    See https://ieeexplore.ieee.org/document/5229304
    """
    def __init__(self, n_bases=10, n_sources=None, reference_id=0, callback=None, eps=EPS):
        """
        Args:
        """
        super().__init__(n_bases=n_bases, n_sources=n_sources, callback=callback, eps=eps)

        self.reference_id = reference_id
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")

class tMNMF(MNMFbase):
    """
    Reference: "Student's t multichannel nonnegative matrix factorization for blind source separation"
    See https://ieeexplore.ieee.org/document/7602889
    """
    def __init__(self, n_bases=10, n_sources=None, reference_id=0, callback=None, eps=EPS):
        """
        Args:
        """
        super().__init__(n_bases=n_bases, n_sources=n_sources, callback=callback, eps=eps)

        self.reference_id = reference_id
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")

class FastGaussMNMF(MNMFbase):
    """
    Reference: "Fast Multichannel Source Separation Based on Jointly Diagonalizable Spatial Covariance Matrices"
    """
    def __init__(self, n_bases=10, n_sources=None, reference_id=0, callback=None, eps=EPS):
        """
        Args:
        """
        super().__init__(n_bases=n_bases, n_sources=n_sources, callback=callback, eps=eps)

        self.reference_id = reference_id
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")