import numpy as np

EPS=1e-12

class ILRMAbase:
    def __init__(self, n_bases=10, eps=EPS):
        self.input = None
        self.n_bases = n_bases

        self.eps = eps
    
    def _reset(self):
        assert self.input is not None, "Specify data!"

        n_bases = self.n_bases
        eps = self.eps 

        n_channels, F_bin, T_bin = self.input.shape

        self.n_channels, self.n_sources = n_channels, n_channels # n_channels == n_sources
        self.F_bin, self.T_bin = F_bin, T_bin

        demix_filter = np.eye(n_channels, dtype=np.complex128)
        self.demix_filter = np.tile(demix_filter, reps=(F_bin, 1, 1))
        self.base = np.random.rand(n_channels, F_bin, n_bases)
        self.activation = np.random.rand(n_channels, n_bases, T_bin)
        
    def __call__(self, input, iteration=100):
        """
        Args:
            input (n_channels, F_bin, T_bin, 2)
        Returns:
            output (n_channels, F_bin, T_bin, 2)
        """
        self.input = input[...,0] + 1j * input[...,1]

        self._reset()

        for idx in range(iteration):
            self.update()
        
        estimation = self.separate(self.input, self.demix_filter)
        estimated_real, estimated_imag = estimation.real, estimation.imag
        output = np.concatenate([estimated_real[...,np.newaxis], estimated_imag[...,np.newaxis]], axis=3)

        return output

    def update(self):
        raise NotImplementedError("Implement 'update' function")

    def projection_back(self, Y, ref):
        """
        Args:
            Y: (n_channels, F_bin, T_bin)
            ref: (F_bin, T_bin)
        Returns:
            scale: (n_channels, F_bin)
        """
        n_channels, F_bin, _ = Y.shape

        numerator = np.sum(Y * ref.conj(), axis=2) # (n_channels, F_bin)
        denominator = np.sum(np.abs(Y)**2, axis=2) # (n_channels, F_bin)
        scale = np.ones((n_channels, F_bin), dtype=np.complex128)
        indices = denominator > 0.0
        scale[indices] = numerator[indices] / denominator[indices]

        return scale
    
    def separate(self, input, demix_filter):
        """
        Args:
            input (n_channels, F_bin, T_bin): complex value
            demix_filter (F_bin, n_sources, n_channels): complex value
        Returns:
            output (n_channels, F_bin, T_bin): complex value
        """
        input = input.transpose(1,0,2)
        estimation = demix_filter @ input
        output = estimation.transpose(1,0,2)

        return output

class GaussILRMA(ILRMAbase):
    def __init__(self, n_bases=10, ref_id=0, eps=EPS):
        super().__init__(n_bases=n_bases, eps=eps)

    def update(self):
        pass