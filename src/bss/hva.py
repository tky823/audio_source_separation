import numpy as np
import scipy.signal as ss

EPS=1e-12

class HVAbase:
    def __init__(self, callback=None, eps=EPS):
        self.callback = callback
        self.eps = eps
        self.input = None
        # self.n_bases = n_bases
        self.loss = []

        # self.partitioning = partitioning
        # self.normalize = normalize

class HVA(HVAbase):
    """
    Harmonic Vector Analysis
    Reference: Determined BSS based on time-frequency masking and its application to harmonic vector analysis
    See https://arxiv.org/abs/2004.14091
    """
    def __init__(self, callback=None, eps=EPS):
        super().__init__(callback=callback, eps=eps)
    
    def update_once(self):
        pass

    def apply_masking(self, x, lambda, kappa):
        y = np.log(np.abs(x) + 1e-3)
        y_mean = y.mean(axis=2)
        y = y - y_mean
        z = ss.fft(y) / y.shape[2]
        M = np.minimum(1, np.abs(z) / lambda)
        for i in range(kappa):
            M = (1 - np.cos(np.pi * M)) / 2
        z = M * z
        z = ss.ifft(z) * y.shape[2]
        y = z
        y = y + y_mean
        y = np.exp(2 * y)
        output = (y / y.sum())**(1 / y.shape[0])

        return output
        