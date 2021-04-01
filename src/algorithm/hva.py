import numpy as np

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