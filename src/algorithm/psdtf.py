import numpy as np

"""
Positive Semidefinite Tensor Factorization
"""

class PSDTFbase:
    def __init__(self):
        pass

class LDPSDTF(PSDTFbase):
    """
    Reference: "Beyond NMF: Time-Domain Audio Source Separation without Phase Reconstruction"
    See https://archives.ismir.net/ismir2013/paper/000032.pdf
    """
    def __init__(self):
        super().__init__()