from types import new_class
import numpy as np

def projection_back(Y, reference):
    """
    Args:
        Y: (n_channels, n_bins, n_frames)
        reference: (n_bins, n_frames) or (n_sources, n_bins, n_frames)
    Returns:
        scale: (n_channels, n_bins) or (n_sources, n_channels, n_bins)
    """
    # TODO: Use pseudo inverse ?
    n_dims = reference.ndim
    if n_dims == 2:
        # Y: (n_channels, n_bins, n_frames)
        # X: (n_bins, n_frames)
        X = reference[:,np.newaxis,:] # (n_bins, 1, n_frames)
        Y = Y.transpose(1,0,2) # (n_bins, n_channels, n_frames)
        Y_Hermite = Y.transpose(0,2,1).conj() # (n_bins, n_frames, n_channels)
        YY_Hermite = Y @ Y_Hermite # (n_bins, n_channels, n_channels)
        YY_Hermite_inverse = np.linalg.inv(YY_Hermite)# (n_bins, n_channels, n_channels)
        A = X @ Y_Hermite @ YY_Hermite_inverse # (n_bins, 1, n_channels)
        scale = A.squeeze().transpose(1,0) # (n_channels, n_bins)
    elif n_dims == 3:
        # Y: (n_channels, n_bins, n_frames)
        # X: (n_sources, n_bins, n_frames)
        X = reference.transpose(1,0,2) # (n_bins, n_sources, n_frames)
        Y = Y.transpose(1,0,2) # (n_bins, n_channels, n_frames)
        Y_Hermite = Y.transpose(0,2,1).conj() # (n_bins, n_frames, n_channels)
        YY_Hermite = Y @ Y_Hermite # (n_bins, n_channels, n_channels)
        YY_Hermite_inverse = np.linalg.inv(YY_Hermite)# (n_bins, n_channels, n_channels)
        A = X @ Y_Hermite @ YY_Hermite_inverse # (n_bins, n_sources, n_channels)
        scale = A.transpose(1,2,0) # (n_sources, n_channels, n_bins)
    else:
        raise ValueError("reference.ndim is expected 2 or 3, but given {}.".format(n_dims))

    return scale
"""
# pyroomacoustics
def projection_back(Y, reference):
    Args:
        Y: (n_channels, n_bins, n_frames)
        reference: (n_bins, n_frames)
    Returns:
        scale: (n_channels, n_bins)
    numerator = np.sum(Y * reference.conj(), axis=2) # (n_channels, n_bins)
    denominator = np.sum(np.abs(Y)**2, axis=2) # (n_channels, n_bins)
    scale = np.ones((n_channels, n_bins), dtype=np.complex128)
    indices = denominator > 0.0
    scale[indices] = numerator[indices] / denominator[indices]

    return scale
"""

if __name__ == '__main__':
    a = np.array([[1,2], [3,4]])
    print(a.ndim)