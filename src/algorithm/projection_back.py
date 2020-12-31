import numpy as np

def projection_back(Y, reference):
    """
    Args:
        Y: (n_channels, n_bins, n_frames)
        reference: (n_bins, n_frames)
    Returns:
        scale: (n_channels, n_bins)
    """
    n_channels, n_bins, _ = Y.shape

    numerator = np.sum(Y * reference.conj(), axis=2) # (n_channels, n_bins)
    denominator = np.sum(np.abs(Y)**2, axis=2) # (n_channels, n_bins)
    scale = np.ones((n_channels, n_bins), dtype=np.complex128)
    indices = denominator > 0.0
    scale[indices] = numerator[indices] / denominator[indices]

    return scale