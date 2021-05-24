import numpy as np


def pca(input):
    """
    Args:
        input (n_channels, n_bins, n_frames)
    Returns:
        output (n_channels, n_bins, n_frames)
    """

    if input.ndim == 3:
        X = input.transpose(1, 2, 0)
        covariance = np.mean(X[:, :, :, np.newaxis] * X[:, :, np.newaxis, :].conj(), axis=1) # (n_bins, n_channels, n_channels)
        _, w = np.linalg.eigh(covariance) # (n_bins, n_channels), (n_bins, n_channels, n_channels)
        X = X @ w.conj()
        output = X.transpose(2, 0, 1)
    else:
        raise ValueError("Invalid dimension.")

    return output