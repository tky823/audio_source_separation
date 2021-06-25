import numpy as np

def minimum_distortion_principle(Y, reference):
    """
    Args:
        Y: (n_sources, n_bins, n_frames)
        reference: (n_bins, n_frames) or (n_channels, n_bins, n_frames)
    Returns:
        scale: (n_sources, n_bins) or (n_channels, n_sources, n_bins)
    """
    # TODO: Use pseudo inverse ?
    n_dims = reference.ndim
    if n_dims == 2:
        # Y: (n_sources, n_bins, n_frames)
        # X: (1, n_bins, n_frames)
        X = reference[np.newaxis, :, :]
    elif n_dims == 3:
        # Y: (n_sources, n_bins, n_frames)
        # X: (n_channels, n_bins, n_frames)
        X = reference
    else:
        raise ValueError("reference.ndim is expected 2 or 3, but given {}.".format(n_dims))
    
    YX_conj = np.sum(Y[np.newaxis, :, :, :].conj() * X[:, np.newaxis, :, :], axis=3) # (n_channels, n_sources, n_bins)
    YY_conj = np.sum(np.abs(Y)**2, axis=2) # (n_sources, n_bins)
    scale = YX_conj / YY_conj # (n_channels, n_sources, n_bins)

    if n_dims == 2:
        scale = scale[0] # (1, n_sources, n_bins) -> (n_sources, n_bins)

    return scale

def generalized_minimum_distortion_principle():
    return

if __name__ == '__main__':
    pass