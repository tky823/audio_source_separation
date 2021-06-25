import numpy as np

def projection_back(Y, reference):
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
        X = reference[np.newaxis,:,:].transpose(1, 0, 2) # (n_bins, n_channels, n_frames)
        Y = Y.transpose(1, 0, 2) # (n_bins, n_sources, n_frames)
        Y_Hermite = Y.transpose(0, 2, 1).conj() # (n_bins, n_frames, n_sources)
        YY_Hermite_inverse = np.linalg.inv(Y @ Y_Hermite) # (n_bins, n_sources, n_sources)
        A = X @ Y_Hermite @ YY_Hermite_inverse # (n_bins, n_channels, n_sources)
        scale = A[:,0,:].transpose(1, 0) # (n_sources, n_bins)
    elif n_dims == 3:
        # Y: (n_sources, n_bins, n_frames)
        # X: (n_channels, n_bins, n_frames)
        X = reference.transpose(1, 0, 2) # (n_bins, n_channels, n_frames)
        Y = Y.transpose(1, 0, 2) # (n_bins, n_sources, n_frames)
        Y_Hermite = Y.transpose(0, 2, 1).conj() # (n_bins, n_frames, n_sources)
        YY_Hermite_inverse = np.linalg.inv(Y @ Y_Hermite) # (n_bins, n_sources, n_sources)
        A = X @ Y_Hermite @ YY_Hermite_inverse # (n_bins, n_channels, n_sources)
        scale = A.transpose(1, 2, 0) # (n_channels, n_sources, n_bins)
    else:
        raise ValueError("reference.ndim is expected 2 or 3, but given {}.".format(n_dims))

    return scale

if __name__ == '__main__':
    pass