import numpy as np

def whitening(input, zero_mean=True, channel_first=True):
    """
    Args:
        input (n_channels, T)
    Returns:
        output (n_channels, T)
    """
    assert zero_mean, "`zero_mean` must be True."
    assert channel_first, "`channel_first` must be True."

    self_cov = input @ input.T
    w, v = np.linalg.eig(self_cov)
    W, inv_W = np.diag(np.sqrt(w)), np.diag(1 / np.sqrt(w))
    output = inv_W @ v.T @ input

    return output

def _test():
    s_0, sr = sf.read("data/single-channel/man-16000.wav")
    s_1, sr = sf.read("data/single-channel/woman-16000.wav")

    T = min(len(s_0), len(s_1))
    s_0, s_1 = s_0[:T], s_1[:T]

    s = np.vstack([s_0, s_1])
    A = np.array([[0.2, 0.5], [-0.8, 0.4]])
    x = A @ s

    plt.figure(figsize=(12, 8))
    plt.scatter(x[0], x[1], color='black', s=5)
    plt.axis('equal')
    plt.savefig("data/whitening/x.png", bbox_inches='tight')
    plt.close()

    x_whitened = whitening(x)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(x_whitened[0], x_whitened[1], color='black', s=5)
    plt.axis('equal')
    plt.savefig("data/whitening/x_whitened.png", bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import soundfile as sf

    plt.rcParams['font.size'] = 18
    _test()
