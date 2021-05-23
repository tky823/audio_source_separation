import numpy as np

EPS=1e-12

def delay_sum_beamform(input, steering_vector, reference_id=0):
    """
    Args:
        input (n_channels, n_bins, n_frames)
        steering_vector (n_bins, n_channels, n_sources)
    Returns:
        output (n_sources, n_bins, n_frames)
    """
    X, A = input, steering_vector
    a_Hermite = A.transpose(2,1,0)[...,np.newaxis].conj() # (n_sources, n_channels, n_bins, 1)
    Y = np.sum(a_Hermite * X, axis=1) # (n_sources, n_bins, n_frames)
    A = A.transpose(1,2,0)[...,np.newaxis] # (n_channels, n_sources, n_bins, 1)
    output = A[reference_id,:,:,:] * Y

    return output

def ml_beamform(input, steering_vector, covariance, reference_id=0, eps=EPS):
    """
    Args:
        input (n_channels, n_bins, n_frames)
        steering_vector (n_bins, n_channels, n_sources)
        covariance (n_bins, n_channels, n_channels)
    Returns:
        output (n_sources, n_bins, n_frames)
    """
    X, A = input.transpose(1,0,2), steering_vector
    R = covariance
    R_inverse = np.linalg.inv(R) # (n_bins, n_channels, n_channels)
    A_Hermite = A.conj() # (n_bins, n_channels, n_sources)
    numerator = R_inverse @ A # (n_bins, n_channels, n_sources)
    denominator = np.sum(A_Hermite * numerator, axis=1, keepdims=True) # (n_bins, 1, n_sources)
    denominator[denominator < eps] = eps
    W = numerator / denominator # (n_bins, n_channels, n_sources)
    W = W.transpose(0,2,1) # (n_bins, n_sources, n_channels)
    Y = W @ X # (n_bins, n_sources, n_frames)
    Y = Y.transpose(1,0,2) # (n_sources, n_bins, n_frames)
    A = A.transpose(1,2,0)[...,np.newaxis] # (n_channels, n_sources, n_bins, 1)
    output = A[reference_id,:,:,:] * Y

    return output

def mvdr_beamform(input, steering_vector, reference_id=0, eps=EPS):
    """
    Args:
        input (n_channels, n_bins, n_frames)
        steering_vector (n_bins, n_channels, n_sources)
    Returns:
        output (n_sources, n_bins, n_frames)
    """
    X = input.transpose(1,0,2)
    R = np.mean(X[:,:,np.newaxis,:] * X[:,np.newaxis,:,:].conj(), axis=3) # (n_bins, n_channels, n_channels)
    output = ml_beamform(input, steering_vector, covariance=R, reference_id=reference_id, eps=eps)

    return output


class DelaySumBeamformer:
    def __init__(self, steering_vector=None, reference_id=0):
        """
        Args:
            steering_vector (n_bins, n_channels, n_sources)
            reference_id <int>
        """
        self.steering_vector = steering_vector
        self.reference_id = reference_id
    
    def __call__(self, input, steering_vector=None):
        """
        Args:
            input (n_channels, n_bins, n_frames)
            steering_vector (n_bins, n_channels, n_sources)
        Returns:
            output (n_sources, n_bins, n_frames)
        """
        self.input = input

        if steering_vector is not None:
            self.steering_vector = steering_vector
        elif self.steering_vector is None:
            raise ValueError("Specify steering vector.")

        output = delay_sum_beamform(input, self.steering_vector, reference_id=self.reference_id)
        self.estimation = output

        return output

class MVDRBeamformer:
    def __init__(self, steering_vector, reference_id=0, eps=EPS):
        """
        Args:
            steering_vector (n_bins, n_channels, n_sources)
            reference_id <int>
        """
        self.steering_vector = steering_vector
        self.reference_id = reference_id
        self.eps = eps
    
    def __call__(self, input, steering_vector=None, covariance=None):
        """
        Args:
            input (n_channels, n_bins, n_frames)
            steering_vector (n_bins, n_channels, n_sources)
        Returns:
            output (n_sources, n_bins, n_frames)
        """
        self.input = input

        if steering_vector is not None:
            self.steering_vector = steering_vector
        elif self.steering_vector is None:
            raise ValueError("Specify steering vector.")

        output = mvdr_beamform(input, self.steering_vector, covariance=covariance, reference_id=self.reference_id, eps=self.eps)
        self.estimation = output

        return output

class MaxSNRBeamformer:
    def __init__(self, steering_vector, reference_id=0, eps=EPS):
        self.steering_vector = steering_vector
        self.reference_id = reference_id
        self.eps = eps
    
    def __call__(self, input, steering_vector=None):
        """
        Args:
            input (n_channels, n_bins, n_frames)
        Returns:
            output (n_sources, n_bins, n_frames)
        """
        if steering_vector is not None:
            self.steering_vector = steering_vector
        elif self.steering_vector is None:
            raise ValueError("Specify steering vector.")


def _convolve_mird(titles, reverb=0.160, degrees=[0], mic_intervals=[8,8,8,8,8,8,8], mic_indices=[0], samples=None):
    intervals = '-'.join([str(interval) for interval in mic_intervals])

    T_min = None

    for title in titles:
        source, _ = read_wav("data/single-channel/{}.wav".format(title))
        T = len(source)
        if T_min is None or T < T_min:
            T_min = T

    mixed_signals = []

    for mic_idx in mic_indices:
        _mixture = 0
        for title_idx in range(len(titles)):
            degree = degrees[title_idx]
            title = titles[title_idx]
            rir_path = "data/MIRD/Reverb{:.3f}_{}/Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_{:.3f}s)_{}_2m_{:03d}.mat".format(reverb, intervals, reverb, intervals, degree)
            rir_mat = loadmat(rir_path)

            rir = rir_mat['impulse_response']

            if samples is not None:
                rir = rir[:samples]

            source, sr = read_wav("data/single-channel/{}.wav".format(title))
            _mixture = _mixture + np.convolve(source[:T_min], rir[:, mic_idx])
        
        mixed_signals.append(_mixture)
    
    mixed_signals = np.array(mixed_signals)

    return mixed_signals

def _test(method='DSBF'):
    # Room impulse response
    sr = 16000
    reverb = 0.16
    duration = 0.5
    samples = int(duration * sr)
    mic_intervals = [3, 3, 3, 8, 3, 3, 3]
    mic_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    mic_position = np.array([[0.13, 0], [0.10, 0], [0.07, 0], [0.04, 0], [-0.04, 0], [-0.07, 0], [-0.10, 0], [-0.13, 0]])
    degrees = [0, 90]
    titles = ['man-16000', 'woman-16000']

    n_sources, n_channels = len(degrees), len(mic_indices)
    mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_intervals=mic_intervals, mic_indices=mic_indices, samples=samples)
    _, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 2048, 1024
    n_bins = fft_size//2 + 1
    frequency = np.arange(0, n_bins) * sr / fft_size
    mixture = stft(mixed_signal, fft_size=fft_size, hop_size=hop_size) # (n_channels, n_bins, n_frames)

    # Steeing vectors
    degrees = np.array(degrees) / 180 * np.pi
    x_source, y_source = np.sin(degrees), np.cos(degrees) # (n_sources,)
    source_position = np.vstack([x_source, y_source]).transpose(1,0) # (n_sources, 2)
    steering_vector = np.exp(2j * np.pi * frequency[:,np.newaxis,np.newaxis] * np.sum(source_position * mic_position[:,np.newaxis,:], axis=2) / sound_speed) # (n_bins, n_channels, n_sources)
    steering_vector = steering_vector / np.sqrt(len(mic_indices))

    if method == 'DSBF':
        beamformer = DelaySumBeamformer(steering_vector=steering_vector)
    elif method == 'MVDR':
        beamformer = MVDRBeamformer(steering_vector=steering_vector)
    else:
        raise NotImplementedError("Not support {} beamformer".format(method))

    estimation = beamformer(mixture)

    spectrogram = np.abs(estimation)
    log_spectrogram = 10 * np.log10(spectrogram**2)
    N, F_bin, T_bin = log_spectrogram.shape
    t = np.arange(T_bin + 1)
    f = np.arange(F_bin + 1)

    for n in range(N):
        plt.figure()
        plt.pcolormesh(t, f, log_spectrogram[n], cmap='jet')
        plt.savefig("data/Beamform/{}/specrtogram-{}.png".format(method, n), bbox_inches='tight')
        plt.close()

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)

    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_sources):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/Beamform/{}/mixture-{}_estimated-{}.wav".format(method, sr, idx), signal=_estimated_signal, sr=sr)

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    from utils.utils_audio import read_wav, write_wav
    from algorithm.stft import stft, istft

    plt.rcParams['figure.dpi'] = 200

    sound_speed=340

    os.makedirs('data/Beamform/DSBF', exist_ok=True)
    os.makedirs('data/Beamform/MVDR', exist_ok=True)

    _test('DSBF')
    _test('MVDR')

elif __name__ == 'bss.beamform':
    import warnings

    warnings.warn("in progress", UserWarning)