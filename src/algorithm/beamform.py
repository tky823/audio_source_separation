import numpy as np

EPS=1e-12

class DelaySumBeamFormer:
    def __init__(self, steering_vector=None, eps=EPS):
        self.steering_vector = steering_vector
        self.eps = eps
    
    def __call__(self, input):
        pass

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
            rir_path = "data/MIRD/Reverb{:.3f}_{}/Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_{:.3f}s)_{}_1m_{:03d}.mat".format(reverb, intervals, reverb, intervals, degree)
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
    mic_intervals = [8, 8, 8, 8, 8, 8, 8]
    mic_indices = [3, 4, 5]
    degrees = [60, 300]
    titles = ['man-16000', 'woman-16000']
    doa = (np.array(degrees) / 180) * np.pi
    source_position = np.vstack([np.cos(doa), np.sin(doa)]).transpose(1,0) # (n_sources, 1, 2)
    mic_position = np.array([[0.04, 0], [-0.04, 0], [-0.12, 0]])[:,np.newaxis,:] # (n_channels, 2)

    n_sources, n_channels = len(degrees), len(mic_indices)
    mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_intervals=mic_intervals, mic_indices=mic_indices, samples=samples)
    _, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 2048, 1024
    n_bins = fft_size//2 + 1
    mixture = stft(mixed_signal, fft_size=fft_size, hop_size=hop_size) # (n_channels, n_bins, n_frames)

    # Steeing vectors
    omega = np.arange(n_bins) * sr / fft_size # (n_bins,)
    omega = omega[:,np.newaxis,np.newaxis] # (n_bins, 1, 1)
    inner_product = np.sum(source_position * mic_position, axis=2) # (n_channels, n_sources)
    steering_vector = np.exp(2j * np.pi * omega * inner_product / sound_speed) / np.sqrt(n_channels) # (n_channels, n_sources)
    norm = np.sqrt(np.sum(steering_vector**2, axis=0)) # (n_sources,)
    norm[norm < EPS] = EPS
    steering_vector = steering_vector / norm # (n_bins, n_channels, n_sources)
    steering_vector = steering_vector.transpose(2,1,0)[...,np.newaxis] # (n_sources, n_channels, n_bins, 1)
    estimation = np.sum(steering_vector.conj() * mixture, axis=1, keepdims=True) # (n_sources, 1, n_bins, n_frames)
    estimation = estimation * steering_vector # (n_sources, n_channels, n_bins, n_frames)
    estimation = estimation[:,0,:,:] # (n_sources, n_bins, n_frames)
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

    sound_speed=340

    os.makedirs('data/Beamform/DSBF', exist_ok=True)

    _test('DSBF')