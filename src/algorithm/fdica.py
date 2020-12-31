import numpy as np

from algorithm.projection_back import projection_back

EPS=1e-12

class FDICAbase:
    def __init__(self, eps=EPS):
        self.input = None

        self.eps = eps
    
    def _reset(self):
        assert self.input is not None, "Specify data!"

        X = self.input

        n_channels, n_bins, n_frames = X.shape
        n_sources = n_channels # n_channels == n_sources

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        W = np.eye(n_channels, dtype=np.complex128)
        self.demix_filter = np.tile(W, reps=(n_bins, 1, 1))
        self.estimation = self.separate(X, demix_filter=W)
        
    def __call__(self, input, iteration=100):
        """
        Args:
            input (n_channels, n_bins, n_frames)
        Returns:
            output (n_channels, n_bins, n_frames)
        """
        self.input = input

        self._reset()

        for idx in range(iteration):
            self.update_once()
        
        X, W = input, self.demix_filter
        output = self.separate(X, demix_filter=W)

        return output

    def update_once(self):
        raise NotImplementedError("Implement 'update' function")
    
    def separate(self, input, demix_filter):
        """
        Args:
            input (n_channels, n_bins, n_frames): 
            demix_filter (n_bins, n_sources, n_channels): 
        Returns:
            output (n_channels, n_bins, n_frames): 
        """
        input = input.transpose(1,0,2)
        estimation = demix_filter @ input
        output = estimation.transpose(1,0,2)

        return output

class FDICA(FDICAbase):
    def __init__(self, distr='laplace', lr=1e-3, reference_id=0, eps=EPS):
        super().__init__(eps=eps)

        self.distr = distr
        self.lr = lr
        self.reference_id = reference_id
    
    def update_once(self):
        reference_id = self.reference_id
        X = self.input

        if self.distr == 'laplace':
            self.update_laplace()
        else:
            raise NotImplementedError("Cannot support {} distribution.".format(self.distr))

        W = self.demix_filter
        Y = self.separate(X, demix_filter=W)
        
        scale = projection_back(Y, reference=X[reference_id])
        Y_hat = Y * scale[...,np.newaxis].conj() # (N, I, J)

        _Y_hat = Y_hat.transpose(1,0,2) # (I, N, J)
        _X = X.transpose(1,0,2) # (I, M, J)
        X_Hermite = X.transpose(1,2,0).conj() # (I, J, M)
        XX_inverse = np.linalg.inv(_X @ X_Hermite)
        self.demix_filter = _Y_hat @ X_Hermite @ XX_inverse
        self.estimation = Y_hat

    def update_laplace(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        lr = self.lr
        eps = self.eps

        W = self.demix_filter
        Y = self.estimation
        eye = np.eye(n_sources, n_channels, dtype=np.complex128)

        _, _, n_frames = Y.shape
        
        Y = Y.transpose(1,0,2) # (n_bins, n_sources, n_frames)
        Y_Hermite = Y.transpose(0,2,1).conj() # (n_bins, n_frames, n_sources)
        denominator = np.abs(Y)
        denominator[denominator < eps] = eps
        Phi = Y / denominator # (n_bins, n_sources, n_frames)

        W = W + lr * ((eye - (Phi @ Y_Hermite) / n_frames) @ W) # (n_bins, n_sources, n_channels)

        self.demix_filter = W

def _convolve_mird(titles, reverb=0.160, degrees=[0], mic_intervals=[8,8,8,8,8,8,8], mic_indices=[0], samples=None):
    intervals = '-'.join([str(interval) for interval in mic_intervals])

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
            _mixture = _mixture + np.convolve(source, rir[:, mic_idx])
        
        mixed_signals.append(_mixture)
    
    mixed_signals = np.array(mixed_signals)

    return mixed_signals


def _test():
    np.random.seed(111)
    
    # Room impulse response
    reverb = 0.16
    duration = 0.5
    samples = int(duration * 16000)
    mic_intervals = [8, 8, 8, 8, 8, 8, 8]
    mic_indices = [2, 5]
    degrees = [60, 300]
    titles = ['man-16000', 'woman-16000']

    mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_intervals=mic_intervals, mic_indices=mic_indices, samples=samples)

    n_sources, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 2048, 1024
    
    mixture = []
    for _mixed_signal in mixed_signal:
        _mixture = stft(_mixed_signal, fft_size=fft_size, hop_size=hop_size)
        mixture.append(_mixture)
    mixture = np.array(mixture)

    # FDICA
    n_bases = 10
    n_channels = len(titles)
    iteration = 1000

    fdica = FDICA()
    estimation = fdica(mixture, iteration=iteration)

    estimated_signal = []
    for _estimation in estimation:
        _estimated_signal = istft(_estimation, fft_size=fft_size, hop_size=hop_size, length=T)
        estimated_signal.append(_estimated_signal)
    estimated_signal = np.array(estimated_signal)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_channels):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/FDICA/mixture-16000_estimated-iter{}-{}.wav".format(iteration, idx), signal=_estimated_signal, sr=16000)

def _test_conv():
    reverb = 0.16
    duration = 0.5
    samples = int(duration * 16000)
    mic_indices = [2, 5]
    degrees = [60, 300]
    titles = ['man-16000', 'woman-16000']
    
    mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_indices=mic_indices, samples=samples)

    write_wav("data/multi-channel/mixture-16000.wav", mixed_signal.T, sr=16000)


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    from utils.utils_audio import read_wav, write_wav
    from algorithm.stft import stft, istft

    plt.rcParams['figure.dpi'] = 200

    os.makedirs("data/multi-channel", exist_ok=True)
    os.makedirs("data/FDICA", exist_ok=True)

    """
    Use multichannel room impulse response database.
    Download database from "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/"
    """

    # _test_conv()
    _test()