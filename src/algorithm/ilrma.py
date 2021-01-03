import numpy as np

from algorithm.projection_back import projection_back

EPS=1e-12

class ILRMAbase:
    def __init__(self, n_bases=10, partitioning=False, normalize=True, callback=None, eps=EPS):
        self.callback = callback
        self.eps = eps
        self.input = None
        self.n_bases = n_bases
        self.loss = [] # TODO: monitoring loss

        self.partitioning = partitioning
        self.normalize = normalize
        
    
    def _reset(self):
        assert self.input is not None, "Specify data!"

        n_bases = self.n_bases

        X = self.input

        n_channels, n_bins, n_frames = X.shape
        n_sources = n_channels # n_channels == n_sources

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        W = np.eye(n_channels, dtype=np.complex128)
        self.demix_filter = np.tile(W, reps=(n_bins, 1, 1))
        self.base = np.random.rand(n_channels, n_bins, n_bases)
        self.activation = np.random.rand(n_channels, n_bases, n_frames)
        self.estimation = self.separate(X, demix_filter=W)

        if self.partitioning:
            self.latent = np.ones(n_sources, n_bases) / n_sources
        
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

            if self.callback is not None:
                self.callback(self)
        
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

class GaussILRMA(ILRMAbase):
    def __init__(self, n_bases=10, partitioning=False, normalize=True, reference_id=0, callback=None, eps=EPS):
        super().__init__(n_bases=n_bases, partitioning=partitioning, normalize=normalize, callback=callback, eps=eps)

        self.reference_id = reference_id
    
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

            if self.callback is not None:
                self.callback(self)
        
        reference_id = self.reference_id
        X, W = input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output

    def update_once(self):
        reference_id = self.reference_id

        X = self.input

        self.update_source_model()
        self.update_space_model()

        W = self.demix_filter
        Y = self.separate(X, demix_filter=W)
        self.estimation = Y
        """
        if self.normalize:
            T = self.base
            P = np.abs(Y)**2
            aux = np.sqrt(P.mean(axis=(1,2))) # (n_sources,)
            W = W / aux[np.newaxis,:,np.newaxis]
            Y = Y / aux[:,np.newaxis,np.newaxis]
            if self.partitioning:
                pass
            else:
                pass
                # self.base = T / aux[:,np.newaxis,np.newaxis]**2
        """
    
    def update_source_model(self):
        eps = self.eps

        X, W = self.input, self.demix_filter
        estimation = self.separate(X, demix_filter=W)
        P = np.abs(estimation)**2

        T, V = self.base, self.activation

        if self.partitioning:
            Z = self.latent

            raise NotImplementedError("Not support for partitioning function.")

        # Update bases
        V_transpose = V.transpose(0,2,1)
        TV = T @ V
        TV[TV < eps] = eps
        division, TV_inverse = P / TV**2, 1 / TV
        TVV = TV_inverse @ V_transpose
        TVV[TVV < eps] = eps
        T = T * np.sqrt(division @ V_transpose / TVV)
        T[T < eps] = eps

        # Update activations
        T_transpose = T.transpose(0,2,1)
        TV = T @ V
        TV[TV < eps] = eps
        division, TV_inverse = P / (TV**2), 1 / TV
        TTV = T_transpose @ TV_inverse
        TTV[TTV < eps] = eps
        V = V * np.sqrt(T_transpose @ division / TTV)
        V[V < eps] = eps

        self.bases, self.activation = T, V

    def update_space_model(self):
        n_sources = self.n_sources
        n_bins = self.n_bins
        eps = self.eps

        X, W = self.input, self.demix_filter
        TV = self.base @ self.activation
        
        X = X.transpose(1,2,0) # (n_bins, n_frames, n_channels)
        X = X[...,np.newaxis]
        X_Hermite = X.transpose(0,1,3,2).conj()
        XX = X @ X_Hermite # (n_bins, n_frames, n_channels, n_channels)
        R = TV[...,np.newaxis, np.newaxis] # (n_sources, n_bins, n_frames, 1, 1)
        U = XX / R
        U = U.mean(axis=2) # (n_sources, n_bins, n_channels, n_channels)

        for source_idx in range(n_sources):
            # W: (n_bins, n_sources, n_channels), U: (N, n_bins, n_channels, n_channels)
            U_n = U[source_idx] # (n_bins, n_channels, n_channels)
            WU = W @ U_n # (n_bins, n_sources, n_channels)
            # TODO: condition number
            WU_inverse = np.linalg.inv(WU)[...,source_idx] # (n_bins, n_sources, n_channels)
            w = WU_inverse.conj() # (n_bins, n_channels)
            wUw = w[:, np.newaxis, :].conj() @ U_n @ w[:, :, np.newaxis]
            denominator = np.sqrt(wUw[...,0])
            W[:, source_idx, :] = w / denominator

        self.demix_filter = W

def _convolve_mird(titles, reverb=0.160, degrees=[0], mic_intervals=[8,8,8,8,8,8,8], mic_indices=[0], samples=None):
    intervals = '-'.join([str(interval) for interval in mic_intervals])

    T_min = None

    for title in titles:
        source, sr = read_wav("data/single-channel/{}.wav".format(title))
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

def _test():
    np.random.seed(111)
    
    # Room impulse response
    sr = 16000
    reverb = 0.16
    duration = 0.5
    samples = int(duration * sr)
    mic_intervals = [8, 8, 8, 8, 8, 8, 8]
    mic_indices = [2, 5]
    degrees = [60, 300]
    titles = ['man-16000', 'woman-16000']

    mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_intervals=mic_intervals, mic_indices=mic_indices, samples=samples)

    n_sources, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 2048, 1024
    mixture = stft(mixed_signal, fft_size=fft_size, hop_size=hop_size)

    # ILRMA
    n_bases = 10
    n_channels = len(titles)
    iteration = 200

    gauss_ilrma = GaussILRMA(n_bases=n_bases)
    estimation = gauss_ilrma(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_channels):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/ILRMA/mixture-{}_estimated-iter{}-{}.wav".format(sr, iteration, idx), signal=_estimated_signal, sr=sr)

def _test_conv():
    sr = 16000
    reverb = 0.16
    duration = 0.5
    samples = int(duration * sr)
    mic_indices = [2, 5]
    degrees = [60, 300]
    titles = ['man-16000', 'woman-16000']
    
    mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_indices=mic_indices, samples=samples)

    write_wav("data/multi-channel/mixture-{}.wav".format(sr), mixed_signal.T, sr=sr)

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.io import loadmat

    from utils.utils_audio import read_wav, write_wav
    from algorithm.stft import stft, istft

    plt.rcParams['figure.dpi'] = 200

    os.makedirs("data/multi-channel", exist_ok=True)
    os.makedirs("data/ILRMA", exist_ok=True)

    """
    Use multichannel room impulse response database.
    Download database from "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/"
    """

    # _test_conv()
    _test()