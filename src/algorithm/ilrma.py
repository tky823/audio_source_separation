import numpy as np

EPS=1e-12

class ILRMAbase:
    def __init__(self, n_bases=10, eps=EPS):
        self.input = None
        self.n_bases = n_bases

        self.eps = eps
    
    def _reset(self):
        assert self.input is not None, "Specify data!"

        n_bases = self.n_bases
        eps = self.eps 

        n_channels, n_bins, n_frames = self.input.shape

        self.n_channels, self.n_sources = n_channels, n_channels # n_channels == n_sources
        self.n_bins, self.n_frames = n_bins, n_frames

        demix_filter = np.eye(n_channels, dtype=np.complex128)
        self.demix_filter = np.tile(demix_filter, reps=(n_bins, 1, 1))
        self.base = np.random.rand(n_channels, n_bins, n_bases)
        self.activation = np.random.rand(n_channels, n_bases, n_frames)
        
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
        
        output = self.separate(self.input, self.demix_filter)

        return output

    def update_once(self):
        raise NotImplementedError("Implement 'update' function")

    def projection_back(self, Y, reference):
        """
        Args:
            Y: (n_channels, n_bins, n_frames)
            ref: (n_bins, n_frames)
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
    def __init__(self, n_bases=10, reference_id=0, eps=EPS):
        super().__init__(n_bases=n_bases, eps=eps)

        self.reference_id = reference_id

    def update_once(self):
        reference_id = self.reference_id

        X = self.input

        self.update_source_model()
        self.update_space_model()

        W = self.demix_filter

        Y = self.separate(X, demix_filter=W)
        Z = self.projection_back(Y, reference=X[reference_id])

        self.estimation = Y * Z[...,np.newaxis]
    
    def update_source_model(self):
        eps = self.eps

        X, W = self.input, self.demix_filter
        
        estimation = self.separate(X, demix_filter=W)
        target = np.abs(estimation)**2

        # Update bases
        T, V = self.base, self.activation
        V_transpose = V.transpose(0,2,1)
        TV = T @ V
        TV[TV < eps] = eps
        division, TV_inverse = target / TV**2, 1 / TV
        TVV = TV_inverse @ V_transpose
        TVV[TVV < eps] = eps
        self.base = T * np.sqrt(division @ V_transpose / TVV)

        # Update activations
        T, V = self.base, self.activation
        T_transpose = T.transpose(0,2,1)
        TV = T @ V
        TV[TV < eps] = eps
        division, TV_inverse = target / (TV**2), 1 / TV
        TTV = T_transpose @ TV_inverse
        TTV[TTV < eps] = eps
        self.activation = V * np.sqrt(T_transpose @ division / TTV)
    
    def update_space_model(self):
        n_sources = self.n_sources
        eps = self.eps

        X, W = self.input, self.demix_filter

        TV = self.base @ self.activation
        R = TV[:, np.newaxis, np.newaxis, :, :] # (N, 1, 1, I, J) power domain
        U = np.einsum('nij,mij->nmij', X, X.conj()) / (R + eps)
        U = U.mean(axis=4) # (N, M, M, I)
        
        for source_idx in range(n_sources):
            # W (I, N, M) U (N, M, M, I)
            U_n = U[source_idx].transpose(2,0,1) # (I, M, M)
            WU = W @ U_n # (I, N, M)
            WU_inverse = np.linalg.inv(WU) # (I, M, N)
            w = WU_inverse[...,source_idx] # (I, M)
            w = w[...,np.newaxis] # (I, M, 1)
            w_Hermite = w.transpose(0,2,1).conj() # (I, 1, M)
            wUw = w_Hermite @ U_n @ w # (I, 1, 1)
            w = w / np.sqrt(wUw)
            w = w.squeeze() # (I, M)
            W[:, source_idx, :] = w.conj()

        self.demix_filter = W

def _convolve_mird(titles, reverb=0.160, degrees=[0], mic_indices=[0], samples=None):
    mixed_signals = []

    for mic_idx in mic_indices:
        _mixture = 0
        for title_idx in range(len(titles)):
            degree = degrees[title_idx]
            title = titles[title_idx]
            rir_path = "data/MIRD/Reverb{:.3f}_8-8-8-8-8-8-8/Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_{:.3f}s)_8-8-8-8-8-8-8_1m_{:03d}.mat".format(reverb, reverb, degree)
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
    mic_indices = [2, 5]
    degrees = [60, 300]
    titles = ['man-16000', 'woman-16000']

    mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_indices=mic_indices, samples=samples)

    n_sources, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 2048, 1024
    n_bases = 6
    n_channels = len(titles)
    iteration = 100

    mixture = []
    for _mixed_signal in mixed_signal:
        _mixture = stft(_mixed_signal, fft_size=fft_size, hop_size=hop_size)
        mixture.append(_mixture)
    mixture = np.array(mixture)

    gauss_ilrma = GaussILRMA(n_bases=n_bases)
    estimation = gauss_ilrma(mixture, iteration=iteration)

    estimated_signal = []
    for _estimation in estimation:
        _estimated_signal = istft(_estimation, fft_size=fft_size, hop_size=hop_size, length=T)
        estimated_signal.append(_estimated_signal)
    estimated_signal = np.array(estimated_signal)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_channels):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/ILRMA/mixture-16000_estimated-iter{}-{}.wav".format(iteration, idx), signal=_estimated_signal, sr=16000)

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
    import numpy as np
    from scipy.io import loadmat

    from utils.utils_audio import read_wav, write_wav
    from algorithm.stft import stft, istft

    plt.rcParams['figure.dpi'] = 200

    os.makedirs("data/multi-channel", exist_ok=True)
    os.makedirs("data/ILRMA", exist_ok=True)

    # _test_conv()
    _test()