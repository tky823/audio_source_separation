import numpy as np

from algorithm.projection_back import projection_back

EPS = 1e-12

class IPSDTAbase:
    """
    Independent Positive Semi-Definite Tensor Analysis
    """
    def __init__(self, n_basis=10, callbacks=None, recordable_loss=True, eps=EPS):
        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None
        self.eps = eps
        
        self.n_basis = n_basis

        self.input = None
        self.recordable_loss = recordable_loss
        if self.recordable_loss:
            self.loss = []
        else:
            self.loss = None
    
    def _reset(self, **kwargs):
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        n_basis = self.n_basis

        X = self.input

        n_channels, n_bins, n_frames = X.shape
        n_sources = n_channels # n_channels == n_sources

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        if not hasattr(self, 'demix_filter'):
            W = np.eye(n_sources, n_channels, dtype=np.complex128)
            self.demix_filter = np.tile(W, reps=(n_bins, 1, 1))
        else:
            W = self.demix_filter.copy()
            self.demix_filter = W
        
        self.estimation = self.separate(X, demix_filter=W)

        if not hasattr(self, 'basis'):
            U = np.random.rand(n_sources, n_basis, n_bins, n_bins) + 1j * np.random.rand(n_sources, n_basis, n_bins, n_bins) # should be positive semi-definite
            self.basis = (U + U.transpose(0, 1, 3, 2).conj()) / 2
        else:
            self.basis = self.basis.copy()
        if not hasattr(self, 'activation'):
            self.activation = np.random.rand(n_sources, n_basis, n_frames)
        else:
            self.activation = self.activation.copy()
    
    def __call__(self, input, iteration=100, **kwargs):
        """
        Args:
            input (n_channels, n_bins, n_frames)
        Returns:
            output (n_channels, n_bins, n_frames)
        """
        self.input = input

        self._reset(**kwargs)

        if self.recordable_loss:
            loss = self.compute_negative_loglikelihood()
            self.loss.append(loss)

        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self)

        for idx in range(iteration):
            self.update_once()

            if self.recordable_loss:
                loss = self.compute_negative_loglikelihood()
                self.loss.append(loss)

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)
        
        X, W = input, self.demix_filter
        output = self.separate(X, demix_filter=W)
        self.estimation = output

        return output
    
    def __repr__(self):
        s = "IPSDTA("
        s += "n_basis={n_basis}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        raise NotImplementedError("Implement 'update_once' method.")
    
    def separate(self, input, demix_filter):
        """
        Args:
            input (n_channels, n_bins, n_frames): 
            demix_filter (n_bins, n_sources, n_channels): 
        Returns:
            output (n_channels, n_bins, n_frames): 
        """
        input = input.transpose(1, 0, 2)
        estimation = demix_filter @ input
        output = estimation.transpose(1, 0, 2)

        return output
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement `compute_negative_loglikelihood` method.")

class GaussIPSDTA(IPSDTAbase):
    """
    """
    def __init__(self, n_basis=10, callbacks=None, recordable_loss=True, eps=EPS):
        """
        Args:
        """
        super().__init__(n_basis=n_basis, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)
    
    def __call__(self, input, iteration=100, **kwargs):
        """
        Args:
            input (n_channels, n_bins, n_frames)
        Returns:
            output (n_channels, n_bins, n_frames)
        """
        self.input = input

        self._reset(**kwargs)

        if self.recordable_loss:
            loss = self.compute_negative_loglikelihood()
            self.loss.append(loss)
        
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self)

        for idx in range(iteration):
            self.update_once()

            if self.recordable_loss:
                loss = self.compute_negative_loglikelihood()
                self.loss.append(loss)

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)

            X, W = input, self.demix_filter
            Y = self.separate(X, demix_filter=W)
        
        reference_id = self.reference_id
        
        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[..., np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output

    def __repr__(self):
        s = "Gauss-IPSDTA("
        s += "n_basis={n_basis}"
        s += ")"

        return s.format(**self.__dict__)

    def update_once(self):
        self.update_source_model()
        self.update_spatial_model()
    
    def update_source_model(self):
        raise NotImplementedError("Implement `update_source_model` method.")
    
    def update_spatial_model(self):
        raise NotImplementedError("Implement `update_spatial_model` method.")
    
    def compute_negative_loglikelihood(self):
        n_frames = self.n_frames

        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W) # (n_sources, n_bins, n_frames)
        
        U, V = self.basis, self.activation # (n_sources, n_basis, n_bins, n_bins), (n_sources, n_basis, n_frames)
        R = np.sum(U[:, :, np.newaxis, :, :] * V[:, :, :, np.newaxis, np.newaxis], axis=1) # (n_sources, n_frames, n_bins, n_bins)

        Y = Y.transpose(0, 2, 1) # (n_sources, n_frames, n_bins)
        Y_Hermite = Y.conj()
        sRs = Y_Hermite[:, :, np.newaxis, :] @ np.linalg.inv(R) @ Y[:, :, :, np.newaxis] # (n_sources, n_frames, 1, 1)
        loss = np.sum(sRs.squeeze(axis=(2, 3)) + np.log(np.linalg.det(R).real)) - 2 * n_frames * np.sum(np.log(np.abs(np.linalg.det(W))))

        return loss

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

def _test_conv():
    sr = 16000
    reverb = 0.16
    duration = 0.5
    samples = int(duration * sr)
    mic_indices = [2, 5]
    degrees = [60, 300]
    titles = ['man-16000', 'woman-16000']

    wav_path = "data/multi-channel/mixture-{}.wav".format(sr)

    if not os.path.exists(wav_path):
        mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_indices=mic_indices, samples=samples)
        write_wav(wav_path, mixed_signal.T, sr=sr)

def _test_gauss_ipsdta(n_basis=10):
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

    # IPSDTA
    n_channels = len(titles)
    iteration = 50

    ipsdta = GaussIPSDTA(n_basis=n_basis)
    print(ipsdta)
    estimation = ipsdta(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_channels):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/IPSDTA/GaussIPSDTA/partitioning0/mixture-{}_estimated-iter{}-{}.wav".format(sr, iteration, idx), signal=_estimated_signal, sr=sr)
    
    plt.figure()
    plt.plot(ipsdta.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/IPSDTA/GaussIPSDTA/partitioning0/loss.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.io import loadmat

    from utils.utils_audio import read_wav, write_wav
    from transform.stft import stft, istft

    plt.rcParams['figure.dpi'] = 200

    os.makedirs("data/multi-channel", exist_ok=True)
    os.makedirs("data/ILRMA/GaussIPSDTA/partitioning0", exist_ok=True)

    """
    Use multichannel room impulse response database.
    Download database from "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/"
    """

    _test_conv()

    print("="*10, "Gauss-IPSDTA", "="*10)
    _test_gauss_ipsdta(n_basis=2)
    print()