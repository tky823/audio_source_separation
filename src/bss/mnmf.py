import numpy as np

from algorithm.stft import stft, istft

EPS=1e-12

__metrics__ = ['EUC', 'KL', 'IS']
__authors__ = ['sawada', 'ozerov']

class MNMFbase:
    def __init__(self, n_bases=10, n_sources=None, callback=None, eps=EPS):
        """
        Args:
            n_bases: number of bases 
        """
        self.callback = callback
        self.eps = eps
        self.input = None
        self.n_bases = n_bases
        self.n_sources = n_sources
        self.loss = []
    
    def _reset(self, **kwargs):
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        n_bases = self.n_bases
        n_sources = self.n_sources

        X = self.input
        n_channels, n_bins, n_frames = X.shape

        if n_sources is None:
            n_sources = n_channels
        self.n_sources, self.n_channels = n_channels, n_sources
        self.n_bins, self.n_frames = n_bins, n_frames

        H = np.eye(n_channels)
        self.spatial = np.tile(H, reps=(n_bins, n_bases, 1, 1))
        self.covariance_input = X[:, :, :, np.newaxis] * X[:, :, np.newaxis, :].conj()
        A = np.random.rand(n_channels, n_sources).astype(np.complex128)
        self.mix_filter = np.tile(A, reps=(n_bins, 1, 1))
        self.estimation = self.separate(X)

        self.base = np.random.rand(n_sources, n_bins, n_bases)
        self.activation = np.random.rand(n_sources, n_bases, n_frames)
    
    def __call__(self, input, iteration=100, **kwargs):
        """
        Args:
            input (n_channels, n_bins, n_frames)
        Returns:
            output (n_channels, n_bins, n_frames)
        """
        self.input = input

        self._reset(**kwargs)

        loss = self.compute_negative_loglikelihood()    
        self.loss.append(loss)

        for idx in range(iteration):
            self.update_once()

            loss = self.compute_negative_loglikelihood()
            self.loss.append(loss)

            if self.callback is not None:
                self.callback(self)
        
        X = input
        output = self.separate(X)

        return output
        
    def update_once(self):
        raise NotImplementedError("Implement 'update_once' function")
    
    def separate(self, input):
        """
        Args:
            input (n_channels, n_bins, n_frames):
        Returns:
            output (n_channels, n_bins, n_frames): 
        """

        return 0
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")

class GaussMNMF(MNMFbase):
    """
    References:
        Sawada's MNMF: "Multichannel Extensions of Non-Negative Matrix Factorization With Complex-Valued Data"
        Ozerov's MNMF: "Multichannel Nonnegative Matrix Factorization in Convolutive Mixtures for Audio Source Separation"
    See https://ieeexplore.ieee.org/document/6410389 and https://ieeexplore.ieee.org/document/5229304
    """
    def __init__(self, n_bases=10, n_sources=None, partitioning=False, normalize=True, reference_id=0, callback=None, author='Sawada', eps=EPS):
        """
        Args:
        """
        super().__init__(n_bases=n_bases, n_sources=n_sources, callback=callback, eps=eps)

        self.partitioning = partitioning
        self.normalize = normalize
        self.reference_id = reference_id

        assert author.lower() in __authors__, "Choose from {}".format(__authors__)

        self.author = author
    
    def __call__(self, input, iteration=100, **kwargs):
        """
        Args:
            input (n_channels, n_bins, n_frames)
        Returns:
            output (n_channels, n_bins, n_frames)
        """
        self.input = input

        self._reset(**kwargs)

        loss = self.compute_negative_loglikelihood()    
        self.loss.append(loss)

        for idx in range(iteration):
            self.update_once()

            loss = self.compute_negative_loglikelihood()
            self.loss.append(loss)

            if self.callback is not None:
                self.callback(self)
        
        reference_id = self.reference_id
        X = self.input
        output = self.separate(X)
        self.estimation = output

        return output
    
    def _reset(self, **kwargs):
        assert self.input is not None, "Specify data!"

        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        n_bases = self.n_bases
        n_sources = self.n_sources

        X = self.input
        n_channels, n_bins, n_frames = X.shape

        if n_sources is None:
            n_sources = n_channels
        self.n_sources, self.n_channels = n_channels, n_sources
        self.n_bins, self.n_frames = n_bins, n_frames

        H = np.eye(n_channels, dtype=np.complex128)
        self.spatial = np.tile(H, reps=(n_bins, n_bases, 1, 1))
        XX = X[:, np.newaxis, :, :] * X[np.newaxis, :, :, :].conj()
        self.covariance_input = XX.transpose(2, 3, 0, 1)
        
        self.base = np.random.rand(n_bins, n_bases)
        self.activation = np.random.rand(n_bases, n_frames)
        self.estimation = self.separate(X)
    
    def _parallel_sort(self, x, indices):
        """
        Args:
            x: (n_bins, n_channels, n_dims, n_eigens)
            indices: (n_bins, n_channels, n_eigens)
        Returnes:
            x: (n_bins, n_channels, n_dims, n_eigens)
        """
        n_bins, n_channels, n_dims, n_eigens = x.shape
        x = x.transpose(0, 1, 3, 2)
        x = x.reshape(n_bins*n_channels*n_eigens, n_dims)
        indices = indices.reshape(n_bins*n_channels, n_eigens)
        shift = np.arange(n_bins*n_channels) * n_eigens
        indices = indices + shift[:, np.newaxis]
        x = x[indices]
        x = x.reshape(n_bins, n_channels, n_eigens, n_dims)
        x = x.transpose(0, 1, 3, 2)

        return x
    
    def separate(self, input):
        """
        Args:
            input (n_channels, n_bins, n_frames):
        Returns:
            output (n_channels, n_bins, n_frames): 
        """
        H, T, V = self.spatial, self.base, self.activation # (n_bins, n_bases, n_channels, n_channels), (n_bins, n_bases), (n_bases, n_frames)

        TV = T[:, :, np.newaxis] * V[np.newaxis, :, :] # (n_bins, n_bases, n_frames)
        HTV = np.sum(H[:, :, np.newaxis, :, :] * TV[:, :, :, np.newaxis, np.newaxis], axis=1) # (n_bins, n_frames, n_channels, n_channels)
        HTV = np.diagonal(HTV, axis1=-2, axis2=-1) # (n_bins, n_frames, n_channels)
        Y = HTV / HTV.sum(axis=-1, keepdims=True)
        output = Y.transpose(2, 0, 1)
        
        return output
    
    def update_once(self):
        n_channels = self.n_channels
        X = self.covariance_input # (n_bins, n_frames, n_channels, n_channels)
        H, T, V = self.spatial, self.base, self.activation # (n_bins, n_bases, n_channels, n_channels), (n_bins, n_bases), (n_bases, n_frames)

        # Update bases
        TV = T[:, :, np.newaxis] * V[np.newaxis, :, :] # (n_bins, n_bases, n_frames)
        X_hat = np.sum(H[:, :, np.newaxis, :, :] * TV[:, :, :, np.newaxis, np.newaxis], axis=1) # (n_bins, n_frames, n_channels, n_channels)
        inv_X_hat = np.linalg.inv(X_hat) # (n_bins, n_frames, n_channels, n_channels)

        XXX = inv_X_hat @ X @ inv_X_hat # (n_bins, n_frames, n_channels, n_channels)
        trace_enumerator = np.trace(XXX[:, np.newaxis, :, :, :] @ H[:, :, np.newaxis, :, :], axis1=-2, axis2=-1) # (n_bins, 1, n_frames, n_channels, n_channels), (n_bins, n_bases, 1, n_channels, n_channels) -> (n_bins, n_bases, n_frames)
        enumerator = np.sum(V[np.newaxis, :, :] * trace_enumerator, axis=2) # (n_bins, n_bases)
        trace_denominator = np.trace(inv_X_hat[:, np.newaxis, :, :, :] @ H[:, :, np.newaxis, :, :], axis1=-2, axis2=-1) # (n_bins, 1, n_frames, n_channels, n_channels), (n_bins, n_bases, 1, n_channels, n_channels) -> (n_bins, n_bases, n_frames)
        denominator = np.sum(V[np.newaxis, :, :] * trace_denominator, axis=2) # (n_bins, n_bases)
        T = T * np.sqrt(enumerator / denominator)

        # Update activations
        TV = T[:, :,np.newaxis] * V[np.newaxis, :, :] # (n_bins, n_bases, n_frames)
        X_hat = np.sum(H[:, :, np.newaxis, :, :] * TV[:, :, :, np.newaxis, np.newaxis], axis=1) # (n_bins, n_frames, n_channels, n_channels)
        inv_X_hat = np.linalg.inv(X_hat) # (n_bins, n_frames, n_channels, n_channels)
        XXX = inv_X_hat @ X @ inv_X_hat # (n_bins, n_frames, n_channels, n_channels)
        trace_enumerator = np.trace(XXX[:, np.newaxis, :, :, :] @ H[:, :, np.newaxis, :, :], axis1=-2, axis2=-1) # (n_bins, 1, n_frames, n_channels, n_channels), (n_bins, n_bases, 1, n_channels, n_channels) -> (n_bins, n_bases, n_frames)
        enumerator = np.sum(T[:, :, np.newaxis]  * trace_enumerator, axis=0) # (n_frames, n_bases)
        trace_denominator = np.trace(inv_X_hat[:, np.newaxis, :, :, :] @ H[:, :, np.newaxis, :, :], axis1=-2, axis2=-1) # (n_bins, 1, n_frames, n_channels, n_channels), (n_bins, n_bases, 1, n_channels, n_channels) -> (n_bins, n_bases, n_frames)
        denominator = np.sum(T[:, :, np.newaxis] * trace_denominator, axis=0) # (n_frames, n_bases)
        V = V * np.sqrt(enumerator / denominator)

        # Update spatial
        TV = T[:, :,np.newaxis] * V[np.newaxis, :, :] # (n_bins, n_bases, n_frames)
        X_hat = np.sum(H[:, :, np.newaxis, :, :] * TV[:, :, :, np.newaxis, np.newaxis], axis=1) # (n_bins, n_frames, n_channels, n_channels)
        inv_X_hat = np.linalg.inv(X_hat) # (n_bins, n_frames, n_channels, n_channels)
        XXX = inv_X_hat @ X @ inv_X_hat # (n_bins, n_frames, n_channels, n_channels)
        VXXX = np.sum(V[np.newaxis, :, :, np.newaxis, np.newaxis] * XXX[:, np.newaxis, :, :, :], axis=2) # (n_bins, n_bases, n_channels, n_channels)

        # H: (n_bins, n_bases, n_channels, n_channels)
        A = np.sum(V[np.newaxis, :, :, np.newaxis, np.newaxis] * inv_X_hat[:, np.newaxis, :, :, :], axis=2)
        B = H @ VXXX @ H
        O = np.zeros_like(A)
        L = np.block([[O, -A], [-B, O]]) # (n_bins, n_bases, n_channels, n_channels)
        w, v = np.linalg.eig(L)
        w = np.real(w)
        indices = np.argsort(w, axis=2)
        v = self._parallel_sort(v, indices=indices)
        e = v[...,:n_channels]
        F, G = np.split(e, n_channels, axis=2)
        H = G @ np.linalg.inv(F)
        H = (H + H.transpose(0, 1, 3, 2).conj()) / 2

        if self.normalize:
            H = H / np.trace(H, axis1=2, axis2=3)[..., np.newaxis, np.newaxis]
        
        self.spatial, self.base, self.activation = H, T, V

    def compute_negative_loglikelihood(self):
        X = self.covariance_input # (n_bins, n_frames, n_channels, n_channels)
        H, T, V = self.spatial, self.base, self.activation # (n_bins, n_bases, n_channels, n_channels), (n_bins, n_bases), (n_bases, n_frames)

        TV = T[:, :, np.newaxis] * V[np.newaxis, :, :] # (n_bins, n_bases, n_frames)
        X_hat = np.sum(H[:, :, np.newaxis, :, :] * TV[:, :, :, np.newaxis, np.newaxis], axis=1) # (n_bins, n_frames, n_channels, n_channels)
        loss = is_divergence(X_hat, X)
        loss = loss.sum()
        return loss

class tMNMF(MNMFbase):
    """
    Reference: "Student's t multichannel nonnegative matrix factorization for blind source separation"
    See https://ieeexplore.ieee.org/document/7602889
    """
    def __init__(self, n_bases=10, n_sources=None, reference_id=0, callback=None, eps=EPS):
        """
        Args:
        """
        super().__init__(n_bases=n_bases, n_sources=n_sources, callback=callback, eps=eps)

        self.reference_id = reference_id
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")

class FastGaussMNMF(MNMFbase):
    """
    Reference: "Fast Multichannel Source Separation Based on Jointly Diagonalizable Spatial Covariance Matrices"
    """
    def __init__(self, n_bases=10, n_sources=None, reference_id=0, callback=None, eps=EPS):
        """
        Args:
        """
        super().__init__(n_bases=n_bases, n_sources=n_sources, callback=callback, eps=eps)

        self.reference_id = reference_id
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' method.")

def is_divergence(input, target, eps=EPS):
    """
    Multichannel Itakura-Saito divergence
    Args:
        input (*, n_channels, n_channels)
        target (*, n_channels, n_channels)
    """
    shape_input, shape_target = input.shape, target.shape
    assert shape_input[-2] == shape_input[-1] and shape_target[-2] == shape_target[-1], "Invalid input shape"
    n_channels = shape_input[-1]

    target, input = input + eps * np.eye(n_channels), target + eps * np.eye(n_channels)
    XX = target @ np.linalg.inv(input)
    loss = np.trace(XX, axis1=-2, axis2=-1) - np.log(np.linalg.det(XX)) - n_channels
    loss = np.real(loss)

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

def _test(method, n_bases=10, domain=2, partitioning=False):
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

    # MNMF
    n_channels = len(titles)
    iteration = 50

    if method == 'Gauss':
        mnmf = GaussMNMF(n_bases=n_bases, partitioning=partitioning)
    else:
        raise ValueError("Not support {}-MNMF.".format(method))
    estimation = mnmf(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_channels):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/MNMF/{}MNMF/partitioning{}/mixture-{}_estimated-iter{}-{}.wav".format(method, int(partitioning), sr, iteration, idx), signal=_estimated_signal, sr=sr)
    
    plt.figure()
    plt.plot(mnmf.loss[1:], color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/MNMF/{}MNMF/partitioning{}/loss.png'.format(method, int(partitioning)), bbox_inches='tight')
    plt.close()

def _test_ilrma(partitioning=False):
    from bss.ilrma import GaussILRMA
    np.random.seed(111)
    
    # Room impulse response
    sr = 16000
    reverb = 0.16
    duration = 0.5
    samples = int(duration * sr)
    mic_intervals = [8, 8, 8, 8, 8, 8, 8]
    mic_indices = [2, 5]
    degrees = [60, 300]
    titles = ['sample-song/sample2_source1', 'sample-song/sample2_source2']

    mixed_signal = _convolve_mird(titles, reverb=reverb, degrees=degrees, mic_intervals=mic_intervals, mic_indices=mic_indices, samples=samples)
    write_wav("data/MNMF/GaussILRMA/partitioning{}/mixture.wav".format(int(partitioning)), signal=mixed_signal.T, sr=sr)

    n_sources, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 2048, 1024
    mixture = stft(mixed_signal, fft_size=fft_size, hop_size=hop_size)
    power_spectrogram = np.abs(mixture)**2
    spectrogram = 10 * np.log10(power_spectrogram + 1e-12)
    _, f, t = spectrogram.shape
    t, f = np.arange(t), np.arange(f)

    plt.figure()
    plt.pcolormesh(t, f, spectrogram[0])
    plt.savefig("data/MNMF/GaussILRMA/partitioning{}/mixture-spectrogram.png".format(int(partitioning)), bbox_inches='tight')
    plt.close()

    # ILRMA
    n_channels = len(titles)
    iteration = 50
    if partitioning:
        n_bases = 5
    else:
        n_bases = 2

    ilrma = GaussILRMA(n_bases=n_bases, partitioning=partitioning)
    estimation = ilrma(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_channels):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/MNMF/GaussILRMA/partitioning{}/mixture-{}_estimated-iter{}-{}.wav".format(int(partitioning), sr, iteration, idx), signal=_estimated_signal, sr=sr)

        power_spectrogram = np.abs(estimation[idx])**2
        spectrogram = 10 * np.log10(power_spectrogram + 1e-12)
        f, t = spectrogram.shape
        t, f = np.arange(t), np.arange(f)

        plt.figure()
        plt.pcolormesh(t, f, spectrogram)
        plt.savefig("data/MNMF/GaussILRMA/partitioning{}/mixture-{}_estimated-iter{}-{}.png".format(int(partitioning), sr, iteration, idx), bbox_inches='tight')
        plt.close()
    
    plt.figure()
    plt.plot(ilrma.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/MNMF/GaussILRMA/partitioning{}/loss.png'.format(int(partitioning)), bbox_inches='tight')
    plt.close()


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


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.io import loadmat

    from utils.utils_audio import read_wav, write_wav

    plt.rcParams['figure.dpi'] = 200

    os.makedirs("data/multi-channel", exist_ok=True)
    os.makedirs("data/MNMF/GaussMNMF/partitioning0", exist_ok=True)
    os.makedirs("data/MNMF/GaussILRMA/partitioning0", exist_ok=True)
    os.makedirs("data/MNMF/GaussILRMA/partitioning1", exist_ok=True)

    """
    Use multichannel room impulse response database.
    Download database from "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/"
    """

    _test_conv()
    # _test(method='Gauss', n_bases=2, partitioning=False)
    _test_ilrma(partitioning=False)
    _test_ilrma(partitioning=True)