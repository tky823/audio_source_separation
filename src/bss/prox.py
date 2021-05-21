import numpy as np
import scipy.sparse as sci_sparse

from algorithm.projection_back import projection_back

EPS=1e-12

"""
    Reference: "Determined Blind Source Separation via Proximal Splitting Algorithm"
    See https://ieeexplore.ieee.org/document/8462338
"""

class PDSBSSbase:
    """
        Blind source separation using Primal-dual splitting algorithm
    """
    def __init__(self, regularizer=1, step_prox_logdet=1e+0, step_prox_penalty=1e+0, step=1e+0, callbacks=None, recordable_loss=True, eps=EPS):
        """
        Args:
            regularizer <float>: Coefficient of source model penalty
            step_prox_logdet <float>: step size parameter referenced `mu1` in "Determined Blind Source Separation via Proximal Splitting Algorithm"
            step_prox_penalty <float>: step size parameter referenced `mu2` in "Determined Blind Source Separation via Proximal Splitting Algorithm"
        """
        self.regularizer = regularizer
        self.step_prox_logdet, self.step_prox_penalty = step_prox_logdet, step_prox_penalty
        self.step = step

        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None
        self.eps = eps

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

        X = self.input

        n_channels, n_bins, n_frames = X.shape
        n_sources = n_channels # n_channels == n_sources

        self.n_sources, self.n_channels = n_sources, n_channels
        self.n_bins, self.n_frames = n_bins, n_frames

        if not hasattr(self, 'demix_filter'):
            W = np.eye(n_sources, n_channels, dtype=np.complex128)
            W = np.tile(W, reps=(n_bins, 1, 1))
            self.demix_filter = W
        else:
            W = self.demix_filter

        self.estimation = self.separate(X, demix_filter=W)

        # Vectorize
        X = X.transpose(1, 2, 0).reshape(n_bins, n_frames, n_channels) # (n_channels, n_bins, n_frames) -> (n_bins, n_frames, n_channels)
        
        FNT, FNM = n_bins * n_sources * n_frames, n_bins * n_sources * n_channels
        X = np.tile(X[:, np.newaxis, :, :], reps=(1, n_sources, 1, 1)).reshape(n_bins * n_sources, n_frames, n_channels)
        indptr = np.arange(n_bins * n_sources + 1)
        indices = np.arange(n_bins * n_sources)
        X = sci_sparse.bsr_matrix((X, indices, indptr), shape=(FNT, FNM))
        _, [norm], _ = sci_sparse.linalg.svds(X, k=1, which='LM') # Largest

        self.input_sparse = X / norm
        w = W.reshape(n_bins * n_sources * n_channels, 1) # (n_bins, n_sources, n_channels) -> (n_bins * n_sources * n_channels, 1)
        self.demix_filter_sparse = sci_sparse.lil_matrix(w)
        self.y = sci_sparse.lil_matrix((FNT, 1), dtype=np.complex128)
        
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

            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self)
        
        X, W = input, self.demix_filter
        output = self.separate(X, demix_filter=W)

        return output
    
    def update_once(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins

        mu1, mu2 = self.step_prox_logdet, self.step_prox_penalty
        alpha = self.step

        X, w = self.input_sparse, self.demix_filter_sparse
        y = self.y
        # w: (n_bins * n_sources * n_channels, 1)
        # X: (n_bins * n_sources * n_frames, n_bins * n_sources * n_channels)
        w_tilde = self.prox_logdet(w - mu1 * mu2 * X.T.conj() @ y, mu1) # update demix_filter
        z = y + X @ (2 * w_tilde - w)
        y_tilde = z - self.prox_penalty(z, 1 / mu2) # update demix_filter
        y = alpha * y_tilde + (1 - alpha) * y
        w = alpha * w_tilde + (1 - alpha) * w
        
        X = self.input
        W = w.toarray().reshape(n_bins, n_sources, n_channels) # -> (n_bins, n_sources, n_channels)

        Y = self.separate(X, demix_filter=W)

        self.y = y
        self.demix_filter_sparse = w
        self.demix_filter = W
        self.estimation = Y
    
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
    
    def prox_logdet(self, demix_filter, mu=1, is_sparse=True):
        """
        Args:
            demix_filter (n_bins * n_sources * n_channels, 1) when `is_sparse` is True, or (n_bins, n_sources, n_channels)
            mu <float>: 
        Returns:
            demix_filter (n_bins * n_sources * n_channels, 1) when `is_sparse` is True, or (n_bins, n_sources, n_channels)
        """
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins

        if is_sparse:
            w = demix_filter.toarray()
            W = w.reshape(n_bins, n_sources, n_channels) # -> (n_bins, n_sources, n_channels)
        else:
            W = demix_filter
        U, Sigma, V = np.linalg.svd(W)
        Sigma = (Sigma + np.sqrt(Sigma**2 + 4 * mu)) / 2
        eyes = np.eye(n_sources, n_channels)
        Sigma = eyes * Sigma[:, :, np.newaxis]
        W_tilde = U @ Sigma @ V

        if is_sparse:
            w_tilde = W_tilde.reshape(n_bins * n_sources * n_channels, 1)
            demix_filter = sci_sparse.lil_matrix(w_tilde)
        else:
            demix_filter = W_tilde
        
        return demix_filter
    
    def prox_penalty(self, demix_filter, mu=1):
        raise NotImplementedError("Implement `prox_penalty` method")
    
    def compute_penalty(self):
        """
        Returns:
            loss <float>
        """
        raise NotImplementedError("Implement `compute_penalty` method in subclass")
    
    def compute_negative_loglikelihood(self):
        loss = self.compute_penalty() + self.compute_negative_logdet()

        return loss
    
    def compute_negative_logdet(self):
        W = self.demix_filter
        loss = - np.log(np.abs(np.linalg.det(W)))
        loss = loss.sum()

        return loss

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


def _test(method='ProxIVA'):
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

    n_channels, T = mixed_signal.shape
    
    # STFT
    fft_size, hop_size = 2048, 1024
    mixture = stft(mixed_signal, fft_size=fft_size, hop_size=hop_size)

    # IVA
    n_sources = len(titles)
    iteration = 200

    if method == 'ProxLaplaceIVA':
        step = 1.75
        iva = ProxLaplaceIVA(step=step)
        iteration = 100
    else:
        raise ValueError("Not support method {}".format(method))

    estimation = iva(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_sources):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/IVA/{}/mixture-{}_estimated-iter{}-{}.wav".format(method, sr, iteration, idx), signal=_estimated_signal, sr=sr)
        
        plt.figure()
        plt.plot(_estimated_signal, color='black')
        plt.savefig('data/IVA/{}/wav-{}.png'.format(method, idx), bbox_inches='tight')
        plt.close()

    plt.figure()
    plt.plot(iva.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/IVA/{}/loss.png'.format(method), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    from utils.utils_audio import read_wav, write_wav
    from transform.stft import stft, istft
    from transform.whitening import whitening
    from bss.iva import ProxLaplaceIVA

    plt.rcParams['figure.dpi'] = 200

    os.makedirs("data/multi-channel", exist_ok=True)
    os.makedirs("data/IVA/ProxLaplaceIVA", exist_ok=True)

    """
    Use multichannel room impulse response database.
    Download database from "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/"
    """
    _test(method='ProxLaplaceIVA')