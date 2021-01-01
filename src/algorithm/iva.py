import numpy as np

from algorithm.projection_back import projection_back

EPS=1e-12

class IVAbase:
    def __init__(self, eps=EPS):
        self.input = None
        self.loss = []

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
    
    def compute_negative_loglikelihood(self):
        Y = self.estimation
        W = self.demix_filter

        P = np.sum(np.abs(Y)**2, axis=1)
        loss = 2 * np.sum(np.sqrt(P), axis=0).mean() - 2 * np.log(np.abs(np.linalg.det(W))).sum()

        return loss

class GradIVA(IVAbase):
    def __init__(self, distr='laplace', lr=1e-1, reference_id=0, eps=EPS):
        super().__init__(eps=eps)

        self.distr = distr
        self.lr = lr
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

        loss = self.compute_negative_loglikelihood()
        self.loss.append(loss)

        for idx in range(iteration):
            self.update_once()
            loss = self.compute_negative_loglikelihood()
            self.loss.append(loss)

        reference_id = self.reference_id
        X, W = input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        Y_hat = Y * scale[...,np.newaxis].conj() # (n_sources, n_bins, n_frames)

        Y_hat = Y_hat.transpose(1,0,2) # (n_bins, n_sources, n_frames)
        X = X.transpose(1,0,2) # (n_bins, n_channels, n_frames)
        X_Hermite = X.transpose(0,2,1).conj() # (n_bins, n_frames, n_channels)
        XX_inverse = np.linalg.inv(X @ X_Hermite)
        self.demix_filter = Y_hat @ X_Hermite @ XX_inverse

        X, W = input, self.demix_filter
        output = self.separate(X, demix_filter=W)

        self.estimation = output

        return output
    
    def update_once(self):
        reference_id = self.reference_id

        if self.distr == 'laplace':
            self.update_laplace()
        else:
            raise NotImplementedError("Cannot support {} distribution.".format(self.distr))

        X = self.input
        W = self.demix_filter
        Y = self.separate(X, demix_filter=W)

        self.estimation = Y
    
    def update_laplace(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        n_frames = self.n_frames
        lr = self.lr
        eps = self.eps

        X = self.input
        W = self.demix_filter
        Y = self.separate(X, demix_filter=W)

        X_Hermite = X.transpose(1,2,0).conj() # (n_bins, n_frames, n_sources)
        W_inverse = np.linalg.inv(W)
        W_inverseHermite = W_inverse.transpose(0,2,1).conj() # (n_bins, n_channels, n_sources)

        Y = Y.transpose(1,0,2) # (n_bins, n_sources, n_frames)
        P = np.abs(Y)**2
        denominator = np.sqrt(P.sum(axis=0))
        denominator[denominator < eps] = eps
        Phi = Y / denominator # (n_bins, n_sources, n_frames)

        delta = (Phi @ X_Hermite) / n_frames - W_inverseHermite
        W = W - lr * delta # (n_bins, n_sources, n_channels)

        self.demix_filter = W

class NaturalGradIVA(GradIVA):
    def __init__(self, distr='laplace', lr=1e-1, reference_id=0, eps=EPS):
        super().__init__(distr=distr, lr=lr, reference_id=reference_id, eps=eps)

        self.distr = distr
        self.lr = lr
        self.reference_id = reference_id

    def update_laplace(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        n_frames = self.n_frames
        lr = self.lr
        eps = self.eps

        X = self.input
        W = self.demix_filter
        Y = self.separate(X, demix_filter=W)
        eye = np.eye(n_sources, n_channels, dtype=np.complex128)

        Y = Y.transpose(1,0,2) # (n_bins, n_sources, n_frames)
        Y_Hermite = Y.transpose(0,2,1).conj() # (n_bins, n_frames, n_sources)
        P = np.abs(Y)**2
        denominator = np.sqrt(P.sum(axis=0))
        denominator[denominator < eps] = eps
        Phi = Y / denominator # (n_bins, n_sources, n_frames)

        delta = ((Phi @ Y_Hermite) / n_frames - eye) @ W
        W = W - lr * delta # (n_bins, n_sources, n_channels)

        self.demix_filter = W


class AuxIVA(IVAbase):
    def __init__(self, eps=EPS):
        super().__init__(eps=eps)

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

def _test(method='AuxIVA'):
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
    lr = 0.1
    n_sources = len(titles)
    iteration = 200

    if method == 'GradIVA':
        iva = GradIVA(lr=lr)
        iteration = 5000
    elif method == 'NaturalGradIVA':
        iva = NaturalGradIVA(lr=lr)
        iteration = 200
    elif method == 'AuxIVA':
        iva = AuxIVA()
        iteration = 200
    else:
        raise ValueError("Not support method {}".format(method))

    estimation = iva(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_sources):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/IVA/{}/mixture-{}_estimated-iter{}-{}.wav".format(method, sr, iteration, idx), signal=_estimated_signal, sr=sr)
    
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
    from algorithm.stft import stft, istft

    plt.rcParams['figure.dpi'] = 200

    os.makedirs("data/multi-channel", exist_ok=True)
    os.makedirs("data/IVA/GradIVA", exist_ok=True)
    os.makedirs("data/iVA/NaturalGradIVA", exist_ok=True)
    os.makedirs("data/iVA/AuxIVA", exist_ok=True)


    """
    Use multichannel room impulse response database.
    Download database from "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/"
    """

    # _test_conv()
    _test(method='GradIVA')
    _test(method='NaturalGradIVA')
    #_test(method='IVA')