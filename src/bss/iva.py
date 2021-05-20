import numpy as np
import scipy.sparse as sci_sparse

from bss.prox import PDSBSSbase
from algorithm.projection_back import projection_back

EPS=1e-12
THRESHOLD=1e+12

class IVAbase:
    def __init__(self, callback=None, eps=EPS):
        self.callback = callback
        self.eps = eps

        self.input = None
        self.loss = []
    
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
            self.demix_filter = np.tile(W, reps=(n_bins, 1, 1))
        else:
            W = self.demix_filter
        self.estimation = self.separate(X, demix_filter=W)
        
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
        
        X, W = input, self.demix_filter
        output = self.separate(X, demix_filter=W)

        return output

    def update_once(self):
        raise NotImplementedError("Implement 'update_once' function")
    
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
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' function.")

class GradIVAbase(IVAbase):
    """
    Reference: "Independent Vector Analysis: Definition and Algorithms"
    See https://ieeexplore.ieee.org/document/4176796
    """
    def __init__(self, lr=1e-1, reference_id=0, callback=None, eps=EPS):
        super().__init__(callback=callback, eps=eps)

        self.lr = lr
        self.reference_id = reference_id
    
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
        X, W = input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def update_once(self):
        raise NotImplementedError("Implement 'update_once' function")

    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' function.")

class GradLaplaceIVA(GradIVAbase):
    def __init__(self, lr=1e-1, reference_id=0, callback=None, eps=EPS):
        super().__init__(callback=callback, eps=eps)

        self.lr = lr
        self.reference_id = reference_id
    
    def update_once(self):
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
        
        X = self.input
        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.estimation = Y
    
    def compute_negative_loglikelihood(self):
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        P = np.sum(np.abs(Y)**2, axis=1)
        loss = 2 * np.sum(np.sqrt(P), axis=0).mean() - 2 * np.log(np.abs(np.linalg.det(W))).sum()

        return loss


class NaturalGradLaplaceIVA(GradIVAbase):
    def __init__(self, lr=1e-1, reference_id=0, callback=None, eps=EPS):
        super().__init__(lr=lr, reference_id=reference_id, callback=callback, eps=eps)

        self.lr = lr
        self.reference_id = reference_id

    def update_once(self):
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
        
        X = self.input
        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.estimation = Y
    
    def compute_negative_loglikelihood(self):
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        P = np.sum(np.abs(Y)**2, axis=1)
        loss = 2 * np.sum(np.sqrt(P), axis=0).mean() - 2 * np.log(np.abs(np.linalg.det(W))).sum()

        return loss


class AuxIVAbase(IVAbase):
    def __init__(self, reference_id=0, callback=None, eps=EPS, threshold=THRESHOLD):
        super().__init__(callback=callback, eps=eps)

        self.reference_id = reference_id
        self.threshold = threshold
    
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
        X, W = input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def update_once(self):
        raise NotImplementedError("Implement 'update_once' function.")

    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' function.")


class AuxLaplaceIVA(AuxIVAbase):
    def __init__(self, reference_id=0, callback=None, eps=EPS, threshold=THRESHOLD):
        super().__init__(reference_id=reference_id, callback=callback, eps=eps, threshold=threshold)
    
    def update_once(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins
        eps, threshold = self.eps, self.threshold

        X, W = self.input, self.demix_filter
        Y = self.estimation
        
        X = X.transpose(1,2,0) # (n_bins, n_frames, n_channels)
        X = X[...,np.newaxis]
        X_Hermite = X.transpose(0,1,3,2).conj()
        XX = X @ X_Hermite # (n_bins, n_frames, n_channels, n_channels)
        P = np.abs(Y)**2 # (n_sources, n_bins, n_frames)
        R = np.sqrt(P.sum(axis=1))[:,np.newaxis,:,np.newaxis,np.newaxis] # (n_sources, 1, n_frames, 1, 1)
        R[R < eps] = eps
        U = XX / R # (n_sources, n_bins, n_frames, n_channels, n_channels)
        U = U.mean(axis=2) # (n_sources, n_bins, n_channels, n_channels)
        E = np.eye(n_sources, n_channels)
        E = np.tile(E, reps=(n_bins,1,1)) # (n_bins, n_sources, n_channels)

        for source_idx in range(n_sources):
            # W: (n_bins, n_sources, n_channels), U: (n_sources, n_bins, n_channels, n_channels)
            w_n_Hermite = W[:,source_idx,:] # (n_bins, n_channels)
            U_n = U[source_idx] # (n_bins, n_channels, n_channels)
            WU = W @ U_n # (n_bins, n_sources, n_channels)
            condition = np.linalg.cond(WU) < threshold # (n_bins,)
            condition = condition[:,np.newaxis] # (n_bins, 1)
            e_n = E[:,source_idx,:]
            w_n = np.linalg.solve(WU, e_n)
            wUw = w_n[:,np.newaxis,:].conj() @ U_n @ w_n[:,:,np.newaxis]
            denominator = np.sqrt(wUw[...,0])
            w_n_Hermite = np.where(condition, w_n.conj() / denominator, w_n_Hermite)
            # if condition number is too big, `denominator[denominator < eps] = eps` may occur divergence of cost function.
            W[:,source_idx,:] = w_n_Hermite
        
        self.demix_filter = W

        X = self.input
        Y = self.separate(X, demix_filter=W)
        
        self.estimation = Y
    
    def compute_negative_loglikelihood(self):
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        P = np.sum(np.abs(Y)**2, axis=1)
        loss = 2 * np.sum(np.sqrt(P), axis=0).mean() - 2 * np.log(np.abs(np.linalg.det(W))).sum()

        return loss

class AuxGaussIVA(AuxIVAbase):
    def __init__(self, reference_id=0, callback=None, eps=EPS, threshold=THRESHOLD):
        super().__init__(reference_id=reference_id, callback=callback, eps=eps, threshold=threshold)

        raise NotImplementedError("in progress")
    
    def update_once(self):
        raise NotImplementedError("in progress...")

class SparseAuxIVA(AuxIVAbase):
    """
    Reference: "A computationally cheaper method for blind speech separation based on AuxIVA and incomplete demixing transform"
    See https://ieeexplore.ieee.org/document/7602921
    """
    def __init__(self, reference_id=0, callback=None, eps=EPS, threshold=THRESHOLD):
        super().__init__(reference_id=reference_id, callback=callback, eps=eps, threshold=threshold)

        raise NotImplementedError("in progress")
    
    def update_once(self):
        raise NotImplementedError("in progress...")

class ProxIVAbase(IVAbase):
    def __init__(self, regularizer=1, step_prox_logdet=1e+0, step_prox_penalty=1e+0, step=1e+0, reference_id=0, callback=None, eps=EPS):
        """
        Args:
            regularizer <float>: Coefficient of source model penalty
            step_prox_logdet <float>: step size parameter referenced `mu1` in "Determined Blind Source Separation via Proximal Splitting Algorithm"
            step_prox_penalty <float>: step size parameter referenced `mu2` in "Determined Blind Source Separation via Proximal Splitting Algorithm"
        """
        super().__init__(callback=callback, eps=eps)

        self.regularizer = regularizer
        self.step_prox_logdet, self.step_prox_penalty = step_prox_logdet, step_prox_penalty
        self.step = step
        self.reference_id = reference_id
    
    def _reset(self, **kwargs):
        super()._reset(**kwargs)

        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins, n_frames = self.n_bins, self.n_frames

        X, W = self.input, self.demix_filter
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
    
    def __call__(self, input, iteration, **kwargs):
        """
        Args:
            input (n_channels, n_bins, n_frames)
            iteration <int>
        Returns:
            output (n_channels, n_bins, n_frames)
        """
        Y = super().__call__(input, iteration=iteration, **kwargs)

        reference_id = self.reference_id
        X, W = input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output

    def update_once(self):
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins = self.n_bins

        mu1, mu2 = self.step_prox_logdet, self.step_prox_penalty
        alpha = self.step

        X, w = self.input_sparse, self.demix_filter_sparse
        y = self.y

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
    
    def prox_logdet(self, demix_filter, mu=1, is_sparse=True):
        """
        Args:
            demix_filter (n_sources * n_channels * n_bins) when `is_sparse` is True, or (n_bins, n_sources, n_channels)
            mu <float>: 
        Returns:
            demix_filter (n_sources * n_channels * n_bins) when `is_sparse` is True, or (n_bins, n_sources, n_channels)
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
    
    def compute_negative_loglikelihood(self):
        loss = self.compute_penalty() + self.compute_negative_logdet()

        return loss
    
    def compute_penalty(self):
        raise NotImplementedError("Implement `compute_penalty` method")
    
    def compute_negative_logdet(self):
        W = self.demix_filter
        loss = - np.log(np.abs(np.linalg.det(W)))
        loss = loss.sum()

        return loss

class ProxLaplaceIVA(PDSBSSbase):
    def __init__(self, regularizer=1, step_prox_logdet=1e+0, step_prox_penalty=1e+0, step=1e+0, reference_id=0, callback=None, eps=EPS):
        super().__init__(regularizer=regularizer, step_prox_logdet=step_prox_logdet, step_prox_penalty=step_prox_penalty, step=step, callback=callback, eps=eps)

        self.reference_id = reference_id
    
    def __call__(self, input, iteration, **kwargs):
        """
        Args:
            input (n_channels, n_bins, n_frames)
            iteration <int>
        Returns:
            output (n_channels, n_bins, n_frames)
        """
        Y = super().__call__(input, iteration=iteration, **kwargs)

        reference_id = self.reference_id
        X, W = input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def prox_penalty(self, z, mu=1, is_sparse=True):
        """
        Args:
            z (n_bins * n_sources * n_frames, 1) <scipy.sparse.lil_matrix>
            mu <float>: 
        Returns:
            z_tilde (n_bins * n_sources * n_frames, 1) <scipy.sparse.lil_matrix>
        """
        n_sources, n_channels = self.n_sources, self.n_channels
        n_bins, n_frames = self.n_bins, self.n_frames
        C = self.regularizer

        assert is_sparse, "`is_sparse` is expected True."

        z = z.toarray().reshape(n_bins, n_sources, n_frames) # (n_bins, n_sources, n_frames)
        zsum = np.sum(np.abs(z)**2, axis=0)
        denominator = np.sqrt(zsum) # (n_sources, n_frames)
        denominator = np.where(denominator <= 0, mu, denominator)
        z_tilde = C * np.maximum(0, 1 - mu / denominator) * z # TODO: correct?
        z_tilde = z_tilde.reshape(n_bins * n_sources * n_frames, 1)
        z_tilde = sci_sparse.lil_matrix(z_tilde)

        return z_tilde
    
    def compute_penalty(self):
        """
        Returns:
            loss <float>
        """
        C = self.regularizer
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)
        
        loss = np.sum(np.abs(Y)**2, axis=1) # (n_sources, n_frames)
        loss = np.sqrt(loss) # (n_sources, n_frames)
        loss = C * loss.sum(axis=(0, 1))
        
        return loss

class SparseProxIVA(ProxIVAbase):
    """
    Reference: "Time-frequency-masking-based Determined BSS with Application to Sparse IVA"
    See https://ieeexplore.ieee.org/document/8682217
    """
    def __init__(self, callback=None, eps=EPS):
        super().__init__(callback=callback, eps=eps)

        raise NotImplementedError("coming soon")


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

def _test(method='AuxLaplaceIVA'):
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

    if method == 'GradLaplaceIVA':
        lr = 0.1
        iva = GradLaplaceIVA(lr=lr)
        iteration = 5000
    elif method == 'NaturalGradLaplaceIVA':
        lr = 0.1
        iva = NaturalGradLaplaceIVA(lr=lr)
        iteration = 200
    elif method == 'AuxLaplaceIVA':
        iva = AuxLaplaceIVA()
        iteration = 50
    elif method == 'ProxLaplaceIVA':
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
    plt.plot(iva.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/IVA/{}/loss.png'.format(method), bbox_inches='tight')
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
    from scipy.io import loadmat

    from utils.utils_audio import read_wav, write_wav
    from transform.stft import stft, istft
    from transform.whitening import whitening

    plt.rcParams['figure.dpi'] = 200

    os.makedirs("data/multi-channel", exist_ok=True)
    os.makedirs("data/IVA/GradLaplaceIVA", exist_ok=True)
    os.makedirs("data/IVA/NaturalGradLaplaceIVA", exist_ok=True)
    os.makedirs("data/IVA/AuxLaplaceIVA", exist_ok=True)
    os.makedirs("data/IVA/ProxLaplaceIVA", exist_ok=True)

    """
    Use multichannel room impulse response database.
    Download database from "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/"
    """

    _test_conv()
    _test(method='GradLaplaceIVA')
    _test(method='NaturalGradLaplaceIVA')
    _test(method='AuxLaplaceIVA')
    _test(method='ProxLaplaceIVA')