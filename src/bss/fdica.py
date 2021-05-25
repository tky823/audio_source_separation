import numpy as np
import itertools

from algorithm.projection_back import projection_back

EPS=1e-12

class FDICAbase:
    def __init__(self, callbacks=None, recordable_loss=True, eps=EPS):
        if callbacks is not None:
            if callable(callbacks):
                callbacks = [callbacks]
            self.callbacks = callbacks
        else:
            self.callbacks = None
        self.eps = eps

        self.input = None
        self.criterion = None
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
            self.demix_filter = np.tile(W, reps=(n_bins, 1, 1))
        else:
            W = self.demix_filter.copy()
            self.demix_filter = W
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

        return output
    
    def __repr__(self) -> str:
        s = "FDICA("
        s += ")"

        return s.format(**self.__dict__)

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

    def solve_permutation(self):
        n_sources, n_bins, n_frames = self.n_sources, self.n_bins, self.n_frames
        eps = self.eps

        permutations = list(itertools.permutations(range(n_sources)))

        W = self.demix_filter # (n_bins, n_sources, n_chennels)
        Y = self.estimation # (n_sources, n_bins, n_frames)

        P = np.abs(Y).transpose(1,0,2) # (n_bins, n_sources, n_frames)
        norm = np.sqrt(np.sum(P**2, axis=1, keepdims=True))
        norm[norm < eps] = eps
        P = P / norm # (n_bins, n_sources, n_frames)
        correlation = np.sum(P @ P.transpose(0,2,1), axis=(1,2)) # (n_sources,)
        indices = np.argsort(correlation)

        min_idx = indices[0]
        P_criteria = P[min_idx] # (n_sources, n_frames)

        for idx in range(1, n_bins):
            min_idx = indices[idx]
            P_max = None
            perm_max = None
            for perm in permutations:
                P_perm = np.sum(P_criteria * P[min_idx, perm,:])
                if P_max is None or P_perm > P_max:
                    P_max = P_perm
                    perm_max = perm
            
            P_criteria = P_criteria + P[min_idx,perm_max,:]
            W[min_idx,:,:] = W[min_idx,perm_max,:]
        
        self.demix_filter = W
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' function.")


class GradFDICAbase(FDICAbase):
    def __init__(self, lr=1e-1, reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        super().__init__(callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

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
        
        self.solve_permutation()

        reference_id = self.reference_id
        X, W = input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        scale = projection_back(Y, reference=X[reference_id])
        output = Y * scale[...,np.newaxis] # (n_sources, n_bins, n_frames)
        self.estimation = output

        return output
    
    def __repr__(self) -> str:
        s = "GradFDICA("
        s += "lr={lr}"
        s += ")"

        return s.format(**self.__dict__)
    
    def compute_negative_loglikelihood(self):
        raise NotImplementedError("Implement 'compute_negative_loglikelihood' function.")

class GradLaplaceFDICA(GradFDICAbase):
    def __init__(self, lr=1e-1, reference_id=0, callbacks=None, recordable_loss=True, eps=EPS):
        super().__init__(lr=lr, reference_id=reference_id, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)
    
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
        denominator = np.abs(Y)
        denominator[denominator < eps] = eps
        Phi = Y / denominator # (n_bins, n_sources, n_frames)

        delta = (Phi @ X_Hermite) / n_frames - W_inverseHermite
        W = W - lr * delta # (n_bins, n_sources, n_channels)

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.estimation = Y
    
    def __repr__(self) -> str:
        s = "GradLaplaceFDICA("
        s += "lr={lr}"
        s += ")"

        return s.format(**self.__dict__)
    
    def compute_negative_loglikelihood(self):
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        loss = 2 * np.abs(Y).sum(axis=0).mean(axis=1) - 2 * np.log(np.abs(np.linalg.det(W)))
        loss = loss.sum()

        return loss

class NaturalGradLaplaceFDICA(GradFDICAbase):
    def __init__(self, lr=1e-1, reference_id=0, is_holonomic=True, callbacks=None, recordable_loss=True, eps=EPS):
        super().__init__(lr=lr, reference_id=reference_id, callbacks=callbacks, recordable_loss=recordable_loss, eps=eps)

        self.is_holonomic = is_holonomic
    
    def __repr__(self) -> str:
        s = "GradLaplaceFDICA("
        s += "lr={lr}"
        s += ", is_holonomic={is_holonomic}"
        s += ")"

        return s.format(**self.__dict__)

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
        denominator = np.abs(Y)
        denominator[denominator < eps] = eps
        Phi = Y / denominator # (n_bins, n_sources, n_frames)

        if self.is_holonomic:
            delta = ((Phi @ Y_Hermite) / n_frames - eye) @ W
        else:
            raise NotImplementedError("only suports for is_holonomic = True")
            offdiag_mask = 1 - eye
            delta = offdiag_mask * ((Phi @ Y_Hermite) / n_frames) @ W
        
        W = W - lr * delta # (n_bins, n_sources, n_channels)

        Y = self.separate(X, demix_filter=W)

        self.demix_filter = W
        self.estimation = Y
    
    def compute_negative_loglikelihood(self):
        X, W = self.input, self.demix_filter
        Y = self.separate(X, demix_filter=W)

        loss = 2 * np.abs(Y).sum(axis=0).mean(axis=1) - 2 * np.log(np.abs(np.linalg.det(W)))
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


def _test(method='NaturalGradFDICA'):
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

    # FDICA
    lr = 0.1
    n_sources = len(titles)
    iteration = 200

    if method == 'GradLaplaceFDICA':
        fdica = GradLaplaceFDICA(lr=lr)
        iteration = 5000
    elif method == 'NaturalGradLaplaceFDICA':
        fdica = NaturalGradLaplaceFDICA(lr=lr)
        iteration = 200
    else:
        raise ValueError("Not support method {}".format(method))

    estimation = fdica(mixture, iteration=iteration)

    estimated_signal = istft(estimation, fft_size=fft_size, hop_size=hop_size, length=T)
    
    print("Mixture: {}, Estimation: {}".format(mixed_signal.shape, estimated_signal.shape))

    for idx in range(n_sources):
        _estimated_signal = estimated_signal[idx]
        write_wav("data/FDICA/{}/mixture-{}_estimated-iter{}-{}.wav".format(method, sr, iteration, idx), signal=_estimated_signal, sr=sr)
    
    plt.figure()
    plt.plot(fdica.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/FDICA/{}/loss.png'.format(method), bbox_inches='tight')
    plt.close()

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
    from scipy.io import loadmat

    from utils.utils_audio import read_wav, write_wav
    from algorithm.stft import stft, istft

    plt.rcParams['figure.dpi'] = 200

    os.makedirs("data/multi-channel", exist_ok=True)
    os.makedirs("data/FDICA/GradLaplaceFDICA", exist_ok=True)
    os.makedirs("data/FDICA/NaturalGradLaplaceFDICA", exist_ok=True)

    """
    Use multichannel room impulse response database.
    Download database from "https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/multi-channel-impulse-response-database/"
    """

    # _test_conv()
    _test(method='GradLaplaceFDICA')
    _test(method='NaturalGradLaplaceFDICA')