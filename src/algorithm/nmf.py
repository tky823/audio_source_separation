import numpy as np
from criterion.divergence import generalized_kl_divergence, is_divergence

EPS=1e-12

__metrics__ = ['EUC', 'KL', 'IS']

class NMFbase:
    def __init__(self, n_bases=2, eps=EPS):
        """
        Args:
            n_bases: number of bases 
        """

        self.n_bases = n_bases
        self.loss = []

        self.eps = eps
    
    def update(self, target, iteration=100):
        n_bases = self.n_bases
        eps = self.eps

        self.target = target
        F_bin, T_bin = target.shape

        self.base = np.random.rand(F_bin, n_bases)
        self.activation = np.random.rand(n_bases, T_bin)

        for idx in range(iteration):
            self.update_once()

            TV = self.base @ self.activation
            loss = self.criterion(TV, target)
            self.loss.append(loss.sum())
        
    def update_once(self):
        raise NotImplementedError("Implement 'update_once' function")

class ComplexNMFbase:
    def __init__(self, n_bases=2, regularizer=0.1, eps=EPS):
        """
        Args:
            n_bases: number of bases 
        """

        self.n_bases = n_bases
        self.regularizer = regularizer
        self.loss = []

        self.eps = eps
    
    def init_phase(self):
        n_bases = self.n_bases
        target = self.target

        phase = np.angle(target)
        self.phase = np.tile(phase[:,np.newaxis,:], reps=(1, n_bases, 1))
    
    def update(self, target, iteration=100):
        n_bases = self.n_bases
        eps = self.eps

        self.target = target
        F_bin, T_bin = target.shape

        self.base = np.random.rand(F_bin, n_bases)
        self.activation = np.random.rand(n_bases, T_bin)
        self.phase = 2 * np.pi * np.random.rand(F_bin, n_bases, T_bin)

        for idx in range(iteration):
            self.update_once()

            TVPhi = np.sum(self.base[:,:,np.newaxis] * self.activation[:,np.newaxis,:] * self.phase, axis=1)
            loss = self.criterion(TVPhi, target)
            self.loss.append(loss.sum())
        
    def update_once(self):
        raise NotImplementedError("Implement 'update_once' method")

class EUCNMF(NMFbase):
    def __init__(self, n_bases=2, eps=EPS):
        """
        Args:
            n_bases: number of bases
        """
        super().__init__(n_bases=n_bases, eps=eps)

        self.criterion = lambda input, target: (input - target)**2

    def update_once(self):
        target = self.target
        eps = self.eps

        T, V = self.base, self.activation

        # Update bases
        V_transpose = V.transpose(1,0)
        TV = T @ V
        TV[TV < eps] = eps
        TVV = TV @ V_transpose
        TVV[TVV < eps] = eps
        T = T * (target @ V_transpose / TVV)

        # Update activations
        T_transpose = T.transpose(1,0)
        TV = T @ V
        TV[TV < eps] = eps
        TTV = T_transpose @ TV
        TTV[TTV < eps] = eps
        V = V * (T_transpose @ target / TTV)

        self.base, self.activation = T, V

class KLNMF(NMFbase):
    def __init__(self, n_bases=2, eps=EPS):
        """
        Args:
            K: number of bases
        """
        super().__init__(n_bases=n_bases, eps=eps)

        self.criterion = generalized_kl_divergence

    def update_once(self):
        target = self.target
        eps = self.eps

        T, V = self.base, self.activation

        # Update bases
        V_transpose = V.transpose(1,0)
        TV = T @ V
        TV[TV < eps] = eps
        Vsum = V_transpose.sum(axis=0, keepdims=True)
        Vsum[Vsum < eps] = eps
        division = target / TV
        T = T * (division @ V_transpose / Vsum)

        # Update activations
        T_transpose = T.transpose(1,0)
        TV = T @ V
        TV[TV < eps] = eps
        Tsum = T_transpose.sum(axis=1, keepdims=True)
        Tsum[Tsum < eps] = eps
        division = target / TV
        V = V * (T_transpose @ division / Tsum)

        self.base, self.activation = T, V

class ISNMF(NMFbase):
    def __init__(self, n_bases=2, eps=EPS):
        """
        Args:
            K: number of bases
        """
        super().__init__(n_bases=n_bases, eps=eps)

        self.criterion = is_divergence

    def update_once(self):
        target = self.target
        eps = self.eps

        T, V = self.base, self.activation

        # Update bases
        V_transpose = V.transpose(1,0)
        TV = T @ V
        TV[TV < eps] = eps
        division, TV_inverse = target / (TV**2), 1 / TV
        TVV = TV_inverse @ V_transpose
        TVV[TVV < eps] = eps
        T = T * np.sqrt(division @ V_transpose / TVV)

        # Update activations
        T_transpose = T.transpose(1,0)
        TV = T @ V
        TV[TV < eps] = eps
        division, TV_inverse = target / (TV**2), 1 / TV
        TTV = T_transpose @ TV_inverse
        TTV[TTV < eps] = eps
        V = V * np.sqrt(T_transpose @ division / TTV)

        self.base, self.activation = T, V

class ComplexEUCNMF(ComplexNMFbase):
    def __init__(self, n_bases=2, regularizer=0.1, p=1, eps=EPS):
        """
        Args:
            n_bases: number of bases
        """
        super().__init__(n_bases=n_bases, eps=eps)

        self.criterion = lambda input, target: np.abs(input - target)**2
        self.regularizer, self.p = regularizer, p
    
    def update(self, target, iteration=100):
        n_bases = self.n_bases
        eps = self.eps

        self.target = target
        F_bin, T_bin = target.shape

        self.base = np.random.rand(F_bin, n_bases)
        self.activation = np.random.rand(n_bases, T_bin)

        self.init_phase()
        self.update_beta()

        for idx in range(iteration):
            self.update_once()

            TVPhi = np.sum(self.base[:,:,np.newaxis] * self.activation[np.newaxis,:,:] * self.phase, axis=1)
            loss = self.criterion(TVPhi, target)
            self.loss.append(loss.sum())
    
    def update_once(self):
        target = self.target
        regularizer, p = self.regularizer, self.p
        eps = self.eps

        T, V, Phi = self.base, self.activation, self.phase
        Ephi = np.exp(1j * Phi)
        Beta = self.Beta
        Beta[Beta < eps] = eps

        X = T[:,:,np.newaxis] * V[np.newaxis,:,:] * Ephi
        ZX = target - X.sum(axis=1)
        
        Z_bar = X + Beta * ZX[:,np.newaxis,:]
        V_bar = V
        V_bar[V_bar < eps] = eps
        Re = np.real(Z_bar.conj() * Ephi)

        # Update bases
        VV = V**2
        numerator = (V[np.newaxis,:,:] / Beta) * Re
        numerator = numerator.sum(axis=2)
        denominator = np.sum(VV[np.newaxis,:,:] / Beta, axis=2) # (n_bins, n_bases)
        denominator[denominator < eps] = eps
        T = numerator / denominator

        # Update activations
        TT = T**2
        numerator = (T[:,:,np.newaxis] / Beta) * Re
        numerator = numerator.sum(axis=0)
        denominator = np.sum(TT[:,:,np.newaxis] / Beta, axis=0) + regularizer * p * V_bar**(p - 2) # (n_bases, n_frames)
        denominator[denominator < eps] = eps
        V = numerator / denominator

        # Update phase    
        phase = np.angle(Z_bar)

        # Normalize bases
        T = T / T.sum(axis=0)

        self.base, self.activation, self.phase = T, V, phase

        # Update beta
        self.update_beta()
    
    def update_beta(self):
        T, V = self.base[:,:,np.newaxis], self.activation[np.newaxis,:,:]
        eps = self.eps

        TV = T * V # (n_bins, n_bases, n_frames)
        TVsum = TV.sum(axis=1, keepdims=True)
        TVsum[TVsum < eps] = eps
        self.Beta = TV / TVsum

def _test(metric='EUC'):
    np.random.seed(111)

    fft_size, hop_size = 1024, 256
    n_bases = 6
    iteration = 100
    
    signal, sr = read_wav("data/single-channel/music-8000.wav")
    
    T = len(signal)

    spectrogram = stft(signal, fft_size=fft_size, hop_size=hop_size)
    amplitude = np.abs(spectrogram)
    power = amplitude**2

    if metric == 'EUC':
        nmf = EUCNMF(n_bases)
    elif metric == 'IS':
        nmf = ISNMF(n_bases)
    elif metric == 'KL':
        nmf = KLNMF(n_bases)
    else:
        raise NotImplementedError("Not support {}-NMF".format(metric))

    nmf.update(power, iteration=iteration)

    amplitude[amplitude < EPS] = EPS

    estimated_power = nmf.base @ nmf.activation
    estimated_amplitude = np.sqrt(estimated_power)
    ratio = estimated_amplitude / amplitude
    estimated_spectrogram = ratio * spectrogram
    
    estimated_signal = istft(estimated_spectrogram, fft_size=fft_size, hop_size=hop_size, length=T)
    estimated_signal = estimated_signal / np.abs(estimated_signal).max()
    write_wav("data/NMF/{}/music-8000-estimated-iter{}.wav".format(metric, iteration), signal=estimated_signal, sr=8000)

    power[power < EPS] = EPS
    log_spectrogram = 10 * np.log10(power)

    plt.figure()
    plt.pcolormesh(log_spectrogram, cmap='jet')
    plt.colorbar()
    plt.savefig('data/NMF/spectrogram.png', bbox_inches='tight')
    plt.close()

    for idx in range(n_bases):
        estimated_power = nmf.base[:, idx: idx+1] @ nmf.activation[idx: idx+1, :]
        estimated_amplitude = np.sqrt(estimated_power)
        ratio = estimated_amplitude / amplitude
        estimated_spectrogram = ratio * spectrogram

        estimated_signal = istft(estimated_spectrogram, fft_size=fft_size, hop_size=hop_size, length=T)
        estimated_signal = estimated_signal / np.abs(estimated_signal).max()
        write_wav("data/NMF/{}/music-8000-estimated-iter{}-base{}.wav".format(metric, iteration, idx), signal=estimated_signal, sr=8000)

        estimated_power[estimated_power < EPS] = EPS
        log_spectrogram = 10 * np.log10(estimated_power)

        plt.figure()
        plt.pcolormesh(log_spectrogram, cmap='jet')
        plt.colorbar()
        plt.savefig('data/NMF/{}/estimated-spectrogram-iter{}-base{}.png'.format(metric, iteration, idx), bbox_inches='tight')
        plt.close()
    
    plt.figure()
    plt.plot(nmf.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/NMF/{}/loss.png'.format(metric), bbox_inches='tight')
    plt.close()

def _test_cnmf(metric='EUC'):
    np.random.seed(111)

    fft_size, hop_size = 1024, 256
    n_bases = 6
    p = 1.2
    iteration = 100
    
    signal, sr = read_wav("data/single-channel/music-8000.wav")
    T = len(signal)
    spectrogram = stft(signal, fft_size=fft_size, hop_size=hop_size)
    
    regularizer = 1e-5 * np.sum(np.abs(spectrogram)**2) / (n_bases**(1 - 2 / p))
    
    if metric == 'EUC':
        nmf = ComplexEUCNMF(n_bases, regularizer=regularizer, p=p)
    else:
        raise NotImplementedError("Not support {}-NMF".format(metric))
    
    nmf.update(spectrogram, iteration=iteration)

    estimated_spectrogram = nmf.base[:,:,np.newaxis] * nmf.activation[np.newaxis,:,:] * np.exp(1j*nmf.phase)
    estimated_spectrogram = estimated_spectrogram.sum(axis=1)
    estimated_signal = istft(estimated_spectrogram, fft_size=fft_size, hop_size=hop_size, length=T)
    estimated_signal = estimated_signal / np.abs(estimated_signal).max()
    write_wav("data/CNMF/{}/music-8000-estimated-iter{}.wav".format(metric, iteration), signal=estimated_signal, sr=sr)

    estimated_power = np.abs(estimated_spectrogram)**2
    estimated_power[estimated_power < EPS] = EPS
    log_spectrogram = 10 * np.log10(estimated_power)

    plt.figure()
    plt.pcolormesh(log_spectrogram, cmap='jet')
    plt.colorbar()
    plt.savefig('data/CNMF/spectrogram.png', bbox_inches='tight')
    plt.close()

    for idx in range(n_bases):
        estimated_spectrogram = nmf.base[:, idx: idx + 1] * nmf.activation[idx: idx + 1, :] * nmf.phase[:, idx, :]

        estimated_signal = istft(estimated_spectrogram, fft_size=fft_size, hop_size=hop_size, length=T)
        estimated_signal = estimated_signal / np.abs(estimated_signal).max()
        write_wav("data/CNMF/{}/music-8000-estimated-iter{}-base{}.wav".format(metric, iteration, idx), signal=estimated_signal, sr=8000)

        estimated_power = np.abs(estimated_spectrogram)**2
        estimated_power[estimated_power < EPS] = EPS
        log_spectrogram = 10 * np.log10(estimated_power)

        plt.figure()
        plt.pcolormesh(log_spectrogram, cmap='jet')
        plt.colorbar()
        plt.savefig('data/CNMF/{}/estimated-spectrogram-iter{}-base{}.png'.format(metric, iteration, idx), bbox_inches='tight')
        plt.close()
    
    plt.figure()
    plt.plot(nmf.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/CNMF/{}/loss.png'.format(metric), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    
    from utils.utils_audio import read_wav, write_wav
    from algorithm.stft import stft, istft

    plt.rcParams['figure.dpi'] = 200

    # "Real" NMF
    os.makedirs('data/NMF/EUC', exist_ok=True)
    os.makedirs('data/NMF/KL', exist_ok=True)
    os.makedirs('data/NMF/IS', exist_ok=True)

    _test(metric='EUC')
    _test(metric='IS')
    _test(metric='KL')

    # Complex NMF
    os.makedirs('data/CNMF/EUC', exist_ok=True)

    _test_cnmf(metric='EUC')