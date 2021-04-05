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
        n_bins, n_frames = target.shape

        self.base = np.random.rand(n_bins, n_bases)
        self.activation = np.random.rand(n_bases, n_frames)

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
        n_bins, n_frames = target.shape

        self.base = np.random.rand(n_bins, n_bases)
        self.activation = np.random.rand(n_bases, n_frames)
        self.phase = 2 * np.pi * np.random.rand(n_bins, n_bases, n_frames)

        for idx in range(iteration):
            self.update_once()

            TVPhi = np.sum(self.base[:,:,np.newaxis] * self.activation[:,np.newaxis,:] * self.phase, axis=1)
            loss = self.criterion(TVPhi, target)
            self.loss.append(loss.sum())
        
    def update_once(self):
        raise NotImplementedError("Implement 'update_once' method")

class EUCNMF(NMFbase):
    def __init__(self, n_bases=2, domain=2, algorithm='mm', eps=EPS):
        """
        Args:
            n_bases: number of bases
        """
        super().__init__(n_bases=n_bases, eps=eps)

        assert 1 <= domain <= 2, "1 <= `domain` <= 2 is not satisfied."
        assert algorithm == 'mm', "algorithm must be 'mm'."

        self.domain = domain
        self.algorithm = algorithm
        self.criterion = lambda input, target: (target - input)**2
    
    def update(self, target, iteration=100):
        n_bases = self.n_bases
        domain = self.domain
        eps = self.eps

        self.target = target
        n_bins, n_frames = target.shape

        self.base = np.random.rand(n_bins, n_bases)
        self.activation = np.random.rand(n_bases, n_frames)

        for idx in range(iteration):
            self.update_once()

            TV = (self.base @ self.activation)**(2 / domain)
            loss = self.criterion(TV, target)
            self.loss.append(loss.sum())

    def update_once(self):
        if self.algorithm == 'mm':
            self.update_once_mm()
        else:
            raise ValueError("Not support {} based update.".format(self.algorithm))
    
    def update_once_mm(self):
        target = self.target
        domain = self.domain
        eps = self.eps

        T, V = self.base, self.activation

        # Update bases
        V_transpose = V.transpose(1,0)
        TV = T @ V
        TV[TV < eps] = eps
        TVV = (TV**((4 - domain) / domain)) @ V_transpose
        TVV[TVV < eps] = eps
        numerator = (target * (TV**((2 - domain) / domain))) @ V_transpose
        T = T * (numerator / TVV)**(domain / (4 - domain))

        # Update activations
        T_transpose = T.transpose(1,0)
        TV = T @ V
        TV[TV < eps] = eps
        TTV = T_transpose @ (TV**((4 - domain) / domain))
        TTV[TTV < eps] = eps
        numerator = T_transpose @ (target * (TV**((2 - domain) / domain)))
        V = V * (numerator / TTV)**(domain / (4 - domain))

        self.base, self.activation = T, V

class KLNMF(NMFbase):
    def __init__(self, n_bases=2, domain=2, algorithm='mm', eps=EPS):
        """
        Args:
            K: number of bases
        """
        super().__init__(n_bases=n_bases, eps=eps)

        assert 1 <= domain <= 2, "1 <= `domain` <= 2 is not satisfied."
        assert algorithm == 'mm', "algorithm must be 'mm'."

        self.domain = domain
        self.algorithm = algorithm
        self.criterion = generalized_kl_divergence
    
    def update(self, target, iteration=100):
        n_bases = self.n_bases
        domain = self.domain
        eps = self.eps

        self.target = target
        n_bins, n_frames = target.shape

        self.base = np.random.rand(n_bins, n_bases)
        self.activation = np.random.rand(n_bases, n_frames)

        for idx in range(iteration):
            self.update_once()

            TV = (self.base @ self.activation)**(2 / domain)
            loss = self.criterion(TV, target)
            self.loss.append(loss.sum())
    
    def update_once(self):
        if self.algorithm == 'mm':
            self.update_once_mm()
        else:
            raise ValueError("Not support {} based update.".format(self.algorithm))

    def update_once_mm(self):
        target = self.target
        domain = self.domain
        eps = self.eps

        T, V = self.base, self.activation

        # Update bases
        V_transpose = V.transpose(1,0)
        TV = T @ V
        TV[TV < eps] = eps
        TVV = (TV**((2 - domain) / domain)) @ V_transpose
        TVV[TVV < eps] = eps
        division = target / TV
        T = T * (division @ V_transpose / TVV)**(domain / 2)

        # Update activations
        T_transpose = T.transpose(1,0)
        TV = T @ V
        TV[TV < eps] = eps
        TTV = T_transpose @ (TV**((2 - domain) / domain))
        TTV[TTV < eps] = eps
        division = target / TV
        V = V * (T_transpose @ division / TTV)**(domain / 2)

        self.base, self.activation = T, V

class ISNMF(NMFbase):
    def __init__(self, n_bases=2, domain=2, algorithm='mm', eps=EPS):
        """
        Args:
            K: number of bases
            algorithm: 'mm': MM algorithm based update
        """
        super().__init__(n_bases=n_bases, eps=eps)

        assert 1 <= domain <= 2, "1 <= `domain` <= 2 is not satisfied."

        self.domain = domain
        self.algorithm = algorithm
        self.criterion = is_divergence
    
    def update(self, target, iteration=100):
        n_bases = self.n_bases
        domain = self.domain
        eps = self.eps

        self.target = target
        n_bins, n_frames = target.shape

        self.base = np.random.rand(n_bins, n_bases)
        self.activation = np.random.rand(n_bases, n_frames)

        for idx in range(iteration):
            self.update_once()

            TV = (self.base @ self.activation)**(2 / domain)
            loss = self.criterion(TV, target)
            self.loss.append(loss.sum())

    def update_once(self):
        if self.algorithm == 'mm':
            self.update_once_mm()
        elif self.algorithm == 'me':
            self.update_once_me()
        else:
            raise ValueError("Not support {} based update.".format(self.algorithm))
    
    def update_once_mm(self):
        target = self.target
        domain = self.domain
        eps = self.eps

        T, V = self.base, self.activation

        # Update bases
        V_transpose = V.transpose(1,0)
        TV = T @ V
        TV[TV < eps] = eps
        division, TV_inverse = target / (TV**((domain + 2) / domain)), 1 / TV
        TVV = TV_inverse @ V_transpose
        TVV[TVV < eps] = eps
        T = T * (division @ V_transpose / TVV)**(domain / (domain + 2))

        # Update activations
        T_transpose = T.transpose(1,0)
        TV = T @ V
        TV[TV < eps] = eps
        division, TV_inverse = target / (TV**((domain + 2) / domain)), 1 / TV
        TTV = T_transpose @ TV_inverse
        TTV[TTV < eps] = eps
        V = V * (T_transpose @ division / TTV)**(domain / (domain + 2))

        self.base, self.activation = T, V
    
    def update_once_me(self):
        target = self.target
        domain = self.domain
        eps = self.eps

        assert domain == 2, "Only domain = 2 is supported."

        T, V = self.base, self.activation

        # Update bases
        V_transpose = V.transpose(1,0)
        TV = T @ V
        TV[TV < eps] = eps
        division, TV_inverse = target / (TV**((domain + 2) / domain)), 1 / TV
        TVV = TV_inverse @ V_transpose
        TVV[TVV < eps] = eps
        T = T * (division @ V_transpose / TVV)

        # Update activations
        T_transpose = T.transpose(1,0)
        TV = T @ V
        TV[TV < eps] = eps
        division, TV_inverse = target / (TV**((domain + 2) / domain)), 1 / TV
        TTV = T_transpose @ TV_inverse
        TTV[TTV < eps] = eps
        V = V * (T_transpose @ division / TTV)

        self.base, self.activation = T, V

class tNMF(NMFbase):
    def __init__(self, n_bases=2, nu=1e+3, domain=2, algorithm='mm', eps=EPS):
        """
        Args:
            K: number of bases
            algorithm: 'mm': MM algorithm based update
        """
        super().__init__(n_bases=n_bases, eps=eps)

        def t_divergence(input, target):
            # TODO: implement criterion
            _input, _target = input + eps, target + eps
            
            return np.log(_input) + (2 + self.nu) / 2 * np.log(1 + (2 / nu) * (_target / _input))

        assert 1 <= domain <= 2, "1 <= `domain` <= 2 is not satisfied."

        self.nu = nu
        self.domain = domain
        self.algorithm = algorithm
        self.criterion = t_divergence
    
    def update(self, target, iteration=100):
        n_bases = self.n_bases
        domain = self.domain
        eps = self.eps

        self.target = target
        n_bins, n_frames = target.shape

        self.base = np.random.rand(n_bins, n_bases)
        self.activation = np.random.rand(n_bases, n_frames)

        for idx in range(iteration):
            self.update_once()

            TV = (self.base @ self.activation)**(2 / domain)
            loss = self.criterion(TV, target)
            self.loss.append(loss.sum())

    def update_once(self):
        if self.algorithm == 'mm':
            self.update_once_mm()
        else:
            raise ValueError("Not support {} based update.".format(self.algorithm))
    
    def update_once_mm(self):
        target = self.target
        domain = self.domain
        nu = self.nu
        eps = self.eps

        assert domain == 2, "`domain` is expected 2."

        T, V = self.base, self.activation
        Z = np.maximum(target, eps)

        # Update bases
        V_transpose = V.transpose(1, 0)
        TV = T @ V
        TV[TV < eps] = eps
        harmonic = 1 / (2 / ((2 + nu) * TV) + nu / ((2 + nu) * Z))
        division, TV_inverse = harmonic / (TV**2), 1 / TV
        TVV = TV_inverse @ V_transpose
        TVV[TVV < eps] = eps
        T = T * np.sqrt(division @ V_transpose / TVV)

        # Update activations
        T_transpose = T.transpose(1, 0)
        TV = T @ V
        TV[TV < eps] = eps
        harmonic = 1 / (2 / ((2 + nu) * TV) + nu / ((2 + nu) * Z))
        division, TV_inverse = harmonic / (TV**2), 1 / TV
        TTV = T_transpose @ TV_inverse
        TTV[TTV < eps] = eps
        V = V * np.sqrt(T_transpose @ division / TTV)

        self.base, self.activation = T, V

class CauchyNMF(NMFbase):
    def __init__(self, n_bases, domain=2, algorithm='naive-multipricative', eps=EPS):
        super().__init__(n_bases=n_bases, eps=eps)

        def cauchy_divergence(input, target):
            eps = self.eps

            _input, _target = input + eps, target + eps
            numerator = 2 * _target**2 + _input**2
            denominator = 3 * _target**2
            
            return np.log(_target / _input) + (3 / 2) * np.log(numerator / denominator)

        assert domain == 2, "Only `domain` = 2 is supported."

        self.domain = domain
        self.algorithm = algorithm
        self.criterion = cauchy_divergence
    
    def update_once(self):
        if self.algorithm == 'naive-multipricative':
            self.update_once_naive()
        elif self.algorithm == 'mm':
            self.update_once_mm()
        elif self.algorithm == 'me':
            self.update_once_me()
        elif self.algorithm == 'mm_fast':
            self.update_once_mm_fast()
        else:
            raise ValueError("Not support {} based update.".format(self.algorithm))

    def update_once_naive(self):
        """
        Cauchy Nonnegative Matrix Factorization
        See https://hal.inria.fr/hal-01170924/document
        """
        target = self.target
        domain = self.domain
        eps = self.eps

        assert domain == 2, "Only 'domain' = 2 is supported."

        T, V = self.base, self.activation

        TV = T @ V
        TV[TV < eps] = eps
        numerator = np.sum(V[np.newaxis,:,:] / TV[:,np.newaxis,:], axis=2)
        C = 2 * target + TV**2
        C[C < eps] = eps
        TVC = TV / C
        denominator = 3 * TVC @ V.transpose(1, 0)
        denominator[denominator < eps] = eps
        T = T * (numerator / denominator)

        TV = T @ V
        TV[TV < eps] = eps
        numerator = np.sum(T[:,:, np.newaxis] / TV[:,np.newaxis,:], axis=0)
        C = 2 * target + TV**2
        C[C < eps] = eps
        TVC = TV / C
        denominator = 3 * T.transpose(1, 0) @ TVC
        denominator[denominator < eps] = eps
        V = V * (numerator / denominator)

        self.base, self.activation = T, V

    def update_once_mm(self):
        target = self.target
        domain = self.domain
        eps = self.eps

        assert domain == 2, "Only 'domain' = 2 is supported."

        T, V = self.base, self.activation

        TV = T @ V
        TV[TV < eps] = eps
        numerator = np.sum(V[np.newaxis,:,:] / TV[:,np.newaxis,:], axis=2)
        C = 2 * target + TV**2
        C[C < eps] = eps
        TVC = TV / C
        denominator = 3 * TVC @ V.transpose(1, 0)
        denominator[denominator < eps] = eps
        T = T * np.sqrt(numerator / denominator)

        TV = T @ V
        TV[TV < eps] = eps
        numerator = np.sum(T[:,:, np.newaxis] / TV[:,np.newaxis,:], axis=0)
        C = 2 * target + TV**2
        C[C < eps] = eps
        TVC = TV / C
        denominator = 3 * T.transpose(1, 0) @ TVC
        denominator[denominator < eps] = eps
        V = V * np.sqrt(numerator / denominator)

        self.base, self.activation = T, V
    
    def update_once_me(self):
        """
        Cauchy Nonnegative Matrix Factorization
        See https://hal.inria.fr/hal-01170924/document
        """
        target = self.target
        domain = self.domain
        eps = self.eps

        assert domain == 2, "Only 'domain' = 2 is supported."

        T, V = self.base, self.activation
        
        TV = T @ V
        TV2Z = TV**2 + target
        TV2Z[TV2Z < eps] = eps
        A = (3 / 4) * (TV / TV2Z) @ V.transpose(1, 0)
        B = np.sum(V[np.newaxis,:,:] / TV[:,np.newaxis,:], axis=2)
        denominator = A + np.sqrt(A**2 + 2 * B * A)
        denominator[denominator < eps] = eps
        T = T * (B / denominator)

        TV = T @ V
        TV2Z = TV**2 + target
        TV2Z[TV2Z < eps] = eps
        A = (3 / 4) * T.transpose(1, 0) @ (TV / TV2Z)
        B = np.sum(T[:,:,np.newaxis] / TV[:,np.newaxis,:], axis=0)
        denominator = A + np.sqrt(A**2 + 2 * B * A)
        denominator[denominator < eps] = eps
        V = V * (B / denominator)

        self.base, self.activation = T, V

    def update_once_mm_fast(self):
        target = self.target
        domain = self.domain
        eps = self.eps

        assert domain == 2, "Only 'domain' = 2 is supported."

        T, V = self.base, self.activation

        # Update bases
        TV = T @ V
        C = 2 * target + TV**2
        CTV = C * TV
        CTV[CTV < eps] = eps
        ZCTV = target / CTV
        C[C < eps] = eps
        TVC = TV / C
        numerator = ZCTV @ V.transpose(1, 0)
        denominator = TVC @ V.transpose(1, 0)
        denominator[denominator < eps] = eps
        T = T * np.sqrt(numerator / denominator)

        # Update bases
        TV = T @ V
        C = 2 * target + TV**2
        CTV = C * TV
        CTV[CTV < eps] = eps
        ZCTV = target / CTV
        C[C < eps] = eps
        TVC = TV / C
        numerator = T.transpose(1, 0) @ ZCTV
        denominator = T.transpose(1, 0) @ TVC
        denominator[denominator < eps] = eps
        V = V * np.sqrt(numerator / denominator)

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

def _test(metric='EUC', algorithm='mm'):
    np.random.seed(111)

    fft_size, hop_size = 1024, 256
    n_bases = 6
    domain = 2
    iteration = 100
    
    signal, sr = read_wav("data/single-channel/music-8000.wav")
    
    T = len(signal)

    spectrogram = stft(signal, fft_size=fft_size, hop_size=hop_size)
    amplitude = np.abs(spectrogram)
    power = amplitude**2

    if metric == 'EUC':
        iteration = 80
        nmf = EUCNMF(n_bases, domain=domain, algorithm=algorithm)
    elif metric == 'KL':
        iteration = 50
        domain = 1.5
        nmf = KLNMF(n_bases, domain=domain, algorithm=algorithm)
    elif metric == 'IS':
        iteration = 50
        nmf = ISNMF(n_bases, domain=domain, algorithm=algorithm)
    elif metric == 't':
        iteration = 50
        nu = 100
        nmf = tNMF(n_bases, nu=nu, domain=domain, algorithm=algorithm)
    elif metric == 'Cauchy':
        iteration = 20
        nmf = CauchyNMF(n_bases, domain=domain, algorithm=algorithm)
    else:
        raise NotImplementedError("Not support {}-NMF".format(metric))

    nmf.update(power, iteration=iteration)

    amplitude[amplitude < EPS] = EPS

    estimated_power = (nmf.base @ nmf.activation)**(2 / domain)
    estimated_amplitude = np.sqrt(estimated_power)
    ratio = estimated_amplitude / amplitude
    estimated_spectrogram = ratio * spectrogram
    
    estimated_signal = istft(estimated_spectrogram, fft_size=fft_size, hop_size=hop_size, length=T)
    estimated_signal = estimated_signal / np.abs(estimated_signal).max()
    write_wav("data/NMF/{}/{}/music-8000-estimated-iter{}.wav".format(metric, algorithm, iteration), signal=estimated_signal, sr=sr)

    power[power < EPS] = EPS
    log_spectrogram = 10 * np.log10(power)

    plt.figure()
    plt.pcolormesh(log_spectrogram, cmap='jet')
    plt.colorbar()
    plt.savefig('data/NMF/spectrogram.png', bbox_inches='tight')
    plt.close()

    for idx in range(n_bases):
        estimated_power = (nmf.base[:, idx: idx+1] @ nmf.activation[idx: idx+1, :])**(2 / domain)
        estimated_amplitude = np.sqrt(estimated_power)
        ratio = estimated_amplitude / amplitude
        estimated_spectrogram = ratio * spectrogram

        estimated_signal = istft(estimated_spectrogram, fft_size=fft_size, hop_size=hop_size, length=T)
        estimated_signal = estimated_signal / np.abs(estimated_signal).max()
        write_wav("data/NMF/{}/{}/music-8000-estimated-iter{}-base{}.wav".format(metric, algorithm, iteration, idx), signal=estimated_signal, sr=sr)

        estimated_power[estimated_power < EPS] = EPS
        log_spectrogram = 10 * np.log10(estimated_power)

        plt.figure()
        plt.pcolormesh(log_spectrogram, cmap='jet')
        plt.colorbar()
        plt.savefig('data/NMF/{}/{}/estimated-spectrogram-iter{}-base{}.png'.format(metric, algorithm, iteration, idx), bbox_inches='tight')
        plt.close()
    
    plt.figure()
    plt.plot(nmf.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/NMF/{}/{}/loss.png'.format(metric, algorithm), bbox_inches='tight')
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
        write_wav("data/CNMF/{}/music-8000-estimated-iter{}-base{}.wav".format(metric, iteration, idx), signal=estimated_signal, sr=sr)

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
    os.makedirs('data/NMF/EUC/mm', exist_ok=True)
    os.makedirs('data/NMF/KL/mm', exist_ok=True)
    os.makedirs('data/NMF/IS/mm', exist_ok=True)
    os.makedirs('data/NMF/IS/me', exist_ok=True)
    os.makedirs('data/NMF/t/mm', exist_ok=True)
    os.makedirs('data/NMF/Cauchy/naive-multipricative', exist_ok=True)
    os.makedirs('data/NMF/Cauchy/mm', exist_ok=True)
    os.makedirs('data/NMF/Cauchy/me', exist_ok=True)
    os.makedirs('data/NMF/Cauchy/mm_fast', exist_ok=True)

    _test(metric='EUC', algorithm='mm')
    _test(metric='KL', algorithm='mm')
    _test(metric='IS', algorithm='mm')
    _test(metric='IS', algorithm='me')
    _test(metric='t', algorithm='mm')
    _test(metric='Cauchy', algorithm='naive-multipricative')
    _test(metric='Cauchy', algorithm='mm')
    _test(metric='Cauchy', algorithm='me')
    _test(metric='Cauchy', algorithm='mm_fast')

    # Complex NMF
    os.makedirs('data/CNMF/EUC', exist_ok=True)

    _test_cnmf(metric='EUC')