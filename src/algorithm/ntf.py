"""
Non-negative tensor factorization
"""
import numpy as np

EPS=1e-12

class NTFbase:
    def __init__(self, n_basis=2, eps=EPS):
        """
        Args:
            n_basis: number of basis
        """

        self.n_basis = n_basis
        self.loss = []

        self.eps = eps
    
    def __call__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        Z, T, V = self.partitioning, self.basis, self.activation

        return Z.copy(), T.copy(), V.copy()
    
    def update(self, target, iteration=100):
        n_basis = self.n_basis
        eps = self.eps

        self.target = target
        n_channels, n_bins, n_frames = target.shape

        self.partitioning = np.random.rand(n_channels, n_basis)
        self.basis = np.random.rand(n_bins, n_basis)
        self.activation = np.random.rand(n_basis, n_frames)

        for idx in range(iteration):
            self.update_once()

            TV = self.basis @ self.activation
            loss = self.compute_loss()
            self.loss.append(loss.sum())
        
    def update_once(self):
        raise NotImplementedError("Implement 'update_once' method")
    
    def compute_loss(self):
        raise NotImplementedError("Implement 'compute_loss' method")

class EUCNTF(NTFbase):
    def __init__(self, n_basis, eps=EPS):
        super().__init__(n_basis=n_basis, eps=eps)
    
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
    
    def update_once(self):
        eps = self.eps

        Z, T, V = self.partitioning, self.basis, self.activation
        X = self.target

        # Update basis
        X_hat = np.sum(Z[:, np.newaxis, :, np.newaxis] * T[np.newaxis, :, :, np.newaxis] * V[np.newaxis, np.newaxis, :, :], axis=2) # (N, I, J)
        numerator = X[:, :, np.newaxis, :] * Z[:, np.newaxis, :, np.newaxis] * V[np.newaxis, np.newaxis, :, :]
        denominator = X_hat[:, :, np.newaxis, :] * Z[:, np.newaxis, :, np.newaxis] * V[np.newaxis, np.newaxis, :, :]
        numerator = numerator.sum(axis=(0, 3))
        numerator[numerator < eps] = eps
        denominator = denominator.sum(axis=(0, 3))
        denominator[denominator < eps] = eps
        T = T * (numerator / denominator)

        # Update activations
        X_hat = np.sum(Z[:, np.newaxis, :, np.newaxis] * T[np.newaxis, :, :, np.newaxis] * V[np.newaxis, np.newaxis, :, :], axis=2) # (N, I, J)
        numerator = X[:, :, np.newaxis, :] * Z[:, np.newaxis, :, np.newaxis] * T[np.newaxis, :, :, np.newaxis]
        denominator = X_hat[:, :, np.newaxis, :] * Z[:, np.newaxis, :, np.newaxis] * T[np.newaxis, :, :, np.newaxis]
        numerator = numerator.sum(axis=(0, 1))
        numerator[numerator < eps] = eps
        denominator = denominator.sum(axis=(0, 1))
        denominator[denominator < eps] = eps
        V = V * (numerator / denominator)

        # Update partitioning
        X_hat = np.sum(Z[:, np.newaxis, :, np.newaxis] * T[np.newaxis, :, :, np.newaxis] * V[np.newaxis, np.newaxis, :, :], axis=2) # (N, I, J)
        numerator = X[:, :, np.newaxis, :] * T[np.newaxis, :, :, np.newaxis] * V[np.newaxis, np.newaxis, :, :]
        denominator = X_hat[:, :, np.newaxis, :] * T[np.newaxis, :, :, np.newaxis] * V[np.newaxis, np.newaxis, :, :]
        numerator = numerator.sum(axis=(1, 3))
        numerator[numerator < eps] = eps
        denominator = denominator.sum(axis=(1, 3))
        denominator[denominator < eps] = eps
        Z = Z * (numerator / denominator)

        self.partitioning, self.basis, self.activation = Z, T, V
    
    def compute_loss(self):
        Z, T, V = self.partitioning, self.basis, self.activation
        X = self.target

        X_hat = np.sum(Z[:, np.newaxis, :, np.newaxis] * T[np.newaxis, :, :, np.newaxis] * V[np.newaxis, np.newaxis, :, :], axis=2) # (N, I, J)
        loss = np.sum((X - X_hat)**2)

        return loss

def _test(metric='EUC', algorithm='mm'):
    np.random.seed(111)

    fft_size, hop_size = 1024, 256
    n_basis = 6
    iteration = 100
    
    signal, sr = read_wav("../../dataset/sample-song/sample-3_mixture_16000.wav")

    spectrogram = stft(signal.T, fft_size=fft_size, hop_size=hop_size)
    amplitude = np.abs(spectrogram)
    power = amplitude**2

    if metric == 'EUC':
        iteration = 80
        ntf = EUCNTF(n_basis)
    else:
        raise NotImplementedError("Not support {}-NTF".format(metric))

    Z, T, V = ntf(power, iteration=iteration)

    plt.figure()
    plt.plot(ntf.loss, color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('data/NTF/{}/{}/loss.png'.format(metric, algorithm), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    
    from utils.utils_audio import read_wav
    from transform.stft import stft

    plt.rcParams['figure.dpi'] = 200

    # "Real" NMF
    os.makedirs('data/NTF/EUC/mm', exist_ok=True)

    _test(metric='EUC')