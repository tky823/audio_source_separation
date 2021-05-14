from scipy.io import wavfile
import numpy as np

def read_wav(path):
    sr, signal = wavfile.read(path)
    signal = signal / 32768
    
    return signal, sr

def write_wav(path, signal, sr, channel_last=True):
    signal = signal * 32768
    signal = np.clip(signal, -32768, 32767).astype(np.int16)

    if not signal.ndim in [1, 2]:
        raise ValueError("Only support 1D or 2D input.")
    if signal.ndim == 2 and not channel_last:
        signal = signal.transpose()
    wavfile.write(path, sr, signal)

def mu_law_compand(x, mu=255):
    return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)

def inv_mu_law_compand(y, mu=255):
    return np.sign(y) * ((1 + mu)**np.abs(y) - 1) / mu
