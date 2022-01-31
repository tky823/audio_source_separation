from scipy.io import wavfile
import numpy as np

def wavread(path, channel_last=True):
    sample_rate, waveform = wavfile.read(path)
    waveform = waveform / 32768

    if not waveform.ndim in [1, 2]:
        raise ValueError("Only support 1D or 2D input.")
    
    if waveform.ndim == 2 and not channel_last:
        waveform = waveform.transpose()
    
    return waveform, sample_rate

def wavwrite(path, signal, sr, channel_last=True):
    signal = 32768 * signal
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
