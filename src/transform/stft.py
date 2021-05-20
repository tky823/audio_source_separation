import numpy as np
from scipy import signal as ss

def stft(input, fft_size, hop_size=None, window_fn='hann', normalize=False):
    # window = build_window(fft_size, window_fn=window_fn) # (fft_size,)
    f, t, output = ss.stft(input, nperseg=fft_size, noverlap=fft_size-hop_size, window=window_fn)
    
    return output

def istft(input, fft_size, hop_size=None, window_fn='hann', normalize=False, length=None):
    # window = build_window(fft_size, window_fn=window_fn) # (fft_size,)
    t, output = ss.istft(input, nperseg=fft_size, noverlap=fft_size-hop_size, window=window_fn)

    if length is not None:
        output = output[...,:length]
    
    return output
    
def build_window(fft_size, window_fn='hann'):
    if window_fn == 'hann':
        window = ss.hann(fft_size, sym=False)
    elif window_fn == 'hamming':
        window = ss.hamming(fft_size, sym=False)
    else:
        raise ValueError("Not support {} window.".format(window_fn))
        
    return window

def build_optimal_window(window, hop_size=None):
    """
    Args:
        window: (window_length,)
    """
    window_length = len(window)

    if hop_size is None:
        hop_size = window_length//2
    
    windows = np.concatenate([
        np.roll(window[np.newaxis,:], hop_size*idx) for idx in range(window_length//hop_size)
    ], axis=0)
    
    power = windows**2
    norm = power.sum(axis=0)
    optimal_window = window / norm
    
    return optimal_window

def _test():
    np.random.seed(111)

    T = 66
    fft_size, hop_size = 8, 2
    window_fn = 'hamming'

    input = np.random.randn(T)

    X = stft(input, fft_size=fft_size, hop_size=hop_size)
    print(X.shape)
    power = np.abs(X)**2
    log_spectrogram = 10*np.log10(power+1e-15)
    
    f, t = log_spectrogram.shape
    t = np.arange(t + 1)
    f = np.arange(f + 1)
    
    plt.figure()
    plt.pcolormesh(t, f, log_spectrogram, cmap='jet')
    plt.savefig("data/STFT/specrtogram.png", bbox_inches='tight')
    plt.close()

    output = istft(X, fft_size=fft_size, hop_size=hop_size, length=T)

    plt.figure()
    plt.plot(input, linewidth=1, color='black')
    plt.plot(output, linewidth=1, color='red')
    plt.xlim(0, T-1)
    plt.savefig("data/STFT/Fourier.png", bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    plt.rcParams['figure.dpi'] = 200
    os.makedirs("data/STFT", exist_ok=True)

    _test()