import warnings

import transform.stft as backend

def stft(input, fft_size, hop_size=None, window_fn='hann', normalize=False):
    warnings.warn("Use transform.stft.stft instead.", DeprecationWarning)
    return backend.stft(input, fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize)

def istft(input, fft_size, hop_size=None, window_fn='hann', normalize=False, length=None):
    warnings.warn("Use transform.stft.istft instead.", DeprecationWarning)
    return backend.istft(input, fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, length=length)
    
def build_window(fft_size, window_fn='hann'):
    warnings.warn("Use transform.stft.build_window instead.", DeprecationWarning)
    return backend.build_window(fft_size, window_fn=window_fn)

def build_optimal_window(window, hop_size=None):
    """
    Args:
        window: (window_length,)
    """
    warnings.warn("Use transform.stft.build_optimal_window instead.", DeprecationWarning)
    return backend.build_optimal_window(window, hop_size=hop_size)


