import numpy as np

EPS = 1e-12

def rceps(input, fft_size=None, eps=EPS):
    if np.iscomplexobj(input):
        raise ValueError("input should be real.")
    
    cepstrum = np.fft.irfft(np.log(np.abs(np.fft.rfft(input, fft_size) + eps)), fft_size)
    
    return cepstrum

def _test_rceps():
    waveform, sample_rate = wavread("./data/single-channel/mtlb.wav")
    cepstrum = rceps(waveform)

    print(cepstrum)

def _test_rceps_echo_cancel():
    waveform, sample_rate = wavread("./data/single-channel/mtlb.wav")

    lag, alpha = 0.23, 0.5
    delta = round(lag * sample_rate)

    orig = np.concatenate([waveform, np.zeros(delta)], axis=0)
    echo = alpha * np.concatenate([np.zeros(delta), waveform], axis=0)
    reverbed = orig + echo

    cepstrum = rceps(reverbed)

    print(cepstrum)

if __name__ == '__main__':
    from utils.audio import wavread

    _test_rceps()

    _test_rceps_echo_cancel()