#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat
import librosa

def main():
    titles = ['man', 'woman']
    target_sr = 16000
    T_min = None

    # Resample
    for title in titles:
        source, sr = librosa.load("./data/{}-44100.mp3".format(title), target_sr)
        T = len(source)
        librosa.output.write_wav("./data/{}-{}.wav".format(title, target_sr), source, target_sr)

        if T_min is None or T < T_min:
            T_min = T
    
    for title in titles:
        source, sr = librosa.load("./data/{}-44100.mp3".format(title), target_sr)
        librosa.output.write_wav("./data/{}-{}.wav".format(title, target_sr), source[:T_min], target_sr)

    # Room impulse response
    reverb = 0.16
    duration = 0.5
    samples = int(duration * target_sr)
    mic_intervals = "3-3-3-8-3-3-3"
    mic_indices = [3, 4]
    degrees = [60, 300]
    
    for mic_idx in mic_indices:
        convolve_mird(titles, reverb=reverb, degrees=degrees, mic_intervals=mic_intervals, mic_idx=mic_idx, sr=target_sr, samples=samples)


def convolve_mird(titles, reverb=0.160, degrees=[0], mic_intervals="3-3-3-8-3-3-3", mic_idx=0, sr=16000, samples=None):
    for title_idx in range(len(titles)):
        degree = degrees[title_idx]
        title = titles[title_idx]
        rir_path = "data/MIRD/Reverb{:.3f}_{}/Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_{:.3f}s)_{}_1m_{:03d}.mat".format(reverb, mic_intervals, reverb, mic_intervals, degree)
        rir_mat = loadmat(rir_path)

        rir = rir_mat['impulse_response']

        if samples is not None:
            rir = rir[:samples]

        source, sr = librosa.load("data/{}-{}.wav".format(title, sr), sr)
        convolved_signals = np.convolve(source, rir[:, mic_idx])

        librosa.output.write_wav("./data/{}-{}_convolved_deg{}-mic{}.wav".format(title, sr, degree, mic_idx), convolved_signals, sr)
    
if __name__ == '__main__':
    main()
