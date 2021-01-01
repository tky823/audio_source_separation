#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.io import loadmat
import librosa

def main():
    titles = ['aew', 'axb', 'bdl']
    target_sr = 16000
    T_min = None

    # Resample
    for idx, title in enumerate(titles):
        source, sr = librosa.load("./data/cmu_us_{}_arctic/wav/arctic_a{:.04}.wav".format(title, idx), target_sr)
        T = len(source)

        if T_min is None or T < T_min:
            T_min = T
    
    for idx, title in enumerate(titles):
        os.makedirs("./data/cmu_us_{}_arctic/trimmed".format(title), exist_ok=True)
        
        source, sr = librosa.load("./data/cmu_us_{}_arctic/wav/arctic_a{:.04}.wav".format(title, idx), target_sr)
        librosa.output.write_wav("./data/cmu_us_{}_arctic/trimmed/source-{}.wav".format(title, target_sr), source[:T_min], target_sr)

    # Room impulse response
    reverb = 0.16
    duration = 0.5
    samples = int(duration * target_sr)
    mic_intervals = "3-3-3-8-3-3-3"
    mic_indices = list(range(8))
    degrees = [0, 15, 30, 45, 60, 75, 90, 270, 285, 300, 315, 330]
    
    for title in titles:
        for degree in degrees:
            for mic_idx in mic_indices:
                convolve_mird(title, reverb=reverb, degree=degree, mic_intervals=mic_intervals, mic_idx=mic_idx, sr=target_sr, samples=samples)


def convolve_mird(title, reverb=0.160, degree=0, mic_intervals="3-3-3-8-3-3-3", mic_idx=0, sr=16000, samples=None):
    rir_path = "data/MIRD/Reverb{:.3f}_{}/Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_{:.3f}s)_{}_1m_{:03d}.mat".format(reverb, mic_intervals, reverb, mic_intervals, degree)
    rir_mat = loadmat(rir_path)

    rir = rir_mat['impulse_response']

    if samples is not None:
        rir = rir[:samples]

    source, sr = librosa.load("data/cmu_us_{}_arctic/trimmed/source-{}.wav".format(title, sr), sr)
    convolved_signals = np.convolve(source, rir[:, mic_idx])

    librosa.output.write_wav("./data/cmu_us_{}_arctic/trimmed/convolved-{}_deg{}-mic{}.wav".format(title, sr, degree, mic_idx), convolved_signals, sr)
    
if __name__ == '__main__':
    main()
