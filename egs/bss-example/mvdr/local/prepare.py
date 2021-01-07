#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
from scipy.io import loadmat
import soundfile as sf

parser = argparse.ArgumentParser(description="Example of frequency-domain ICA (FDICA)")

parser.add_argument('--data_root', type=str, default=None, help='Path for dataset ROOT directory.')
parser.add_argument('--titles', type=str, default="aew axb bdl", help='Path for dataset ROOT directory.')
parser.add_argument('--reverb', type=float, default=0.16, help='The reverberation time (T60).')
parser.add_argument('--duration', type=float, default=0.5, help='The trimming time of impulse response.')
parser.add_argument('--mic_intervals', type=str, default="8-8-8-8-8-8-8", help='The microphone intervals.')
parser.add_argument('--distance', type=float, default=1, help='The distance between micprophone and sources.')

def main(args):
    data_root = args.data_root
    titles = args.titles.split(' ')
    target_sr = 16000
    T_min = None

    # Resample
    for idx, title in enumerate(titles):
        path = os.path.join(data_root, "cmu_us_{}_arctic/wav/arctic_a{:04d}.wav".format(title, idx+1))
        source, sr = sf.read(path)
        T = len(source)

        if T_min is None or T < T_min:
            T_min = T
    
    for idx, title in enumerate(titles):
        path = os.path.join(data_root, "cmu_us_{}_arctic/wav/arctic_a{:04d}.wav".format(title, idx+1))
        source, sr = sf.read(path)

        path = os.path.join(data_root, "cmu_us_{}_arctic/trimmed".format(title))
        os.makedirs(path, exist_ok=True)
        path = os.path.join(data_root, "cmu_us_{}_arctic/trimmed/source-{}.wav".format(title, target_sr))
        sf.write(path, source[:T_min], target_sr)

    # Room impulse response
    reverb = args.reverb
    duration = args.duration
    samples = int(duration * target_sr)
    mic_intervals = args.mic_intervals
    mic_indices = list(range(8))
    distance = args.distance
    degrees = [0, 15, 30, 45, 60, 75, 90, 270, 285, 300, 315, 330, 345]
    
    for title in titles:
        for degree in degrees:
            for mic_idx in mic_indices:
                convolve_mird(data_root, title, reverb=reverb, degree=degree, mic_intervals=mic_intervals, mic_idx=mic_idx, distance=distance, sr=target_sr, samples=samples)


def convolve_mird(data_root, title, reverb=0.160, degree=0, mic_intervals="3-3-3-8-3-3-3", mic_idx=0, distance=1, sr=16000, samples=None):
    rir_path = os.path.join(data_root, "MIRD/Reverb{:.3f}_{}/Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_{:.3f}s)_{}_{:.0f}m_{:03d}.mat".format(reverb, mic_intervals, reverb, mic_intervals, distance, degree))
    rir_mat = loadmat(rir_path)

    rir = rir_mat['impulse_response']

    if samples is not None:
        rir = rir[:samples]

    wav_path = os.path.join(data_root, "cmu_us_{}_arctic/trimmed/source-{}.wav".format(title, sr))
    source, sr = sf.read(wav_path)
    convolved_signals = np.convolve(source, rir[:, mic_idx])

    wav_path = os.path.join(data_root, "cmu_us_{}_arctic/trimmed/convolved-{}_deg{}-mic{}.wav".format(title, sr, degree, mic_idx))
    sf.write(wav_path, convolved_signals, sr)
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
