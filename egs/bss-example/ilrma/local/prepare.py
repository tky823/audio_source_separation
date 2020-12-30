#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa

def main():
    target_sr = 16000

    titles = ['man-16000', 'woman-16000']

    for title in titles:
        source, sr = librosa.load("./data/{}.mp3".format(title), target_sr)
        librosa.write_wav("./data/{}.wav".format(title), source, target_sr)
    
if __name__ == '__main__':
    main()
