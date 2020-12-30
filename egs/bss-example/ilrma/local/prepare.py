#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa

def main():
    target_sr = 16000

    titles = ['man', 'woman']

    for title in titles:
        source, sr = librosa.load("./data/{}-44100.mp3".format(title), target_sr)
        librosa.output.write_wav("./data/{}-16000.wav".format(title), source, target_sr)
    
if __name__ == '__main__':
    main()
