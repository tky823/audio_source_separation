#!/bin/bash

export PATH="./local:$PATH"

reverb=0.160
mic_intervals='3-3-3-8-3-3-3'

mkdir -p "data/single-channel"
wget "https://soundeffect-lab.info/sound/voice/mp3//game/wizard-greeting1.mp3" -O "data/single-channel/man-44100.mp3"
wget "https://soundeffect-lab.info/sound/voice/mp3//game/swordwoman-win1.mp3" -O "data/single-channel/woman-44100.mp3"

mkdir -p "data/MIRD/Reverb${reverb}_${mic_intervals}"
url="https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/Impulse_response_Acoustic_Lab_Bar-Ilan_University__Reverberation_${reverb}s__${mic_intervals}.zip"
mkdir -p "data/MIRD/Reverb${reverb}_${mic_intervals}"
wget ${url} -O "data/MIRD/mird.zip"
unzip "data/MIRD/mird.zip" -d "data/MIRD/Reverb${reverb}_${mic_intervals}" 

prepare.py