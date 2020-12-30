#!/bin/bash

export PATH="./local:$PATH"

reverb=0.160
mic_intervals='3-3-3-8-3-3-3'

mkdir -p "data"
mkdir -p "data/wizard"
mkdir -p "data/swordwoman"
mkdir -p "data/thief-boy"

wget "https://soundeffect-lab.info/sound/voice/mp3//game/wizard-greeting1.mp3" -O "data/wizard/source-44100.mp3"
wget "https://soundeffect-lab.info/sound/voice/mp3//game/swordwoman-win1.mp3" -O "data/swordwoman/source-44100.mp3"
wget "https://soundeffect-lab.info/sound/voice/mp3//game/thief-boy-greeting1.mp3" -O "data/thief-boy/source-44100.mp3"

mkdir -p "data/MIRD/Reverb${reverb}_${mic_intervals}"
url="https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/Impulse_response_Acoustic_Lab_Bar-Ilan_University__Reverberation_${reverb}s__${mic_intervals}.zip"
mkdir -p "data/MIRD/Reverb${reverb}_${mic_intervals}"
wget ${url} -O "data/MIRD/mird.zip"
unzip "data/MIRD/mird.zip" -d "data/MIRD/Reverb${reverb}_${mic_intervals}" 

prepare.py