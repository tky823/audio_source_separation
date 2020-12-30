#!/bin/bash

export PYTHONPATH="./local:$PYTHONPATH"

mkdir "data"
wget "https://soundeffect-lab.info/sound/voice/mp3//game/wizard-greeting1.mp3" -O "data/man-44100.mp3"
wget "https://soundeffect-lab.info/sound/voice/mp3//game/swordwoman-win1.mp3" -O "data/woman-44100.mp3"

prepare.py