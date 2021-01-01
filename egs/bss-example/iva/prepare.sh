#!/bin/bash

export PATH="./local:$PATH"

reverb=0.160
mic_intervals='3-3-3-8-3-3-3'

mkdir -p "data"

wget "http://festvox.org/cmu_arctic/packed/cmu_us_aew_arctic.tar.bz2" -O "data/cmu_us_aew_arctic.tar.bz2"
wget "http://festvox.org/cmu_arctic/packed/cmu_us_axb_arctic.tar.bz2" -O "data/cmu_us_axb_arctic.tar.bz2"
wget "http://festvox.org/cmu_arctic/packed/cmu_us_bdl_arctic.tar.bz2" -O "data/cmu_us_bdl_arctic.tar.bz2"

tar -xvf "data/cmu_us_aew_arctic.tar.bz2" -C "data"
tar -xvf "data/cmu_us_axb_arctic.tar.bz2" -C "data"
tar -xvf "data/cmu_us_bdl_arctic.tar.bz2" -C "data"

mkdir -p "data/MIRD/Reverb${reverb}_${mic_intervals}"
url="https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/Impulse_response_Acoustic_Lab_Bar-Ilan_University__Reverberation_${reverb}s__${mic_intervals}.zip"
mkdir -p "data/MIRD/Reverb${reverb}_${mic_intervals}"
wget ${url} -O "data/MIRD/mird.zip"
unzip "data/MIRD/mird.zip" -d "data/MIRD/Reverb${reverb}_${mic_intervals}" 

prepare.py