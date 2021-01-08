#!/bin/bash

export PATH="./local:$PATH"

data_root="data"
titles="aew axb bdl"
reverb=0.160
duration=0.5
mic_intervals='3-3-3-8-3-3-3'
distance=1

mkdir -p ${data_root}

for title in ${titles} ; do
    if [ -e "${data_root}/cmu_us_${title}_arctic" ]; then
        echo "Already downloaded dataset cmu_us_${title}_arctic"
    else
        wget "http://festvox.org/cmu_arctic/packed/cmu_us_${title}_arctic.tar.bz2" -O "${data_root}/cmu_us_${title}_arctic.tar.bz2"
        tar -xvf "${data_root}/cmu_us_${title}_arctic.tar.bz2" -C "${data_root}"
    fi
done

if [ -e "${data_root}/MIRD/Reverb${reverb}_${mic_intervals}" ]; then
    echo "Already downloaded dataset MIRD"
else
    mkdir -p "${data_root}/MIRD/Reverb${reverb}_${mic_intervals}"
    url="https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/Impulse_response_Acoustic_Lab_Bar-Ilan_University__Reverberation_${reverb}s__${mic_intervals}.zip"
    wget ${url} -O "${data_root}/MIRD/mird.zip"
    unzip "${data_root}/MIRD/mird.zip" -d "${data_root}/MIRD/Reverb${reverb}_${mic_intervals}" 
fi


prepare.py \
--data_root "${data_root}" \
--titles "${titles}" \
--reverb ${reverb} \
--duration ${duration} \
--mic_intervals ${mic_intervals} \
--distance ${distance}