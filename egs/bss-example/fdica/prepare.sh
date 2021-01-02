#!/bin/bash

export PATH="./local:$PATH"

data_root="data"
titles="[aew,axb,bdl]"
reverb=0.160
duration=0.5
mic_intervals='3-3-3-8-3-3-3'

mkdir -p ${data_root}

wget "http://festvox.org/cmu_arctic/packed/cmu_us_aew_arctic.tar.bz2" -O "${data_root}/cmu_us_aew_arctic.tar.bz2"
wget "http://festvox.org/cmu_arctic/packed/cmu_us_axb_arctic.tar.bz2" -O "${data_root}/cmu_us_axb_arctic.tar.bz2"
wget "http://festvox.org/cmu_arctic/packed/cmu_us_bdl_arctic.tar.bz2" -O "${data_root}/cmu_us_bdl_arctic.tar.bz2"

tar -xvf "${data_root}/cmu_us_aew_arctic.tar.bz2" -C "${data_root}"
tar -xvf "${data_root}/cmu_us_axb_arctic.tar.bz2" -C "${data_root}"
tar -xvf "${data_root}/cmu_us_bdl_arctic.tar.bz2" -C "${data_root}"

mkdir -p "${data_root}/MIRD/Reverb${reverb}_${mic_intervals}"
url="https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/Impulse_response_Acoustic_Lab_Bar-Ilan_University__Reverberation_${reverb}s__${mic_intervals}.zip"
mkdir -p "${data_root}/MIRD/Reverb${reverb}_${mic_intervals}"
wget ${url} -O "${data_root}/MIRD/mird.zip"
unzip "${data_root}/MIRD/mird.zip" -d "${data_root}/MIRD/Reverb${reverb}_${mic_intervals}" 

prepare.py \
--data_root ${data_root} \
--titles ${titles} \
--reverb ${reverb} \
--duration ${duration} \
--mic_intervals ${mic_intervals}