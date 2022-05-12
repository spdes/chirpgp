#!/bin/bash

#SBATCH --mem 8G
#SBATCH -t 00:05:00

cd $WRKDIR/chirp_estimation

module load buildtool-easybuild
module load Anaconda3/2021.05-nsc1

source ./venv/bin/activate
python setup.py develop

cd tetralith

python ./jobs/anf.py &
python ./jobs/hilbert.py &
python ./jobs/mean_spectrogram.py &
wait
