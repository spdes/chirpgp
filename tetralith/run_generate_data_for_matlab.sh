#!/bin/bash

#SBATCH --mem=8G
#SBATCH -t 00:01:00

cd $WRKDIR/chirp_estimation

module load buildtool-easybuild
module load Anaconda3/2021.05-nsc1

source ./venv/bin/activate
python setup.py develop

cd tetralith

python generate_chirp_for_matlab.py &
python generate_harmonic_chirp_for_matlab.py &
wait
