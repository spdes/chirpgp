#!/bin/bash

#SBATCH -n 8
#SBATCH --mem 8G
#SBATCH -t 00:20:00

cd $WRKDIR/chirp_estimation

module load buildtool-easybuild
module load Anaconda3/2021.05-nsc1

source ./venv/bin/activate
python setup.py develop

cd tetralith

python ./jobs/mle_polynomial.py
