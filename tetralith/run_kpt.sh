#!/bin/bash

#SBATCH --mem=16G
#SBATCH -t 00:30:00

cd $WRKDIR/chirp_estimation

module load buildtool-easybuild
module load Anaconda3/2021.05-nsc1

source ./venv/bin/activate
python setup.py develop

cd tetralith

if [ ! -d "./logs" ]
then
    echo "Log folder does not exists. Trying to mkdir"
    mkdir logs
fi

python ./jobs/kpt_mle.py > ./logs/kpt_mle.log &
python ./jobs/harmonic_kpt_mle.py > ./logs/harmonic_kpt_mle.log &
wait
