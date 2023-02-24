#!/bin/bash

#SBATCH --mem=64G
#SBATCH -N 1 --exclusive
#SBATCH -t 03:00:00

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

python ./jobs/harmonic_ekfs_mle.py > ./logs/harmonic_ekfs_mle.log &
python ./jobs/harmonic_ckfs_mle.py > ./logs/harmonic_ckfs_mle.log &
wait
