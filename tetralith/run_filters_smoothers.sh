#!/bin/bash

#SBATCH --mem=64G
#SBATCH -N 1 --exclusive
#SBATCH -t 02:00:00

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

python ./jobs/ekfs_mle.py > ./logs/ekfs_mle.log &
python ./jobs/ghfs_mle.py > ./logs/ghfs_mle.log &
python ./jobs/ckfs_mle.py > ./logs/ckfs_mle.log &
python ./jobs/cd_ekfs_mle.py > ./logs/cd_ekfs_mle.log &
python ./jobs/cd_ghfs_mle.py > ./logs/cd_ghfs_mle.log &
python ./jobs/cd_ckfs_mle.py > ./logs/cd_ckfs_mle.log &
python ./jobs/lascala_ekfs_mle.py > ./logs/lascala_ekfs_mle.log &
python ./jobs/lascala_ghfs_mle.py > ./logs/lascala_ghfs_mle.log &
wait
