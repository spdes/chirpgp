#!/bin/bash

#SBATCH --mem=130G
#SBATCH -n 20
#SBATCH -t 01:00:00

lam=$1
b=$2

cd $WRKDIR/chirp_estimation

module load buildtool-easybuild
module load Anaconda3/2021.05-nsc1

source ./venv/bin/activate
python setup.py develop

cd tetralith

echo 'Compute CRLB of the model'
echo 'Outputs are saved in ./logs.'
echo 'Start in 3 seconds.'
sleep 3

if [ ! -d "./logs" ]
then
    echo "Log folder does not exists. Trying to mkdir"
    mkdir logs
fi

python ./jobs/crlb_model.py -lam $lam -b $b -delta 0.1 -ell 1.0 -sigma 1.0 -Xi 0.1
