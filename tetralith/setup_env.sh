#!/bin/bash

#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=00:10:00

cd $WRKDIR/chirp_estimation

module load buildtool-easybuild
module load Anaconda3/2021.05-nsc1

mkdir venv
python -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip

pip install -r requirements.txt
python setup.py develop

if [ ! -d "./tetralith/results" ]
then
    echo "Folder results does not exists. Trying to mkdir"
    mkdir ./tetralith/results
fi
