#!/bin/bash

#SBATCH --mem=64G
#SBATCH -N 1 --exclusive
#SBATCH -t 03:00:00

cd $WRKDIR/chirp_estimation

module load buildtool-easybuild
module load MATLAB/R2022a-nsc1

cd tetralith/jobs

if [ ! -d "../logs" ]
then
    echo "Log folder does not exists. Trying to mkdir"
    mkdir logs
fi

matlab -nosplash -nodesktop -r "fhc;"
