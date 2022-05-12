#!/bin/bash

echo 'Run all experiments in parallel on a local computer instead of Tetralith.'
echo 'You might need a descent CPU to run these fast, and >16G mem.'
echo 'Outputs are saved in ./logs.'
echo 'Start in 10 seconds.'
sleep 10

if [ ! -d "./logs" ]
then
    echo "Log folder does not exists. Trying to mkdir"
    mkdir logs
fi

python -u ./jobs/anf.py | tee ./logs/anf.log &
python -u ./jobs/hilbert.py | tee ./logs/hilbert.log &
python -u ./jobs/mean_spectrogram.py | tee ./logs/mean_spectrogram.log &
python -u ./jobs/mle_polynomial.py | tee ./logs/mle_polynomial.log &
python -u ./jobs/ekfs_mle.py | tee ./logs/ekfs_mle.log &
python -u ./jobs/ghfs_mle.py | tee ./logs/ghfs_mle.log &
python -u ./jobs/cd_ekfs_mle.py | tee ./logs/cd_ekfs_mle.log &
python -u ./jobs/cd_ghfs_mle.py | tee ./logs/cd_ghfs_mle.log &
python -u ./jobs/lascala_ekfs_mle.py | tee ./logs/lascala_ekfs_mle.log &
python -u ./jobs/lascala_ghfs_mle.py | tee ./logs/lascala_ghfs_mle.log &
wait
