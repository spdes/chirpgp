for lam in "0.1" "0.4" "0.7" "1.0"; do
    for b in "0.1" "0.4" "0.7" "1.0"; do
        sbatch run_crlb_ekf.sh $lam $b
        sbatch run_crlb_ghf.sh $lam $b
        sbatch run_crlb_model.sh $lam $b
    done
done
