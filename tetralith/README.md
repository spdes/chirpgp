# Tetralith

By running the scripts in this folder, you can exactly reproduce the numerical results in the paper.

If you are using a Slurm-based computation cluster similar to that
of [https://www.nsc.liu.se/systems/tetralith/](https://www.nsc.liu.se/systems/tetralith/), please adapt and
run `bash ./run_all.sh`.

You can also run the experiments on your personal computer. Run `bash ./run_local.sh`. If you encounter out-of-memory
problem, try run `bash ./run_local_low_mem.sh` or change the number of parallel runs by yourself
in `./run_local_low_mem.sh`.

Results will be dumped in a new folder `./results` along with a log of the runnings save in `./logs`. You can plot/print
the results right away by running the scripts in `../paper_plots_tables`.

## Other files

- `generate_rndkeys.py`: Generate 1000 independent random seeds for the Monte Carlo runs. Seeds are saved
  in `rnd_keys.npy`.
- `generate_chirp_for_matlab.py` and `generate_harmonic_chirp_for_matlab.py`. The FHC method is implemented in Matlab. To use it, we need to convert the generated measurements to matlab.
- `setup_env.sh`: Setup Python (anaconda) venv in Tetralith.
