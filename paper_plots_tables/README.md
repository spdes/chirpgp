# Plots

The scripts in this folder generate the figures in the paper.

- `plot_chirp_samples.py`: Plot samples drawn from the chirp estimation model. **Figure 3**.
- `plot_cov_chirp_sde_cond_v.py`: Plot the covariance function of the chirp model by conditioning on a realisation of
  IF. **Figure 2**.
- `plot_cov_harmonic_sde.py`: Plot the covariance function of a harmonic SDE. **Figure 1**.
- `plot_estimation.py`: Plot the toymodel experiment results. **Figure 4**.
- `print_rmse_table.py`: Print and plot the RMSE statistics. **Table I** and **Figure 6**.

# Note

To use `./plot_estimation.py` and `./print_rmse_table.py`, you need to run the experiments to get their results and save
in `../tetralith/results`. Please refer to the readme in `../tetralith/` for how to do the experiments.

The learnt model parameters in Table II can be found in the log files in `../tetralith/logs`.
