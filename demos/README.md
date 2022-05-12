# Demos

The scripts in this folder feature a number of chirp instantaneous frequency estimation methods on a toymodel defined in
Equation **. In particular, our proposed method is illustrated in these files:

- `./ekfs_mle.py`: Solve the proposed chirp-IF estimation model using an extended Kalman filter and smoother and
  estimate model parameters using MLE.
- `./ghfs_mle.py`: Solve the proposed chirp-IF estimation model using a Gauss--Hermite sigma-points filter and smoother
  and estimate model parameters using MLE.
- `./cd_ekfs_mle.py`: Solve the proposed chirp-IF estimation model *in continous time* by using a continuous-discrete
  extended filter and smoother and estimate model parameters using MLE.
- `./cd_ghfs_mle.py`: Solve the proposed chirp-IF estimation model *in continous time* by using a continuous-discrete
  Gauss--Hermite sigma-points filter and smoother and estimate model parameters using MLE.

For the sake of comparison, we also implemented a few classical/baseline methods in the following files.

- `./classical_methods/anf.py`: Pilot adaptive notch filter.
- `./classical_methods/hilbert.py`: Hilbert transform method.
- `./classical_methods/mean_spectrogram.py`: First-moment power spectrum method.
- `./classical_methods/mle_polynomial.py`: Polynomial regression method.
- `./classical_methods/lascala_ekfs_mle.py`: Almost the same as with `./ekfs_mle.py` except that the model is taken from
  La Scala et a., 1996.
- `./classical_methods/lascala_ghfs_mle.py`: Almost the same as with `./ghfs_mle.py` except that the model is taken from
  La Scala et a., 1996.
