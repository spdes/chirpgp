"""
fastF0Nls - Fast Nonlinear Least Squares Estimation of the Fundamental Frequency (aka Pitch).

All rights reserved Jesper Kjær Nielsen.

https://github.com/jkjaer/fastF0Nls.
"""
import argparse
import os
import math
import ctypes
import jax
import jax.numpy as jnp
import numpy as np
import scipy
import chirpgp.tools
import matplotlib.pyplot as plt
from ctypes import c_void_p, c_double, c_int
from jax.config import config

from chirpgp.toymodels import meow_freq, constant_mag, gen_harmonic_chirp, damped_exp_mag, random_ou_mag, gen_chirp

config.update("jax_enable_x64", True)

libpath = os.path.dirname(os.path.realpath(__file__)) + "/../../others/fastF0Nls/cpp/lib/single_pitch.so"
lib = ctypes.cdll.LoadLibrary(libpath)

lib.single_pitch_new.argtypes = [c_int, c_int, c_int, c_void_p]
lib.single_pitch_new.restype = c_void_p

lib.single_pitch_est.argtypes = [c_void_p, c_void_p, c_double, c_double]
lib.single_pitch_est.restype = c_double

lib.single_pitch_est_fast.argtypes = [c_void_p, c_void_p, c_double, c_double]
lib.single_pitch_est_fast.restype = c_double

lib.single_pitch_del.argtypes = [c_void_p]
lib.single_pitch_del.restype = None

lib.single_pitch_model_order.argtypes = [c_void_p]
lib.single_pitch_model_order.restype = int


class single_pitch():
    def __init__(self, nData, maxModelOrder, pitchBounds, nFftGrid=None):
        """
        Initialize the object

        Example:
        N = 500
        L = 15
        pitch_bounds = np.array([0.01, 0.45])
        sp = single_pitch(L, N, pitch_bounds)

        Uses default nFftGrid = 5NL. Otherwise set via parameter nFftGrid

        Pitch bound is in normalized frequency (1.0 = Nyquist frequency)

        """

        if nFftGrid == None:
            nFftGrid = 5 * nData * maxModelOrder

        self.obj = lib.single_pitch_new(maxModelOrder, nFftGrid, nData,
                                        pitchBounds.ctypes.data)

    def est(self, data, lnBFZeroOrder=0.0, eps=1e-3, method=0):
        """
        Estimates based on double vector (data)

        The function returns the estimated frequency in radians per sample

        Addtional parameters
        lnBFZeroOrder: log Baysian factor for order zero. (default 0.0)
               May be increased if non desirable low power signals interfere
               and should considered noise.
        eps:  Requested accuracy in radians per sample. (Defaut 1e-3)
              Higher accuracy (smaller eps) may require more solve time
              and vice-versa

        If method = 0 (default), then the algorithm uses the following steps

         1. Calculate the objective function for all candidate model
           order and on the Fourier grid
         2. Perform model order selection
         3. Refine the for the selected model order

        If method != 0, then the algorithm uses the more computational
        demanding steps (but possible more accurate)

         1. Calculate the objective function for all candidate model
           order and on the Fourier grid
         2. Refine the best estimates for each model order
         3. Perform model order selection

        """

        if method == 0:
            return lib.single_pitch_est_fast(self.obj, data.ctypes.data,
                                             lnBFZeroOrder, eps)
        else:
            return lib.single_pitch_est(self.obj, data.ctypes.data,
                                        lnBFZeroOrder, eps)

    def modelOrder(self):
        """
        Retuns the estimated model order for the lastest solve.

        """

        return lib.single_pitch_model_order(self.obj)

    def __del__(self):
        lib.single_pitch_del(self.obj)


def _force_odd(number):
    if number % 2 == 0:
        return number + 1
    else:
        return number


def pitch_track(ys, fs, num_harmonics, window_length=300, window_overlap=295):
    T = ys.shape[0]
    f0Bounds = np.array([2, 15]) / fs
    maxNoHarmonics = num_harmonics
    f0estimator = single_pitch(window_length, maxNoHarmonics, f0Bounds)

    # Set up windows
    num_windows = round((T - window_length) / (window_length - window_overlap)) + 1
    windows_centre_positions = window_length / 2 + np.arange(num_windows) * (window_length - window_overlap)
    windows_times = windows_centre_positions * dt

    # Sliding window pitch tracking
    f0estimates = np.zeros((num_windows,))
    for k in range(num_windows):
        idx = k * (window_length - window_overlap)
        ys_chunk = ys[idx:idx + window_length]
        f0estimates[k] = (fs / (2 * math.pi)) * f0estimator.est(ys_chunk, eps=1e-7, method=1)

    return windows_times, f0estimates


parser = argparse.ArgumentParser(description='Single or harmonic chirp')
parser.add_argument('-harmonic', type=float, help='Set 1 for harmonic, 0 for non-harmonic.')
args = parser.parse_args()

# Times
dt = 0.001
T = 3141
ts = jnp.linspace(dt, dt * T, T)

# Frequency
true_freq_func, true_phase_func = meow_freq(offset=8.)

# Loop over MC runs
num_mcs = 100
for mc in range(num_mcs):

    key = jnp.asarray(np.load('./rnd_keys.npy')[mc])
    key_for_measurements, key_for_ou = jax.random.split(key)

    for mag, name in zip((constant_mag(1.), damped_exp_mag(0.3), random_ou_mag(1., 1., key_for_ou)),
                         ('const', 'damped', 'ou')):
        if args.harmonic:
            num_harmonics = 3
            true_chirp = gen_harmonic_chirp(ts, [mag] * num_harmonics, true_phase_func)
        else:
            num_harmonics = 1
            true_chirp = gen_chirp(ts, mag, true_phase_func)

        Xi = 0.1
        ys = np.asarray(true_chirp + math.sqrt(Xi) * jax.random.normal(key_for_measurements, shape=(ts.size,)))

        # Optimal hand-tuned window length
        window_length = 300
        window_overlap = window_length - 1

        # Estimate and median-smoothing
        windows_times, f0estimates = pitch_track(ys, 1 / dt, num_harmonics,
                                                 window_length=window_length, window_overlap=window_overlap)
        smoothed_f0estimates = scipy.signal.medfilt(f0estimates, _force_odd(round(window_length / 2)))
        rmse = chirpgp.tools.rmse(smoothed_f0estimates, true_freq_func(windows_times))

        if args.harmonic:
            file_name = f'./results/harmonic_fastf0nls_{name}_{mc}.npz'
        else:
            file_name = f'./results/fastf0nls_{name}_{mc}.npz'

        np.savez(file_name, windows_times=windows_times, f0estimates=f0estimates,
                 smoothed_f0estimates=smoothed_f0estimates, rmse=rmse)
        print('Results saved in ' + file_name)
