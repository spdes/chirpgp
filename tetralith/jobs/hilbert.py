import math
import jax
import jax.numpy as jnp
import scipy.signal
import numpy as np
import chirpgp.tools
from chirpgp.classical_methods import hilbert_method
from chirpgp.toymodels import gen_chirp, meow_freq, constant_mag, damped_exp_mag, random_ou_mag

jax.config.update("jax_enable_x64", True)

# Times
dt = 0.001
fs = 1 / dt
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
        true_chirp = gen_chirp(ts, mag, true_phase_func)

        Xi = 0.1
        ys = true_chirp + math.sqrt(Xi) * jax.random.normal(key_for_measurements, shape=(ts.size,))

        sos = scipy.signal.butter(N=8, Wn=18, btype='lowpass', analog=False, fs=fs, output='sos')
        filtered_ys = scipy.signal.sosfiltfilt(sos, ys)

        estimates = hilbert_method(ts, filtered_ys)
        rmse = chirpgp.tools.rmse(true_freq_func(ts)[1:], estimates)

        file_name = f'./results/hilbert_{name}_{mc}.npz'
        np.savez(file_name, estimates=estimates, rmse=rmse)

        print('Results saved in ' + file_name)
