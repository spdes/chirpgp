"""
In order to compare to the methods that are implemented in Matlab, we export the signals and measurements in the .mat
format.
Please execute this file in the folder ./tetralith.
"""
import math
import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import savemat
from chirpgp.toymodels import gen_chirp, meow_freq, constant_mag, damped_exp_mag, random_ou_mag
from jax.config import config

config.update("jax_enable_x64", True)

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
        true_chirp = gen_chirp(ts, mag, true_phase_func)

        Xi = 0.1
        ys = true_chirp + math.sqrt(Xi) * jax.random.normal(key_for_measurements, shape=(ts.size,))

        mdict = {'dt': dt,
                 'T': T,
                 'Xi': Xi,
                 'ys': ys}

        filename = './matlab_data/' + f'chirp_mc_{mc}_mag_{name}.mat'
        savemat(filename, mdict, oned_as='column')
        print(f'Data saved in {filename}.')
