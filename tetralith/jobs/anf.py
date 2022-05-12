import math
import jax
import jax.numpy as jnp
import numpy as np
import chirpgp.tools
from chirpgp.classical_methods import adaptive_notch_filter
from chirpgp.toymodels import gen_chirp_envelope, meow_freq, constant_mag, damped_exp_mag, random_ou_mag
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
        true_chirp = gen_chirp_envelope(ts, mag, true_phase_func, 0.)

        Xi = 0.1
        ys = true_chirp + math.sqrt(Xi) * jax.random.normal(key_for_measurements, shape=(ts.size,))

        mu = 0.015
        gamma_w = mu ** 2 / 2
        gamma_alpha = mu * gamma_w / 4
        estimates, _, _ = adaptive_notch_filter(ts, ys, alpha0=0., w0=true_freq_func(dt), s0=1 + 0.j,
                                                mu=mu, gamma_alpha=gamma_alpha, gamma_w=gamma_w)
        rmse = chirpgp.tools.rmse(true_freq_func(ts), estimates)

        file_name = f'./results/anf_{name}_{mc}.npz'
        np.savez(file_name, estimates=estimates, rmse=rmse)

        print('Results saved in ' + file_name)
