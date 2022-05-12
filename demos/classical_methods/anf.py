import math
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from chirpgp.classical_methods import adaptive_notch_filter
from chirpgp.toymodels import gen_chirp_envelope, meow_freq, constant_mag, damped_exp_mag, random_ou_mag
from chirpgp.tools import rmse
from jax.config import config

config.update("jax_enable_x64", True)

# Times
dt = 0.001
T = 3141
ts = jnp.linspace(dt, dt * T, T)

# Random keys
key = jax.random.PRNGKey(555)
key, subkey = jax.random.split(key)

# Frequency
true_freq_func, true_phase_func = meow_freq(offset=8.)

for mag in [constant_mag(1.),
            damped_exp_mag(0.3),
            random_ou_mag(1., 1., subkey)]:
    # Generate chirp
    true_chirp = gen_chirp_envelope(ts, mag, true_phase_func, 0.)

    # Generate chirp measurements
    Xi = 0.1
    ys = true_chirp + math.sqrt(Xi) * jax.random.normal(key, shape=(ts.size,))

    # Estimate
    mu = 0.015
    gamma_w = mu ** 2 / 2
    gamma_alpha = mu * gamma_w / 4
    estimated_freqs, _, _ = adaptive_notch_filter(ts, ys, alpha0=0., w0=true_freq_func(dt), s0=1 + 0.j,
                                                  mu=mu, gamma_alpha=gamma_alpha, gamma_w=gamma_w)

    print(rmse(true_freq_func(ts), estimated_freqs))

    plt.plot(ts, true_freq_func(ts))
    plt.plot(ts, estimated_freqs)
    plt.show()
