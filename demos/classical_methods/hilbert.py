import math
import jax
import jax.numpy as jnp
import scipy.signal
import matplotlib.pyplot as plt
from chirpgp.classical_methods import hilbert_method
from chirpgp.toymodels import gen_chirp, meow_freq, constant_mag, damped_exp_mag, random_ou_mag
from chirpgp.tools import rmse

jax.config.update("jax_enable_x64", True)

# Times
dt = 0.001
fs = 1 / dt
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
    true_chirp = gen_chirp(ts, mag, true_phase_func)

    # Generate chirp measurements
    Xi = 0.1
    ys = true_chirp + math.sqrt(Xi) * jax.random.normal(key, shape=(ts.size,))

    # Filter out noises
    sos = scipy.signal.butter(N=8, Wn=18, btype='lowpass', analog=False, fs=fs, output='sos')
    filtered_ys = scipy.signal.sosfiltfilt(sos, ys)

    # Estimate
    estimated_freqs = hilbert_method(ts, filtered_ys)

    print(rmse(true_freq_func(ts)[1:], estimated_freqs))

    plt.plot(ts, true_freq_func(ts))
    plt.plot(ts[1:], estimated_freqs)
    plt.show()
