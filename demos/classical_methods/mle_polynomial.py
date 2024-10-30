import math
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from chirpgp.classical_methods import mle_polynomial
from chirpgp.toymodels import gen_chirp, meow_freq, constant_mag, damped_exp_mag, random_ou_mag
from chirpgp.tools import rmse

jax.config.update("jax_enable_x64", True)

# Times
dt = 0.001
T = 3141
ts = jnp.linspace(dt, dt * T, T)

# Random keysmean_power_spectrum
key = jax.random.PRNGKey(555)
key, subkey = jax.random.split(key)
key_for_poly, _ = jax.random.split(key)

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

    # Give a good initial polynomial coefficients, see `init_poly_coeffs.m`
    initial_poly_coeffs = [1.,
                           7.791782e+00, 5.488218e+00, -2.723514e+01, 9.018465e+00, 1.431405e+02,
                           -2.483806e+02, 1.738925e+02, -6.028065e+01, 1.003177e+01, -5.527010e-01, -1.907047e-02]
    perb = jnp.array(initial_poly_coeffs) * 2e-5
    init_params = jnp.array(initial_poly_coeffs) + perb * jax.random.normal(key_for_poly,
                                                                            shape=(len(initial_poly_coeffs),))

    # Estimate
    optimised_poly_coeffs, obj_vals = mle_polynomial(ts, ys, Xi, init_params, method='levenberg_marquardt',
                                                     lr=0.4, nu=0.3)
    estimated_freqs = jnp.polyval(jnp.flip(optimised_poly_coeffs[1:]), ts)

    print(rmse(true_freq_func(ts), estimated_freqs))

    plt.plot(ts, true_freq_func(ts))
    plt.plot(ts, estimated_freqs)
    plt.show()

    plt.plot(obj_vals)
    plt.show()
