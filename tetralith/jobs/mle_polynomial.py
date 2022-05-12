import math
import jax
import jax.numpy as jnp
import scipy.signal
import numpy as np
import chirpgp.tools
from chirpgp.classical_methods import mle_polynomial
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

        # See `init_poly_coeffs.m` in the demo folder
        initial_poly_coeffs = [1.,
                               7.791782e+00, 5.488218e+00, -2.723514e+01, 9.018465e+00, 1.431405e+02,
                               -2.483806e+02, 1.738925e+02, -6.028065e+01, 1.003177e+01, -5.527010e-01, -1.907047e-02]
        perb = jnp.array(initial_poly_coeffs) * 2e-5
        init_params = jnp.array(initial_poly_coeffs) + perb * jax.random.normal(jax.random.PRNGKey(666),
                                                                                shape=(len(initial_poly_coeffs),))

        optimised_poly_coeffs, _ = mle_polynomial(ts, ys, Xi, init_params, method='levenberg_marquardt',
                                                  lr=0.4, nu=0.3)
        estimates = jnp.polyval(jnp.flip(optimised_poly_coeffs[1:]), ts)
        rmse = chirpgp.tools.rmse(true_freq_func(ts), estimates)

        file_name = f'./results/mle_polynomial_{name}_{mc}.npz'
        np.savez(file_name, estimates=estimates, rmse=rmse)

        print('Results saved in ' + file_name)
