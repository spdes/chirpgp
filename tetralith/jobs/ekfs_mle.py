import math
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import chirpgp.tools
from chirpgp.models import g, g_inv, build_chirp_model
from chirpgp.filters_smoothers import ekf, eks
from chirpgp.quadratures import gaussian_expectation
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

        init_theta = g_inv(jnp.array([0.1, 0.1, 0.1, 1., 1., 7.]))


        def obj_func(theta: jnp.ndarray):
            _, _, m_and_cov, m0, P0, H = build_chirp_model(g(theta))
            return ekf(m_and_cov, H, Xi, m0, P0, dt, ys)[-1][-1]


        opt_solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=obj_func)
        opt_vals, opt_state = opt_solver.run(init_theta)
        opt_params = g(opt_vals)
        print(f'Parameter learnt: {opt_params}. Convergence: {opt_state}')

        if opt_state.success:

            _, _, m_and_cov, m0, P0, H = build_chirp_model(opt_params)


            @jax.jit
            def filtering(measurements):
                return ekf(m_and_cov, H, Xi, m0, P0, dt, measurements)


            @jax.jit
            def smoothing(mfs, Pfs):
                return eks(m_and_cov, mfs, Pfs, dt)


            # Trigger jit
            _dummy = filtering(jnp.ones((2,)))
            smoothing(_dummy[0], _dummy[1])

            filtering_results = filtering(ys)
            smoothing_results = smoothing(filtering_results[0], filtering_results[1])

            estimated_freqs_mean = gaussian_expectation(ms=smoothing_results[0][:, 2],
                                                        chol_Ps=jnp.sqrt(smoothing_results[1][:, 2, 2]),
                                                        func=g, force_shape=True)[:, 0]
            rmse = chirpgp.tools.rmse(true_freq_func(ts), estimated_freqs_mean)
        else:
            print(f'This MC {mc} run with {mag} is divergent.')
            smoothing_results = (np.nan, np.nan)
            rmse = np.nan

        file_name = f'./results/ekfs_mle_{name}_{mc}.npz'
        np.savez(file_name, smoothing_mean=smoothing_results[0], smoothing_cov=smoothing_results[1], rmse=rmse)

        print('Results saved in ' + file_name)
