import math
import jax
import jaxopt
import jax.numpy as jnp
import jax.scipy.optimize
import matplotlib.pyplot as plt
from chirpgp.models import g, g_inv, build_harmonic_chirp_model
from chirpgp.filters_smoothers import sgp_filter, sgp_smoother
from chirpgp.quadratures import gaussian_expectation, SigmaPoints
from chirpgp.toymodels import gen_harmonic_chirp, meow_freq, constant_mag, damped_exp_mag, random_ou_mag
from chirpgp.tools import rmse

jax.config.update("jax_enable_x64", True)

# Times
dt = 0.001
T = 3141
ts = jnp.linspace(dt, dt * T, T)

# Random keys
key = jax.random.PRNGKey(777)
key, subkey = jax.random.split(key)

# Number of harmonics
num_harmonics = 3

sgps = SigmaPoints.cubature(d=8)

# Frequency
true_freq_func, true_phase_func = meow_freq(offset=8.)

for mag in [constant_mag(1.),
            damped_exp_mag(0.3),
            random_ou_mag(1., 1., subkey)]:

    # Generate chirp
    true_chirp = gen_harmonic_chirp(ts, [mag] * num_harmonics, true_phase_func)

    # Generate chirp measurements
    Xi = 0.1
    ys = true_chirp + math.sqrt(Xi) * jax.random.normal(key, shape=(ts.size,))

    # MLE parameter estimation
    # From left to right, they are, lam, b, delta, ell, sigma, m0_1
    init_theta = g_inv(jnp.array([0.1, 0.1, 0.1, 1., 1., 7.]))

    # Objective function
    @jax.jit
    def obj_func(theta: jnp.ndarray):
        _, _, m_and_cov, m0, P0, H = build_harmonic_chirp_model(g(theta), num_harmonics=num_harmonics)
        return sgp_filter(m_and_cov, sgps, H, Xi, m0, P0, dt, ys)[-1][-1]

    # Optimise
    opt_solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=obj_func)
    opt_vals, opt_state = opt_solver.run(init_theta)
    opt_params = g(opt_vals)
    print(f'Parameter learnt: {opt_params}. Convergence: {opt_state}')

    # Filtering and smoothing based on the learnt parameters
    _, _, m_and_cov, m0, P0, H = build_harmonic_chirp_model(opt_params, num_harmonics=num_harmonics)

    @jax.jit
    def filtering(measurements):
        return sgp_filter(m_and_cov, sgps, H, Xi, m0, P0, dt, measurements)

    @jax.jit
    def smoothing(mfs, Pfs):
        return sgp_smoother(m_and_cov, sgps, mfs, Pfs, dt)

    # Trigger jit
    _dummy = filtering(jnp.ones((2,)))
    smoothing(_dummy[0], _dummy[1])

    filtering_results = filtering(ys)
    smoothing_results = smoothing(filtering_results[0], filtering_results[1])

    # Note that the distribution of f=g(V) is not Gaussian
    # The confidence interval in the following may not be centred at E[g(V)]
    estimated_freqs_mean = gaussian_expectation(ms=smoothing_results[0][:, -2],
                                                chol_Ps=jnp.sqrt(smoothing_results[1][:, -2, -2]),
                                                func=g, force_shape=True)[:, 0]

    print('RMSE: ', rmse(true_freq_func(ts), estimated_freqs_mean))

    # Plot
    plt.plot(ts, true_freq_func(ts), c='tab:blue', linestyle='--', label='True fundamental freq')
    plt.plot(ts, estimated_freqs_mean, c='black', label='Estimated fundamental freq')
    plt.fill_between(ts,
                     g(smoothing_results[0][:, -2] - 1.96 * jnp.sqrt(smoothing_results[1][:, -2, -2])),
                     g(smoothing_results[0][:, -2] + 1.96 * jnp.sqrt(smoothing_results[1][:, -2, -2])),
                     color='black',
                     edgecolor='none',
                     alpha=0.15)

    plt.plot(ts, 2 * true_freq_func(ts), c='tab:blue', linestyle='--', label='Second harmonic')
    plt.plot(ts, 2 * estimated_freqs_mean, c='black', label='Estimated second harmonic')
    plt.fill_between(ts,
                     2 * g(smoothing_results[0][:, -2] - 1.96 * jnp.sqrt(smoothing_results[1][:, -2, -2])),
                     2 * g(smoothing_results[0][:, -2] + 1.96 * jnp.sqrt(smoothing_results[1][:, -2, -2])),
                     color='black',
                     edgecolor='none',
                     alpha=0.15)

    plt.plot(ts, 3 * true_freq_func(ts), c='tab:blue', linestyle='--', label='Second harmonic')
    plt.plot(ts, 3 * estimated_freqs_mean, c='black', label='Estimated second harmonic')
    plt.fill_between(ts,
                     3 * g(smoothing_results[0][:, -2] - 1.96 * jnp.sqrt(smoothing_results[1][:, -2, -2])),
                     3 * g(smoothing_results[0][:, -2] + 1.96 * jnp.sqrt(smoothing_results[1][:, -2, -2])),
                     color='black',
                     edgecolor='none',
                     alpha=0.15)

    plt.legend()
    plt.show()

    plt.scatter(ts, ys, s=1, alpha=0.3, c='tab:blue', edgecolors='none', label='Measurements')
    plt.plot(ts, true_chirp, label='True chirp')
    plt.plot(ts, smoothing_results[0] @ H, label='Estimated')
    plt.legend()
    plt.show()
