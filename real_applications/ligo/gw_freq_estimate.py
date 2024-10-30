"""
Estimate gravitational wave and its instantaneous frequency.
"""
import jax
import jaxopt
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from chirpgp.models import g, g_inv, build_chirp_model
from chirpgp.filters_smoothers import sgp_filter, sgp_smoother
from chirpgp.quadratures import gaussian_expectation, SigmaPoints

jax.config.update("jax_enable_x64", True)

# Load gravitational wave strain data. Please download them by yourself, see README.md
ts, ys = jnp.asarray(np.genfromtxt('./data/fig1-observed-H.txt').T)
ts_true_gw, true_gw = jnp.asarray(np.genfromtxt('./data/fig1-waveform-H.txt').T)
dt = jnp.diff(ts)[0]
Xi = 0.3

# Sigma points
sgps = SigmaPoints.gauss_hermite(d=4, order=3)

# MLE parameter estimation
# From left to right, they are, lam, b, delta, ell, sigma, m0_1
init_theta = g_inv(jnp.array([0.1, 2., 0.5, 0.02, 40., 1.]))


# Objective function
def obj_func(theta: jnp.ndarray):
    _, _, m_and_cov, m0, P0, H = build_chirp_model(g(theta))
    return sgp_filter(m_and_cov, sgps, H, Xi, m0, P0, dt, ys)[-1][-1]


# Optimise
opt_solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=obj_func)
opt_vals, opt_state = opt_solver.run(init_theta)
opt_params = g(opt_vals)
print(f'Parameter learnt: {opt_params}. Convergence: {opt_state}')

# Filtering and smoothing based on the learnt parameters
_, _, m_and_cov, m0, P0, H = build_chirp_model(opt_params)


@jax.jit
def filtering(measurements):
    return sgp_filter(m_and_cov, sgps, H, Xi, m0, P0, dt, measurements)


@jax.jit
def smoothing(mfs, Pfs):
    return sgp_smoother(m_and_cov, sgps, mfs, Pfs, dt)


# Trigger jit
_dummy = filtering(jnp.ones((1,)))
smoothing(_dummy[0], _dummy[1])

# Compute posterior distributions
filtering_results = filtering(ys)
smoothing_results = smoothing(filtering_results[0], filtering_results[1])
estimated_freqs_mean = gaussian_expectation(ms=smoothing_results[0][:, 2],
                                            chol_Ps=jnp.sqrt(smoothing_results[1][:, 2, 2]),
                                            func=g, force_shape=True)[:, 0]

# Plot
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "san-serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 20})

fig, axs = plt.subplots(nrows=2, figsize=(14, 8), sharex='col')

axs[0].scatter(ts, ys, s=5, alpha=0.5, c='tab:purple', edgecolors='none', label='Measurements')
axs[0].plot(ts_true_gw, true_gw, linewidth=2, c='tab:blue', linestyle='--', label='Numerical relativity')
axs[0].plot(ts, smoothing_results[0][:, 1], linewidth=2, c='black', label='Estimated chirp')
axs[0].fill_between(ts,
                    smoothing_results[0][:, 1] - 1.96 * jnp.sqrt(smoothing_results[1][:, 1, 1]),
                    smoothing_results[0][:, 1] + 1.96 * jnp.sqrt(smoothing_results[1][:, 1, 1]),
                    color='black',
                    edgecolor='none',
                    alpha=0.15,
                    label='0.95 quantile')
axs[0].set_ylabel(r'Strain $\times 10^{-21}$')
axs[0].grid(linestyle='--', alpha=0.3, which='both')
axs[0].legend(ncol=2, scatterpoints=5, fontsize=20)

axs[1].plot(ts, estimated_freqs_mean, c='black', linewidth=2, label='Estimated frequency')
axs[1].fill_between(ts,
                    g(smoothing_results[0][:, 2] - 1.96 * jnp.sqrt(smoothing_results[1][:, 2, 2])),
                    g(smoothing_results[0][:, 2] + 1.96 * jnp.sqrt(smoothing_results[1][:, 2, 2])),
                    color='black',
                    edgecolor='none',
                    alpha=0.15,
                    label='0.95 quantile')
axs[1].grid(linestyle='--', alpha=0.3, which='both')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Frequency (Hz)')
axs[1].set_xticks([0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45])
axs[1].set_yticks([0, 50, 100, 150, 200, 250, 300])
axs[1].legend(loc='upper center', ncol=2, fontsize=20)

plt.tight_layout(pad=0.1)
plt.subplots_adjust(left=0.069, bottom=0.08)
plt.savefig('./gw.pdf')
plt.show()
