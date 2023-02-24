"""
This generated Figure 7 in the paper.
"""
import math
import jax
import scipy
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from chirpgp.models import g
from chirpgp.quadratures import gaussian_expectation
from chirpgp.toymodels import meow_freq, gen_harmonic_chirp, constant_mag, damped_exp_mag, random_ou_mag
from jax.config import config

config.update("jax_enable_x64", True)

# mc = 11
mc = 17

path = '../tetralith/results/'

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "san-serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

fig, axs = plt.subplots(nrows=5, ncols=1, sharex='col', sharey='row', figsize=(7, 11))

# Produce the signal and measurements at that MC seed
dt = 0.001
T = 3141
fs = 1 / dt
ts = jnp.linspace(dt, dt * T, T)

key = jnp.asarray(np.load('../tetralith/rnd_keys.npy')[mc])
key_for_measurements, key_for_ou = jax.random.split(key)

true_freq_func, true_phase_func = meow_freq(offset=8.)

mag, mag_name = random_ou_mag(1., 1., key_for_ou), 'ou'

true_chirp = gen_harmonic_chirp(ts, [mag] * 3, true_phase_func)

Xi = 0.1
ys = true_chirp + math.sqrt(Xi) * jax.random.normal(key_for_measurements, shape=(ts.size,))

# Plot true chirp and CKFS chirp estimation
axs[0].plot(ts, true_chirp, c='tab:blue', linestyle='-', label='True chirp')
axs[0].scatter(ts, ys, s=2, alpha=0.5, c='tab:purple', edgecolors='none', label='Measurements')
axs[0].grid(linestyle='--', alpha=0.3, which='both')

# Plot true IF
for i in range(1, 5):
    axs[i].plot(ts, true_freq_func(ts), c='tab:blue', linestyle='--', label='True fundamental IF')

# Plot CKFS IF estimates
smoothing_mean = np.load(path + f'harmonic_ckfs_mle_{mag_name}_{mc}.npz')['smoothing_mean']
smoothing_cov = np.load(path + f'harmonic_ckfs_mle_{mag_name}_{mc}.npz')['smoothing_cov']
estimated_freqs_mean = gaussian_expectation(ms=smoothing_mean[:, -2],
                                            chol_Ps=jnp.sqrt(smoothing_cov[:, -2, -2]),
                                            func=g, force_shape=True)[:, 0]
axs[1].plot(ts, estimated_freqs_mean, c='black', label='IF posterior mean')
axs[1].fill_between(ts,
                    g(smoothing_mean[:, -2] - 1.96 * jnp.sqrt(smoothing_cov[:, -2, -2])),
                    g(smoothing_mean[:, -2] + 1.96 * jnp.sqrt(smoothing_cov[:, -2, -2])),
                    color='black',
                    edgecolor='none',
                    alpha=0.15, label='0.95 quantile')
axs[1].grid(linestyle='--', alpha=0.3, which='both')

# Plot KPT
smoothing_mean = np.load(path + f'harmonic_kpt_mle_{mag_name}_{mc}.npz')['smoothing_mean']
smoothing_cov = np.load(path + f'harmonic_kpt_mle_{mag_name}_{mc}.npz')['smoothing_cov']
estimated_freqs_mean = gaussian_expectation(ms=smoothing_mean[:, 0] / 2 / math.pi * fs,
                                            chol_Ps=jnp.sqrt(
                                                smoothing_cov[:, 0, 0]) / 2 / math.pi * fs,
                                            func=g, force_shape=True)[:, 0]
axs[2].plot(ts, estimated_freqs_mean, c='black', label='IF posterior mean')
axs[2].fill_between(ts,
                    g((smoothing_mean[:, 0] - 1.96 * jnp.sqrt(smoothing_cov[:, 0, 0])) / 2 / math.pi * fs),
                    g((smoothing_mean[:, 0] + 1.96 * jnp.sqrt(smoothing_cov[:, 0, 0])) / 2 / math.pi * fs),
                    color='black',
                    edgecolor='none',
                    alpha=0.15, label='0.95 quantile')
axs[2].grid(linestyle='--', alpha=0.3, which='both')

# Plot FHC
file_name = path + f'harmonic_fhc_{mag_name}_{mc}.mat'
fhc_results = scipy.io.loadmat(file_name)
fhc_ts = np.squeeze(fhc_results['segmentTimes'])
fhc_estimates = np.squeeze(fhc_results['pitchTrack'])
axs[3].plot(fhc_ts, fhc_estimates, c='black', label='IF estimate')
axs[3].grid(linestyle='--', alpha=0.3, which='both')

# Plot fastf0nls
fastf0nls_results = np.load(path + f'harmonic_fastf0nls_{mag_name}_{mc}.npz')
fastf0nls_ts = fastf0nls_results['windows_times']
fastf0nls_estimates = fastf0nls_results['smoothed_f0estimates']
axs[4].plot(fastf0nls_ts, fastf0nls_estimates, c='black', label='IF estimate')
axs[4].grid(linestyle='--', alpha=0.3, which='both')

# Legends
axs[0].legend(loc='lower center', ncol=2, fontsize=12, scatterpoints=5)
for k in range(1, 5):
    axs[k].legend(loc='lower left', ncol=1, fontsize=12)

# X title
axs[0].set_title(r'$t\mapsto\alpha(t)$ random process')

# X labels
axs[-1].set_xlabel('$t$')

# Y labels
axs[0].set_ylabel('Harmonic chirp')
axs[1].set_ylabel('CKFS MLE')
axs[2].set_ylabel('KPT')
axs[3].set_ylabel('FHC')
axs[4].set_ylabel('FastNLS')

plt.tight_layout(pad=0.1)
plt.subplots_adjust(left=0.095)
plt.savefig('./harmonic-estimation.pdf')
plt.show()
