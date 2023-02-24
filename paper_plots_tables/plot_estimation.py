"""
Plot some demonstrative estimation results.

                   const        damped       OU
GHFS (chirp)
GHFS (IF)
GHFS [24] (IF)
Spectrogram (IF)
ANF (IF)
"""
import math
import jax
import scipy
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from chirpgp.models import g
from chirpgp.quadratures import gaussian_expectation
from chirpgp.toymodels import meow_freq, gen_chirp, constant_mag, damped_exp_mag, random_ou_mag
from jax.config import config

config.update("jax_enable_x64", True)

mc = 0

path = '../tetralith/results/'

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "san-serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

fig, axs = plt.subplots(nrows=6, ncols=3, sharex='col', sharey='row', figsize=(15, 12))

# Produce the signal and measurements at that MC seed
dt = 0.001
T = 3141
fs = 1 / dt
ts = jnp.linspace(dt, dt * T, T)

key = jnp.asarray(np.load('../tetralith/rnd_keys.npy')[mc])
key_for_measurements, key_for_ou = jax.random.split(key)

true_freq_func, true_phase_func = meow_freq(offset=8.)

for (col, mag), mag_name in zip(enumerate([constant_mag(1.), damped_exp_mag(0.3), random_ou_mag(1., 1., key_for_ou)]),
                                ['const', 'damped', 'ou']):
    true_chirp = gen_chirp(ts, mag, true_phase_func)

    Xi = 0.1
    ys = true_chirp + math.sqrt(Xi) * jax.random.normal(key_for_measurements, shape=(ts.size,))

    # Plot true chirp and GHFS chirp estimation
    axs[0, col].plot(ts, true_chirp, c='tab:blue', linestyle='-', label='True chirp')
    axs[0, col].scatter(ts, ys, s=2, alpha=0.5, c='tab:purple', edgecolors='none', label='Measurements')
    axs[0, col].grid(linestyle='--', alpha=0.3, which='both')

    # Plot true IF
    for i in range(1, 6):
        axs[i, col].plot(ts, true_freq_func(ts), c='tab:blue', linestyle='--', label='True IF')

    # Plot GHFS IF estimates
    smoothing_mean = np.load(path + f'ghfs_mle_{mag_name}_{mc}.npz')['smoothing_mean']
    smoothing_cov = np.load(path + f'ghfs_mle_{mag_name}_{mc}.npz')['smoothing_cov']
    estimated_freqs_mean = gaussian_expectation(ms=smoothing_mean[:, -2],
                                                chol_Ps=jnp.sqrt(smoothing_cov[:, -2, -2]),
                                                func=g, force_shape=True)[:, 0]
    axs[1, col].plot(ts, estimated_freqs_mean, c='black', label='IF posterior mean')
    axs[1, col].fill_between(ts,
                             g(smoothing_mean[:, -2] - 1.96 * jnp.sqrt(smoothing_cov[:, -2, -2])),
                             g(smoothing_mean[:, -2] + 1.96 * jnp.sqrt(smoothing_cov[:, -2, -2])),
                             color='black',
                             edgecolor='none',
                             alpha=0.15, label='0.95 quantile')
    axs[1, col].grid(linestyle='--', alpha=0.3, which='both')

    # Plot GHFS (La Scala) IF estimates
    smoothing_mean = np.load(path + f'lascala_ghfs_mle_{mag_name}_{mc}.npz')['smoothing_mean']
    smoothing_cov = np.load(path + f'lascala_ghfs_mle_{mag_name}_{mc}.npz')['smoothing_cov']
    estimated_freqs_mean = gaussian_expectation(ms=smoothing_mean[:, 2],
                                                chol_Ps=jnp.sqrt(smoothing_cov[:, 2, 2]),
                                                func=g, force_shape=True)[:, 0]
    axs[2, col].plot(ts, estimated_freqs_mean, c='black', label='IF posterior mean')
    axs[2, col].fill_between(ts,
                             g(smoothing_mean[:, 2] - 1.96 * jnp.sqrt(smoothing_cov[:, 2, 2])),
                             g(smoothing_mean[:, 2] + 1.96 * jnp.sqrt(smoothing_cov[:, 2, 2])),
                             color='black',
                             edgecolor='none',
                             alpha=0.15, label='0.95 quantile')
    axs[2, col].grid(linestyle='--', alpha=0.3, which='both')

    # Plot KPT
    smoothing_mean = np.load(path + f'kpt_mle_{mag_name}_{mc}.npz')['smoothing_mean']
    smoothing_cov = np.load(path + f'kpt_mle_{mag_name}_{mc}.npz')['smoothing_cov']
    estimated_freqs_mean = gaussian_expectation(ms=smoothing_mean[:, 0] / 2 / math.pi * fs,
                                                chol_Ps=jnp.sqrt(
                                                    smoothing_cov[:, 0, 0]) / 2 / math.pi * fs,
                                                func=g, force_shape=True)[:, 0]
    axs[3, col].plot(ts, estimated_freqs_mean, c='black', label='IF posterior mean')
    axs[3, col].fill_between(ts,
                             g((smoothing_mean[:, 0] - 1.96 * jnp.sqrt(smoothing_cov[:, 0, 0])) / 2 / math.pi * fs),
                             g((smoothing_mean[:, 0] + 1.96 * jnp.sqrt(smoothing_cov[:, 0, 0])) / 2 / math.pi * fs),
                             color='black',
                             edgecolor='none',
                             alpha=0.15, label='0.95 quantile')
    axs[3, col].grid(linestyle='--', alpha=0.3, which='both')

    # Plot FHC
    file_name = path + f'fhc_{mag_name}_{mc}.mat'
    fhc_results = scipy.io.loadmat(file_name)
    fhc_ts = np.squeeze(fhc_results['segmentTimes'])
    fhc_estimates = np.squeeze(fhc_results['pitchTrack'])
    axs[4, col].plot(fhc_ts, fhc_estimates, c='black', label='IF estimate')
    axs[4, col].grid(linestyle='--', alpha=0.3, which='both')

    # Plot fastf0nls
    fastf0nls_results = np.load(path + f'fastf0nls_{mag_name}_{mc}.npz')
    fastf0nls_ts = fastf0nls_results['windows_times']
    fastf0nls_estimates = fastf0nls_results['smoothed_f0estimates']
    axs[5, col].plot(fastf0nls_ts, fastf0nls_estimates, c='black', label='IF estimate')
    axs[5, col].grid(linestyle='--', alpha=0.3, which='both')


# Legends
axs[0, 0].legend(loc='lower center', ncol=2, fontsize=12, scatterpoints=5)
for k in range(1, 6):
    axs[k, 0].legend(loc='lower left', ncol=1, fontsize=12)

# X title
axs[0, 0].set_title(r'$\alpha(t)=1$')
axs[0, 1].set_title(r'$\alpha(t) = \mathrm{e}^{-0.3 \, t}$')
axs[0, 2].set_title(r'$t\mapsto\alpha(t)$ random process')

# X labels
for i in range(3):
    axs[-1, i].set_xlabel('$t$')

# Y labels
axs[0, 0].set_ylabel('Chirp')
axs[1, 0].set_ylabel('GHFS MLE')
axs[2, 0].set_ylabel('GHFS MLE on (2)')
axs[3, 0].set_ylabel('KPT MLE')
axs[4, 0].set_ylabel('FHC')
axs[5, 0].set_ylabel('FastNLS')

plt.tight_layout(pad=0.1)
plt.savefig('./estimation-gallery.pdf')
plt.show()
