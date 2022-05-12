"""
Generates Figure 4 in the paper. Plot some demonstrative estimation results.

Note that you need to run the scripts in ../tetralith to get the results first.
"""
import math
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from chirpgp.models import g
from chirpgp.quadratures import gaussian_expectation
from chirpgp.toymodels import meow_freq, gen_chirp, constant_mag, damped_exp_mag, random_ou_mag
from jax.config import config

config.update("jax_enable_x64", True)

mc = 1

path = '../tetralith/'

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "san-serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

fig, axs = plt.subplots(nrows=5, ncols=3, sharex='col', sharey='row', figsize=(15, 12))

# Produce the signal and measurements at that MC seed
dt = 0.001
T = 3141
ts = jnp.linspace(dt, dt * T, T)

key = jnp.asarray(np.load(path + 'rnd_keys.npy')[mc])
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
    for i in range(1, 5):
        axs[i, col].plot(ts, true_freq_func(ts), c='tab:blue', linestyle='--', label='True IF')

    # Plot GHFS IF estimates
    smoothing_mean = np.load(path + 'results/' + f'ghfs_mle_{mag_name}_{mc}.npz')['smoothing_mean']
    smoothing_cov = np.load(path + 'results/' + f'ghfs_mle_{mag_name}_{mc}.npz')['smoothing_cov']
    estimated_freqs_mean = gaussian_expectation(ms=smoothing_mean[:, 2],
                                                chol_Ps=jnp.sqrt(smoothing_cov[:, 2, 2]),
                                                func=g, force_shape=True)[:, 0]
    axs[1, col].plot(ts, estimated_freqs_mean, c='black', label='IF posterior mean')
    axs[1, col].fill_between(ts,
                             g(smoothing_mean[:, 2] - 1.96 * jnp.sqrt(smoothing_cov[:, 2, 2])),
                             g(smoothing_mean[:, 2] + 1.96 * jnp.sqrt(smoothing_cov[:, 2, 2])),
                             color='black',
                             edgecolor='none',
                             alpha=0.15, label='0.95 confidence')
    axs[1, col].grid(linestyle='--', alpha=0.3, which='both')

    # Plot GHFS (La Scala) IF estimates
    smoothing_mean = np.load(path + 'results/' + f'lascala_ghfs_mle_{mag_name}_{mc}.npz')['smoothing_mean']
    smoothing_cov = np.load(path + 'results/' + f'lascala_ghfs_mle_{mag_name}_{mc}.npz')['smoothing_cov']
    estimated_freqs_mean = gaussian_expectation(ms=smoothing_mean[:, 2],
                                                chol_Ps=jnp.sqrt(smoothing_cov[:, 2, 2]),
                                                func=g, force_shape=True)[:, 0]
    axs[2, col].plot(ts, estimated_freqs_mean, c='black', label='IF posterior mean')
    axs[2, col].fill_between(ts,
                             g(smoothing_mean[:, 2] - 1.96 * jnp.sqrt(smoothing_cov[:, 2, 2])),
                             g(smoothing_mean[:, 2] + 1.96 * jnp.sqrt(smoothing_cov[:, 2, 2])),
                             color='black',
                             edgecolor='none',
                             alpha=0.15, label='0.95 confidence')
    axs[2, col].grid(linestyle='--', alpha=0.3, which='both')

    # Plot spectrogram IF estimates
    estimated_freqs = np.load(path + 'results/' + f'mean_spectrogram_{mag_name}_{mc}.npz')['estimates']
    segment_times = np.load(path + 'results/' + f'mean_spectrogram_{mag_name}_{mc}.npz')['segment_times']
    axs[3, col].plot(segment_times, estimated_freqs, c='black', label='IF estimate')
    axs[3, col].grid(linestyle='--', alpha=0.3, which='both')

    # Plot ANF IF estimates
    estimated_freqs = np.load(path + 'results/' + f'anf_{mag_name}_{mc}.npz')['estimates']
    axs[4, col].plot(ts, estimated_freqs, c='black', label='IF estimate')
    axs[4, col].grid(linestyle='--', alpha=0.3, which='both')

# Legends
axs[0, 0].legend(loc='lower center', ncol=2, fontsize=12, scatterpoints=5)
axs[1, 0].legend(loc='lower left', ncol=1, fontsize=12)
axs[2, 0].legend(loc='lower left', ncol=1, fontsize=12)
axs[3, 0].legend(loc='lower left', ncol=1, fontsize=12)
axs[4, 0].legend(loc='lower left', ncol=1, fontsize=12)

# X title
axs[0, 0].set_title(r'$\alpha(t)=1$')
axs[0, 1].set_title(r'$\alpha(t) = \mathrm{e}^{-0.3 \, t}$')
axs[0, 2].set_title(r'$\alpha(t)$ random process')

# X labels
for i in range(3):
    axs[-1, i].set_xlabel('$t$')

# Y labels
axs[0, 0].set_ylabel('Chirp')
axs[1, 0].set_ylabel('GHFS MLE')
axs[2, 0].set_ylabel('GHFS MLE on (2)')
axs[3, 0].set_ylabel('Spectrogram')
axs[4, 0].set_ylabel('ANF')

plt.tight_layout(pad=0.1)
plt.savefig('./estimation-gallery.pdf')
plt.show()
