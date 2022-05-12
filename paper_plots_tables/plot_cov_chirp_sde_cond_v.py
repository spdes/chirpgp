"""
Plot the covariance function of chirp SDE. This generates Figure 2 in the paper.
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from chirpgp.cov_funcs import approx_cond_cov_chirp_sde
from chirpgp.models import g
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Use float32 to save memory to have more MC runs.
# This script works on a PC with 32G memory.

# Times
dt = 0.01
T = 1000
ts = jnp.linspace(dt, T * dt, T)

# Parameters
lam = 0.1
b = 0.5
delta = 0.1
ell = 1.
sigma = 3.

# Random key and number of MC simulations
key = jax.random.PRNGKey(123)
key = jax.random.split(key, num=100)[91]  # Other demonstrative seeds: 7, 20, 24, 91, 97
num_mcs = 1000

# Obtain trajectory of v and conditional covariance matrix
vs, cov_matrix = approx_cond_cov_chirp_sde(ts, lam, b, ell, sigma, delta, num_mcs, key)
print(cov_matrix.shape)

# Plot
path_figs = './cov-chirp-sde-cond-f.png'

plt.rcParams.update({
    'text.usetex': True,
    'font.size': 18})

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(7, 11), gridspec_kw={'height_ratios': [3, 1]})

img = axs[0].pcolormesh(ts, ts, cov_matrix[:, :, 1, 1], edgecolors='none', shading='nearest', cmap=plt.cm.Blues_r)
axs[0].set_ylabel("$t'$")
axs[0].set_yticks([2., 4., 6., 8., 10.])

axs[1].plot(ts, g(vs[:, 0]))
axs[1].set_ylabel(r'$f(t)$')
axs[1].set_xlabel('$t$')
axs[1].set_xticks([2., 4., 6., 8., 10.])
axs[1].grid(linestyle='--', alpha=0.3, which='both')

# Adjust and add colorbar
# https://matplotlib.org/stable/gallery/axes_grid1/demo_colorbar_with_inset_locator.html
axs[0].set_aspect('equal')
axins = inset_axes(axs[0],
                   width="2%",
                   height="100%",
                   loc='lower left',
                   bbox_to_anchor=(1.02, 0., 1, 1),
                   bbox_transform=axs[0].transAxes,
                   borderpad=0,
                   )
cbar = fig.colorbar(img, ticks=[-1, 0, 1], cax=axins)

plt.subplots_adjust(left=0.09, hspace=0., top=0.8)

# NOTE: Matplotlib seems to have a bug here. For some reasons, result from plt.savefig is different from plt.show.
plt.savefig(path_figs, bbox_inches='tight', dpi=200, transparent=True)
