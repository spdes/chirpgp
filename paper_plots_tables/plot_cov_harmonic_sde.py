"""
Plot the covariance function of harmonic SDE. This generated Figure 2 in the paper.
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
from chirpgp.cov_funcs import vmap_cov_harmonic_sde
from jax.config import config

config.update("jax_enable_x64", True)

# Times
dt = 0.01
T = 1000
ts = jnp.linspace(dt, T * dt, T)

# Parameters
f = 0.5
lam = 0.1
b = 0.5

# Initial covariance
init_cov = b ** 2 / (2 * lam) * jnp.eye(2)

# Obtain covariance matrix
cov_matrix = vmap_cov_harmonic_sde(ts, ts, init_cov, f, lam, b)

# Plot
path_figs = './cov-harmonic-sde.png'

plt.rcParams.update({
    'text.usetex': True,
    'font.size': 20})

cax = plt.pcolormesh(ts, ts, cov_matrix[:, :, 1, 1], edgecolors='none', shading='nearest', cmap=plt.cm.Blues_r)
cbar = plt.colorbar(cax, ticks=[-1, 0, 1])
plt.xlabel('$t$')
plt.ylabel("$t'$")
plt.xticks([2., 4., 6., 8., 10.])
plt.yticks([2., 4., 6., 8., 10.])

plt.tight_layout(pad=0.01)
plt.savefig(path_figs, dpi=200)
