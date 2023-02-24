"""
Plot samples drawn from the chirp prior. This generated Figure 4 in the paper.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from chirpgp.tools import simulate_sde
from chirpgp.models import model_chirp, disc_chirp_lcd
from chirpgp.models import g
from jax.config import config

config.update("jax_enable_x64", True)

plt.rcParams.update({
    'text.usetex': True,
    'font.family': "san-serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

# Times
dt = 0.001
T = 10000
ts = jnp.linspace(dt, dt * T, T)

# Random key
key = jax.random.PRNGKey(2021)
keys = jax.random.split(key, 4)

# Model parameters, ordered by lam, b, delta, ell, sigma
# Feel free to try other combinations of parameters
params_all = ((jnp.array(0.1), 0.2, 0.1, 0.5, 3.),
              (jnp.array(0.1), 0.5, 0.1, 1., 1.),
              (jnp.array(0.3), 0.5, 0.1, 1., 3.),
              (jnp.array(0.3), 0.01, 0.1, 1., 3.))

# Simulate and plot
fig, axs = plt.subplots(nrows=2, ncols=4, sharex='col', sharey='row', figsize=(16, 6))

for params, key, i in zip(params_all, keys, range(4)):
    lam, b, delta, ell, sigma = params
    drift, dispersion, m0, P0, H = model_chirp(lam, b, ell, sigma, delta)
    m_and_cov = disc_chirp_lcd(lam, b, ell, sigma)
    traj = simulate_sde(m_and_cov, m0, P0, dt, T, key, const_diag_cov=False)

    axs[0][i].plot(ts, traj @ H)
    axs[0][i].grid(linestyle='--', alpha=0.3, which='both')
    axs[0][i].set_title(rf'$\lambda$={lam}, $b$={b}, $\ell$={ell}, $\sigma$={sigma}', fontsize=17)

    axs[1][i].plot(ts, g(traj[:, 2]))
    axs[1][i].grid(linestyle='--', alpha=0.3, which='both')

    axs[1][i].set_xlabel('$t$')

axs[0][0].set_ylabel('Chirp sample $H \, U(t)$')
axs[1][0].set_ylabel('Frequency sample $f(t)$')

plt.tight_layout(pad=0.1)
plt.savefig('./chirp-sde-samples.pdf')
plt.show()
