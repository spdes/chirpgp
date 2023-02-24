"""
Compute the GHF error of the model.
"""
import argparse
import math
import jax
import jax.scipy
import jax.numpy as jnp
import numpy as np
from functools import partial
from chirpgp.models import model_chirp, disc_chirp_lcd
from chirpgp.quadratures import SigmaPoints
from chirpgp.filters_smoothers import sgp_filter
from jax.config import config

config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('-lam', type=float, help='lam')
parser.add_argument('-b', type=float, help='b')
parser.add_argument('-delta', type=float, help='delta')
parser.add_argument('-ell', type=float, help='ell')
parser.add_argument('-sigma', type=float, help='sigma')
parser.add_argument('-Xi', type=float, help='Xi')

args = parser.parse_args()

lam, b, delta, ell, sigma, Xi = args.lam, args.b, args.delta, args.ell, args.sigma, args.Xi
_, _, m0, P0, H = model_chirp(lam, b, ell, sigma, delta)
m_and_cov = disc_chirp_lcd(lam, b, ell, sigma)

dt = 0.01
T = 500
ts = jnp.linspace(dt, T * dt, T)

chol_P0 = jax.lax.linalg.cholesky(P0)
_, state_cov = m_and_cov(jnp.zeros((4,)), dt)
chol_state_cov = jax.lax.linalg.cholesky(state_cov)


@partial(jax.vmap, in_axes=[0])
def simulate_sde_vmap(key):
    x0 = m0 + chol_P0 @ jax.random.normal(key=key, shape=m0.shape)
    key, _ = jax.random.split(key)
    rnds_x = jax.random.normal(key=key, shape=(T, 4))
    rnds_y = jax.random.normal(key=key, shape=(T,))

    def scan_body(carry, elem):
        x = carry
        rnd_x, rnd_y = elem

        m, _ = m_and_cov(x, dt)
        x = m + chol_state_cov @ rnd_x
        y = jnp.dot(H, x) + math.sqrt(Xi) * rnd_y
        return x, (x, y)

    _, (xs, ys) = jax.lax.scan(scan_body, x0, (rnds_x, rnds_y))
    return xs, ys


num_mcs = 1000000

key = jax.random.PRNGKey(666)
keys = jax.random.split(key, num_mcs)

xss, yss = simulate_sde_vmap(keys)
chirps, vs = xss[:, :, 1], xss[:, :, 2]

sgps = SigmaPoints.gauss_hermite(d=4, order=3)


@jax.jit
@partial(jax.vmap, in_axes=(0, ))
def filtering(measurements):
    mfs, _, _ = sgp_filter(m_and_cov, sgps, H, Xi, m0, P0, dt, measurements)
    return mfs[:, 1], mfs[:, 2]


# Trigger JIT
_ = filtering(jnp.ones((2, 2)))

# Do filtering
estimate_chirps, estimate_vs = filtering(yss)

# Compute chirp errors
err_mean_chirps = np.mean((estimate_chirps - chirps) ** 2, axis=0)
err_std_chirps = np.std((estimate_chirps - chirps) ** 2, axis=0)

# Compute v-component errors
# Note: it is quite non-trivial to directly compute the errors on the
# frequency due to the bijection.
err_mean_vs = np.mean((estimate_vs - vs) ** 2, axis=0)
err_std_vs = np.std((estimate_vs - vs) ** 2, axis=0)

file_name = f'./results/crlb_ghf_lam_{lam}_b_{b}_Xi_{Xi}.npz'
np.savez(file_name, ts=ts,
         err_mean_chirps=err_mean_chirps, err_std_chirps=err_std_chirps,
         err_mean_vs=err_mean_vs, err_std_vs=err_std_vs)

print('Results saved in ' + file_name)
