import math
import jax
import jax.numpy as jnp
from chirpgp.filters_smoothers import ekf, eks, cd_ekf, cd_eks
from chirpgp.tools import simulate_sde
import tme.base_jax as tme
import numpy.testing as npt
from jax.config import config

config.update("jax_enable_x64", True)

dim_x = 3
kappa = 10.
lam = 28.
mu = 2.
H = jnp.array([1., 0., 0.])
R = 2.


def drift(u):
    return jnp.array([kappa * (u[1] - u[0]),
                      u[0] * (lam - u[2]) - u[1],
                      u[0] * u[1] - mu * u[2]])


def dispersion(_):
    return 5. * jnp.eye(3)


@jax.jit
def tme_m_cov(u, dt):
    return tme.mean_and_cov(u, dt, drift, dispersion, order=2)


# Trigger
tme_m_cov(jnp.ones((dim_x,)), 0.001)

dt = 0.001
T = 2000
ts = jnp.linspace(dt, dt * T, T)
Xi = 2.

m0 = jnp.zeros((dim_x,))
P0 = 1. * jnp.eye(dim_x)

key = jax.random.PRNGKey(666)

trajectory = simulate_sde(tme_m_cov, m0, P0, dt, T, key, const_diag_cov=False)

key, _ = jax.random.split(key)
ys = trajectory[:, 0] + math.sqrt(Xi) * jax.random.normal(key, shape=(ts.size,))

ekf_results = ekf(tme_m_cov, H, Xi, m0, P0, dt, ys)
eks_results = eks(tme_m_cov, ekf_results[0], ekf_results[1], dt)

cd_ekf_results = cd_ekf(drift, dispersion, H, Xi, m0, P0, dt, ys)
cd_eks_results = cd_eks(drift, dispersion, cd_ekf_results[0], cd_ekf_results[1], dt)

npt.assert_allclose(ekf_results[0], cd_ekf_results[0], rtol=0.2)
npt.assert_allclose(ekf_results[1], cd_ekf_results[1], rtol=0.21)
npt.assert_allclose(ekf_results[2], cd_ekf_results[2], rtol=1e-5, atol=1e-2)
