# Copyright (C) 2021 Zheng Zhao and the chirpgp contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Compute the covariance functions of the chirp SDEs.

Note that t0 is assumed to be zero.
"""
import math
import jax
import jax.numpy as jnp
from chirpgp.models import jndarray, model_chirp, disc_chirp_lcd, disc_chirp_lcd_cond_v, disc_m32
from chirpgp.tools import simulate_sde, simulate_function_parametrised_sde
from functools import partial
from typing import Callable, Union, Tuple


def transition_harmonic_sde(t: float, s: float, lam: float, w: float) -> jndarray:
    r"""Transition semigroup of harmonic SDE.

    .. math::

        F(t, s) = [ ..., ...
                    ..., ...] exp(-\lambda \, (t - s))

    Parameters
    ----------
    t : float
        Time t.
    s : float
        Time s.
    lam : float
        Damping parameter.
    w : float
        Angular frequency, that is, w = 2 pi f.

    Returns
    -------
    jnp.ndarray
    """
    dt = t - s
    return jnp.array([[jnp.cos(dt * w), -jnp.sin(dt * w)],
                      [jnp.sin(dt * w), jnp.cos(dt * w)]]) * jnp.exp(-lam * dt)


def marginal_cov_harmonic_sde(t: float, s: float, cov_xs: jndarray, lam: float, b: float, w: float) -> jndarray:
    r"""Marginal covariance of the harmonic SDE in Equation 4 from any starting time s.

    .. math ::

        \mathrm{Cov}[X(t)] = F(t, s) Cov[X(s)] F(t, s)^T + \Sigma(t, s)

    starting from any :math:`s`.

    Parameters
    ----------
    t : float
        Time t.
    s : float
        Time s.
    cov_xs : jnp.ndarray
        The covariance of X at time s.
    lam : float
        Damping parameter.
    b : float
        Dispersion.
    w : float
        Angular frequency, that is, w = 2 pi f.

    Returns
    -------
    jnp.ndarray
    """
    transition = transition_harmonic_sde(t, s, lam, w)
    if lam == 0:
        return transition @ cov_xs @ transition.T + b ** 2 * (t - s) * jnp.eye(2)
    else:
        return transition @ cov_xs @ transition.T + b ** 2 / (2 * lam) * (1 - jnp.exp(-2 * lam * (t - s))) * jnp.eye(2)


def cov_harmonic_sde(t1: float, t2: float, cov_xs: jndarray, f: float, lam: float, b: float) -> jndarray:
    r"""Covariance function of the harmonic SDE in Equation 4.

    ..math ::

        \mathrm{Cov}[X(t1), X(t2)]

    Parameters
    ----------
    t1 : float
        Time t1.
    t2 : float
        Time t2
    cov_xs : jnp.ndarray
        The covariance of X at time s.
    f : float
        Frequency
    lam : float
        Damping parameter.
    b : float
        Dispersion.

    Returns
    -------
    jnp.ndarray
    """
    w = 2 * math.pi * f

    def cond_true(x):
        return marginal_cov_harmonic_sde(t1, 0., cov_xs, lam, b, w) @ transition_harmonic_sde(t2, t1, lam, w).T

    def cond_false(x):
        return transition_harmonic_sde(t1, t2, lam, w) @ marginal_cov_harmonic_sde(t2, 0., cov_xs,
                                                                                   lam, b, w)

    return jax.lax.cond(t1 < t2,
                        cond_true,
                        cond_false,
                        jnp.array([0.]))


vmap_marginal_cov_harmonic_sde = jax.vmap(marginal_cov_harmonic_sde,
                                          in_axes=[0, None, None, None, None, None])
vmap_cov_harmonic_sde = jax.vmap(jax.vmap(cov_harmonic_sde,
                                          in_axes=[0, None, None, None, None, None]),
                                 in_axes=[None, 0, None, None, None, None])


def _monte_carlo_cov_of_sde(gen_trajectory: Callable[[jndarray], jndarray],
                            T: int,
                            key: jndarray,
                            num_mcs: int):
    keys = jax.random.split(key, num_mcs)

    trajs = gen_trajectory(keys)

    mc_means = jnp.mean(trajs, axis=0)

    adhoc_vmap_outer = jax.vmap(jnp.outer, in_axes=[0, 0])

    @partial(jax.vmap, in_axes=[None, 0])
    @partial(jax.vmap, in_axes=[0, None])
    def cov_func(k1: int, k2: int):
        r"""Cov[X(t_k1), X(t_k2)] = E[(X(t_k1) - m(t_k1)) (X(t_k2) - m(t_k2))^T] \approx MC"""
        return jnp.sum(adhoc_vmap_outer(trajs[:, k1] - mc_means[k1], trajs[:, k2] - mc_means[k2]), axis=0) / (T - 1)

    ks = jnp.arange(0, T, 1)
    return cov_func(ks, ks)


def approx_cov_chirp_sde(ts: jndarray, lam: float, b: float, ell: float, sigma: float, delta: float, num_mcs: int,
                         key: jndarray) -> jndarray:
    """Approximate the covariance function of the chirp SDE by Monte Carlo simulation.

    Parameters
    ----------
    ts

    Returns
    -------

    """
    _, _, m0, P0, _ = model_chirp(lam, b, ell, sigma, delta)
    m_and_cov = disc_chirp_lcd(lam, b, ell, sigma)

    dt = jnp.diff(ts)[0]
    T = ts.size

    @partial(jax.vmap, in_axes=[0])
    def gen_trajectory(_key):
        return simulate_sde(m_and_cov, m0, P0, dt, T, _key, const_diag_cov=False)

    return _monte_carlo_cov_of_sde(gen_trajectory, T, key, num_mcs)


def approx_cond_cov_chirp_sde(ts: jndarray, lam: Union[jndarray, float], b: float, ell: float, sigma: float,
                              delta: float, num_mcs: int,
                              key: jndarray) -> Tuple[jndarray, jndarray]:
    _, _, m0, P0, _ = model_chirp(lam, b, ell, sigma, delta)

    # Simulate a trajectory of V.
    m_and_cov_of_v = disc_m32(ell, sigma)

    dt = jnp.diff(ts)[0]
    T = ts.size

    vs = simulate_sde(m_and_cov_of_v, m0[2:], P0[2:, 2:], dt, T, key, const_diag_cov=False)

    # Simulate MC samples of X | V then compute cond cov.
    m_and_cov_of_x = disc_chirp_lcd_cond_v(lam, b)

    @partial(jax.vmap, in_axes=[0])
    def gen_trajectory(_key):
        return simulate_function_parametrised_sde(m_and_cov_of_x, vs[:, 0], m0[:2], P0[:2, :2], dt, T, _key,
                                                  const_diag_cov=True)

    key, _ = jax.random.split(key)
    return vs, _monte_carlo_cov_of_sde(gen_trajectory, T, key, num_mcs)


def psd_chirp_sde(num_mcs: int):
    # TODO: Apply PSD estimation then MC
    pass
