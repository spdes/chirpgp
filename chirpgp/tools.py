# Copyright (C) 2022 Zheng Zhao and the chirpgp contributors
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

import jax.numpy as jnp
import jax.scipy
from functools import partial
from typing import Tuple, Callable, Union

__all__ = ['lti_sde_to_disc',
           'simulate_lgssm',
           'simulate_sde',
           'simulate_function_parametrised_sde',
           'fwd_transformed_pdf',
           'chol_partial_const_diag',
           'rmse']

jndarray = jnp.ndarray


def _smart_outer(z: jndarray):
    dim = z.ndim
    if dim == 0:
        return (z ** 2).reshape(1, 1)
    elif dim == 1:
        return jnp.outer(z, z)
    elif dim == 2:
        return z @ z.T
    else:
        raise ValueError(f"The shape {dim} of operand is not recognizable.")


def lti_sde_to_disc(A: jndarray, B: jndarray, dt: float) -> Tuple[jndarray, jndarray]:
    r"""Discretise LTI SDE. Axelsson and Gustafsson 2015

    .. math::

        d X(t) = A \, X(t) dt + B dW(t),

    to

    .. math::

        X(t_k) = F \, X(t_{k-1}) + Q_{k-1},  Q_{k-1} \sim N(0, \Sigma)

    Parameters
    ----------
    A : jnp.ndarray
        Drift matrix.
    B : jnp.ndarray
        Dispersion matrix.
    dt : float
        Time interval.

    Returns
    -------
    jndarray, jndarray
        Transition mean and covariance.

    """
    dim = A.shape[0]

    F = jax.scipy.linalg.expm(A * dt)
    phi = jnp.vstack([jnp.hstack([A, _smart_outer(B)]), jnp.hstack([jnp.zeros_like(A), -A.T])])
    AB = jax.scipy.linalg.expm(phi * dt) @ jnp.vstack([jnp.zeros_like(A), jnp.eye(dim)])
    Sigma = AB[0:dim, :] @ F.T
    return F, Sigma


def simulate_lgssm(F: jndarray, Sigma: jndarray,
                   x0: jndarray, T: int,
                   key: jndarray) -> jndarray:
    """Evenly simulate a trajectory from a linear Gaussian state-space model

    Parameters
    ----------
    F : jnp.ndarray (d, d)
        Transition matrix.
    Sigma : jnp.ndarray (d, d)
        Transition covariance
    x0 : jnp.ndarray (d, )
        Initial value
    T : int
        Number of measurements.
    key : jnp.ndarray
        PRNkey.

    Returns
    -------
    jnp.ndarray (T, d)
        Trajectory.
    """
    d = x0.size
    rnds = jax.random.normal(key=key, shape=(T, d))
    chol = jax.scipy.linalg.cholesky(Sigma, lower=True)

    def scan_body(carry, elem):
        x = carry
        rnd = elem

        x = F @ x + chol @ rnd
        return x, x

    _, trajectory = jax.lax.scan(scan_body, x0, rnds)
    return trajectory


def simulate_sde(m_and_cov: Callable[[jndarray, float], Tuple[jndarray, jndarray]],
                 m0: jndarray, P0: jndarray, dt: float, T: int, key: jndarray,
                 const_diag_cov: bool = False) -> jndarray:
    """Simulate a trajectory from an SDE (given by m_and_cov) using Gaussian increment approximation.

    Parameters
    ----------
    m_and_cov : Callable
        Function that returns the conditional mean and covariance of the SDE.
    m0 : jnp.ndarray (d, )
        Initial mean.
    P0 : jnp.ndarray (d, d)
        Initial covariance.
    dt : float
        Time interval.
    T : int
        Number of measurements.
    key : jnp.ndarray
        PRNkey.
    const_diag_cov
        Set it true means that the cov returned from :code:`m_and_cov` is diagonal matrix.

    Returns
    -------
    jnp.ndarray (T, d)
        Trajectory.

    Notes
    -----
    This implementation assumes that the dimension of W is the same as with the state.
    """
    dim = m0.size

    x0 = m0 + jax.scipy.linalg.cholesky(P0, lower=True) @ jax.random.normal(key=key, shape=(dim,))

    key, _ = jax.random.split(key)
    dws = jax.random.normal(key=key, shape=(T, dim))

    def scan_body(carry, elem):
        x = carry
        dw = elem

        m, cov = m_and_cov(x, dt)
        if const_diag_cov:
            chol = jnp.sqrt(cov)
        else:
            chol = jnp.linalg.cholesky(cov)
        x = m + chol @ dw
        return x, x

    _, traj = jax.lax.scan(scan_body, x0, dws)
    return traj


def simulate_sde_init(m_and_cov: Callable[[jndarray, float], Tuple[jndarray, jndarray]],
                      x0: jndarray, dt: float, T: int, key: jndarray,
                      const_diag_cov: bool = False) -> jndarray:
    dim = x0.shape[0]

    key, _ = jax.random.split(key)
    dws = jax.random.normal(key=key, shape=(T, dim))

    def scan_body(carry, elem):
        x = carry
        dw = elem

        m, cov = m_and_cov(x, dt)
        if const_diag_cov:
            chol = jnp.sqrt(cov)
        else:
            chol = jnp.linalg.cholesky(cov)
        x = m + chol @ dw
        return x, x

    _, traj = jax.lax.scan(scan_body, x0, dws)
    return traj


def simulate_function_parametrised_sde(m_and_cov: Callable, vs: jndarray,
                                       m0: jndarray, P0: jndarray, dt: float, T: int, key: jndarray,
                                       const_diag_cov: bool = False) -> jndarray:
    """Almost the same as with :code:`simulate_sde` except that :code:`m_and_cov` now accepts inputs that depends on
    time.
    """
    dim = m0.size

    x0 = m0 + jax.scipy.linalg.cholesky(P0, lower=True) @ jax.random.normal(key=key, shape=(dim,))

    key, _ = jax.random.split(key)
    dws = jax.random.normal(key=key, shape=(T, dim))

    def scan_body(carry, elem):
        x = carry
        v, dw = elem

        m, cov = m_and_cov(x, v, dt)
        if const_diag_cov:
            chol = jnp.sqrt(cov)
        else:
            chol = jnp.linalg.cholesky(cov)
        x = m + chol @ dw
        return x, x

    _, traj = jax.lax.scan(scan_body, x0, (vs, dws))
    return traj


def fwd_transformed_pdf(pdf_x: Callable[[float], float],
                        g_inv: Callable[[float], float]) -> Callable[[jndarray], jndarray]:
    r"""PDF of transformed random variable.

    .. math::

        Y = g(X), \quad X \sim p_X(x),

    where :math::`g` is strictly monotone. Then

    .. math::

        p_Y(y) = p_X(g^{-1}(y)) \, \abs{\frac{d g^{-1}}{dy}(y)}.

    Parameters
    ----------
    pdf_x : Callable
        PDF of random variable X.
    g_inv : Callable
        Inverse function of g.

    Returns
    -------
    Callable
        PDF of Y=g(X).

    """

    @partial(jax.vmap, in_axes=[0])
    def pdf_y(y):
        return pdf_x(g_inv(y)) * jnp.abs(jax.grad(g_inv)(y))

    return pdf_y


def chol_partial_const_diag(a: jndarray, n: int, *args, **kwargs) -> jndarray:
    """Cholesky decomposition for matrix with constant diag on its n-th rank.

    Parameters
    ----------
    a : jnp.ndarray
        A matrix.
    n : int
        Rank.

    Returns
    -------
    jnp.ndarray
        L of L L^T = a or L^T L = a.
    """
    return jax.scipy.linalg.block_diag(jnp.sqrt(a[:n, :n]), jax.scipy.linalg.cholesky(a[n:, n:], *args, **kwargs))


def rmse(x1: jndarray, x2: jndarray, reduce_sum: bool = True) -> Union[float, jndarray]:
    """Root mean square error.

    Parameters
    ----------
    x1 : jnp.ndarray (T, d)
    x2 : jnp.ndarray (T, d)
    reduce_sum : bool, default=True
        Letting this be :code:`True` will take a sum of the RMSEs from all dimensions d.
    """
    val = jnp.sqrt(jnp.mean((x1 - x2) ** 2, axis=0))
    if reduce_sum:
        return jnp.sum(val)
    else:
        return val
