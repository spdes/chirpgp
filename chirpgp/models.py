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

"""
Implementations of chirp-IF estimation models.
"""
import math
import jax.lax
import jax.scipy
import jax.numpy as jnp
import tme.base_jax as tme
from typing import Tuple, Callable, Union

__all__ = ['jndarray',
           'g',
           'g_inv',
           'model_chirp',
           'model_lascala',
           'disc_chirp_lcd',
           'disc_chirp_lcd_cond_v',
           'disc_chirp_tme',
           'disc_chirp_euler_maruyama',
           'disc_model_lascala_lcd',
           'disc_m32',
           'build_chirp_model',
           'build_lascala_model']

jndarray = jnp.ndarray


def g(x): return jnp.log(jnp.exp(x) + 1.)


def g_inv(x): return jnp.log(jnp.exp(x) - 1.)


def _stationary_cov_m32(ell: float, sigma: float):
    return jnp.array([[sigma ** 2, 0.],
                      [0., (math.sqrt(3) / ell) ** 2 * sigma ** 2]])


def _m32_solution(ell: float, sigma: float, dt: float) -> Tuple[jndarray, jndarray]:
    """Explicit solution of Matern 3/2 SDE. Derived by symbolic computation.
    """
    gamma = math.sqrt(3) / ell
    eta = dt * gamma
    beta = sigma ** 2 * jnp.exp(-2 * eta)

    transition = jnp.array([[1 + eta, dt],
                            [-dt * gamma ** 2, 1 - eta]]) * jnp.exp(-eta)
    Sigma = jnp.array([[sigma ** 2 - beta * (2 * eta + 2 * eta ** 2 + 1), 2 * dt ** 2 * gamma ** 3 * beta],
                       [2 * dt ** 2 * gamma ** 3 * beta,
                        gamma ** 2 * (sigma ** 2 + beta * (2 * eta - 2 * eta ** 2 - 1))]])
    return transition, Sigma


def model_chirp(lam: float,
                b: float,
                ell: float,
                sigma: float,
                delta: float) -> Tuple[Callable[[jndarray], jndarray],
                                       Callable[[jndarray], jndarray],
                                       jndarray, jndarray, jndarray]:
    r"""The proposed chirp and IF model in Equation 14.

    Parameters
    ----------
    lam : float
        Damping factor.
    b : float
        Chirp dispersion constant.
    ell : float
        IF prior length scale.
    sigma : float
        IF prior magnitude scale.
    delta : float
        Chirp initial covariance :math:`P_0^X = \delta \, eye(2)`.

    Returns
    -------
    Callable, Callable, jnp.ndarray, jnp.ndarray, jnp.ndarray
        Drift function, dispersion, initial mean, initial covariance, and measurement H.
    """

    def drift(u: jndarray) -> jndarray:
        w = 2 * math.pi * g(u[2])
        gamma = math.sqrt(3) / ell
        return jnp.array([[-lam, -w, 0., 0.],
                          [w, -lam, 0., 0.],
                          [0., 0., 0., 1.],
                          [0., 0., -(gamma ** 2), -2 * gamma]]) @ u

    def dispersion(_: jndarray) -> jndarray:
        return jnp.diag(jnp.array([b, b, 0., 2 * sigma * (math.sqrt(3) / ell) ** 1.5]))

    m0 = jnp.array([0., 1., 0., 0.])
    P0 = jax.scipy.linalg.block_diag(delta, delta, _stationary_cov_m32(ell, sigma))

    H = jnp.array([0., 1., 0., 0.])
    return drift, dispersion, m0, P0, H


def model_lascala(ell: float,
                  sigma: float,
                  delta: float) -> Tuple[Callable[[jndarray], jndarray], Callable[[jndarray], jndarray],
                                         jndarray, jndarray, jndarray]:
    r"""The chirp model used by Synder and La Scala. This function is for pedagogical purpose only.
    See disc_model_old_lcd for how to use this model in practice.

    The continuous version of their models can be written as follows.

    .. math::

        d [V(t)         [0                      1]           [ 0 ]
           dV(t)/dt] =  [-3/ell^2  -2sqrt{3}/ell ] V(t) dt + [...] dW(t)
        Y_k = \sin(\phi_0 + 2 pi \int^{t_k}_0 g(V(s)) ds) + \xi

    or equivalently,

    .. math::

        d \phi(t) = g(V(t)) dt
        d [V(t)         [0                      1]           [ 0 ]
           dV(t)/dt] =  [-3/ell^2  -2sqrt{3}/ell ] V(t) dt + [...] dW(t)
        Y_k = \sin(\phi_0 + 2 pi \phi(t)) + \xi

    or

    .. math::

        d X(t) =        [0             -2 pi g(V(t))
                         2 pi g(V(t))        0      ]     X(t) dt
        d [V(t)         [0                      1]                  [ 0 ]
           dV(t)/dt] =  [-3/ell^2  -2sqrt{3}/ell ]        V(t) dt + [...] dW(t)
        Y_k = [0 1 0 0] X(t) + \xi

    The last one is almost the same as with :code:`model_chirp`, except that the damping and dispersion are not present.

    This function implements the last one in order to best respect the original authors' ideas.

    Parameters
    ----------
    ell : float
        IF prior length scale.
    sigma : float
        IF prior magnitude scale.
    delta : float
        Initial covariance coefficient of chirp.

    References
    ----------
    The state-variable approach to analog communication theory. 1968, Donald L. Snyder.

    Conditions for stability of the extended Kalman filter and their application to the frequency tracking problem.
    1995, B. La Scala.

    On the parametrization and design of an extended Kalman filter frequency tracker. 2000. S. Bittanti and S. Savaresi.

    Notes
    -----
    Please also note that I am putting a smooth Matern prior on IF for the sake of experiment consistency, while in
    their original papers it is an Ornstein--Uhlenbeck process which is non-smooth.

    Finally note that I have a positive transformation function :math:`g` to enforce positive frequency. This was not
    used in their original papers either.
    """

    def drift(u: jndarray) -> jndarray:
        w = 2 * math.pi * g(u[2])
        gamma = math.sqrt(3) / ell
        return jnp.array([[0., -w, 0., 0.],
                          [w, 0., 0., 0.],
                          [0., 0., 0., 1.],
                          [0., 0., -(gamma ** 2), -2 * gamma]]) @ u

    def dispersion(_: jndarray) -> jndarray:
        return jnp.diag(jnp.array([0., 0., 0., 2 * sigma * (math.sqrt(3) / ell) ** 1.5]))

    m0 = jnp.array([0., 1., 0., 0.])
    P0 = jax.scipy.linalg.block_diag(delta, delta, _stationary_cov_m32(ell, sigma))

    H = jnp.array([0., 1., 0., 0.])
    return drift, dispersion, m0, P0, H


def disc_chirp_lcd(lam: Union[float, jndarray],
                   b: float,
                   ell: float,
                   sigma: float) -> Callable[[jndarray, float], Tuple[jndarray, jndarray]]:
    """Locally conditional discretisation of the chirp model.

    Parameters
    ----------
    lam : float
        Damping factor.
    b : float
        Chirp dispersion constant.
    ell : float
        IF prior length scale.
    sigma : float
        IF prior magnitude scale.

    Returns
    -------
    Callable
        A function that returns the conditional mean and covariance.

    References
    ----------
    Page 77. Zheng Zhao. State-space deep Gaussian processes with applications. Aalto University, 2021.

    Notes
    -----
    In order to use this in `jax.scipy.minimize`, the param `lam` must be a jax type, otherwise `jax.cond` works no.
    """

    def m_and_cov(u: jndarray, dt: float) -> Tuple[jndarray, jndarray]:
        w = 2 * math.pi * g(u[2])
        blk_harmonic = jnp.array([[jnp.cos(dt * w), -jnp.sin(dt * w)],
                                  [jnp.sin(dt * w), jnp.cos(dt * w)]]) * jnp.exp(-lam * dt)
        blk_m32_m, blk_m32_Sigma = _m32_solution(ell, sigma, dt)

        cond_m = jax.scipy.linalg.block_diag(blk_harmonic, blk_m32_m) @ u
        Sigma = jax.lax.cond(lam == 0.,
                             lambda _: jax.scipy.linalg.block_diag(b ** 2 * dt, b ** 2 * dt, blk_m32_Sigma),
                             lambda _: jax.scipy.linalg.block_diag(
                                 b ** 2 / (2 * lam) * (1 - jnp.exp(-2 * lam * dt)),
                                 b ** 2 / (2 * lam) * (1 - jnp.exp(-2 * lam * dt)),
                                 blk_m32_Sigma),
                             1.)
        return cond_m, Sigma

    return m_and_cov


def disc_chirp_lcd_cond_v(lam: float, b: float) -> Callable[[jndarray, float, float], Tuple[jndarray, jndarray]]:
    """Locally conditional discretisation of the chirp model by conditioning on a trajectory of V.
    """

    def m_and_cov(u: jndarray, v: float, dt: float) -> Tuple[jndarray, jndarray]:
        w = 2 * math.pi * g(v)
        cond_m = jnp.array([[jnp.cos(dt * w), -jnp.sin(dt * w)],
                            [jnp.sin(dt * w), jnp.cos(dt * w)]]) * jnp.exp(-lam * dt) @ u

        Sigma = jax.lax.cond(lam == 0,
                             lambda _: jnp.eye(2) * b ** 2 * dt,
                             lambda _: jnp.eye(2) * b ** 2 / (2 * lam) * (1 - jnp.exp(-2 * lam * dt)),
                             1.)
        return cond_m, Sigma

    return m_and_cov


def disc_chirp_euler_maruyama():
    """It is not recommended to use Euler--Maruyama on this chirp model due to stiffness, hence, not implemented.
    """
    return NotImplemented


def disc_chirp_tme(lam: Union[float, jndarray],
                   b: float,
                   ell: float,
                   sigma: float,
                   order: int = 3):
    drift, dispersion, _, _, _ = model_chirp(lam, b, ell, sigma, 1.)

    def m_and_cov(u: jndarray, dt: float) -> Tuple[jndarray, jndarray]:
        return tme.mean_and_cov(u, dt, drift, dispersion, jnp.eye(4), order)

    return m_and_cov


def disc_m32(ell: float, sigma: float) -> Callable[[jndarray, float], Tuple[jndarray, jndarray]]:
    """Exact discretisation of M32 SDE.
    """

    def m_and_cov(u: jndarray, dt: float) -> Tuple[jndarray, jndarray]:
        transition, Sigma = _m32_solution(ell, sigma, dt)
        return transition @ u, Sigma

    return m_and_cov


def disc_model_lascala_lcd(ell: float, sigma: float) -> Callable[[jndarray, float], Tuple[jndarray, jndarray]]:
    """LCD discretisation of the old model :code:`model_old`.
    """

    def m_and_cov(u: jndarray, dt: float) -> Tuple[jndarray, jndarray]:
        w = 2 * math.pi * g(u[2])
        blk_harmonic = jnp.array([[jnp.cos(dt * w), -jnp.sin(dt * w)],
                                  [jnp.sin(dt * w), jnp.cos(dt * w)]])
        blk_m32_m, blk_m32_Sigma = _m32_solution(ell, sigma, dt)

        cond_m = jax.scipy.linalg.block_diag(blk_harmonic, blk_m32_m) @ u
        Sigma = jax.scipy.linalg.block_diag(0., 0., blk_m32_Sigma)

        return cond_m, Sigma

    return m_and_cov


def build_chirp_model(params: jndarray) -> Tuple[Callable[[jndarray], jndarray],
                                                 Callable[[jndarray], jndarray],
                                                 Callable[[jndarray, float], Tuple[jndarray, jndarray]],
                                                 jndarray, jndarray, jndarray]:
    """Build chirp model from given parameters for optimisation.

    Parameters
    ----------
    params : jnp.ndarray
        Parameters. From left to right, they are, lam, b, delta, ell, sigma, m0_1.

    Returns
    -------
    Callable, Callable, Callable, jnp.ndarray, jnp.ndarray, jnp.ndarray
        Drift, dispersion, conditional mean and cov, m0, P0, and H. All that are essential to run filter and smoother.
    """
    lam, b, delta, ell, sigma, m0_v = params

    drift, dispersion, _, P0, H = model_chirp(lam, b, ell, sigma, delta)
    m0 = jnp.array([0., 0., m0_v, 0.])

    m_and_cov = disc_chirp_lcd(lam, b, ell, sigma)
    return drift, dispersion, m_and_cov, m0, P0, H


def build_lascala_model(params: jndarray) -> Tuple[Callable[[jndarray], jndarray],
                                                   Callable[[jndarray], jndarray],
                                                   Callable[[jndarray, float], Tuple[jndarray, jndarray]],
                                                   jndarray, jndarray, jndarray]:
    """See the docstring of :code:`build_chirp_model`.

    Parameters
    ----------
    params : jnp.ndarray
        Parameters. From left to right, they are, lam, b, delta, ell, sigma, m0_1.

    Returns
    -------
    Callable, Callable, Callable, jnp.ndarray, jnp.ndarray, jnp.ndarray
        Drift, dispersion, conditional mean and cov, m0, P0, and H. All that are essential to run filter and smoother.
    """
    delta, ell, sigma, m0_v = params

    drift, dispersion, _, P0, H = model_lascala(ell, sigma, delta)
    m0 = jnp.array([0., 0., m0_v, 0.])

    m_and_cov = disc_model_lascala_lcd(ell, sigma)
    return drift, dispersion, m_and_cov, m0, P0, H
