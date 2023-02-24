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
import numpy as np
import tme.base_jax as tme
from jax import jacfwd, jacrev
from typing import Tuple, Callable, Union, Sequence

__all__ = ['jndarray',
           'g',
           'g_inv',
           'model_chirp',
           'model_harmonic_chirp',
           'model_lascala',
           'disc_chirp_lcd',
           'disc_chirp_lcd_cond_v',
           'disc_harmonic_chirp_lcd',
           'disc_chirp_tme',
           'disc_chirp_euler_maruyama',
           'disc_model_lascala_lcd',
           'disc_m32',
           'build_chirp_model',
           'build_harmonic_chirp_model',
           'build_lascala_model',
           'build_kpt_chirp_model',
           'posterior_cramer_rao']

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

    def dispersion(_) -> jndarray:
        return jnp.diag(jnp.array([b, b, 0., 2 * sigma * (math.sqrt(3) / ell) ** 1.5]))

    m0 = jnp.array([0., 1., 0., 0.])
    P0 = jax.scipy.linalg.block_diag(delta, delta, _stationary_cov_m32(ell, sigma))

    H = jnp.array([0., 1., 0., 0.])
    return drift, dispersion, m0, P0, H


def model_harmonic_chirp(lam: float,
                         b: float,
                         ell: float,
                         sigma: float,
                         delta: float,
                         num_harmonics: int = 1,
                         freq_scale: float = 1.) -> Tuple[Callable[[jndarray], jndarray],
                                                          Callable[[jndarray], jndarray],
                                                          jndarray, jndarray, jndarray]:
    r"""The proposed harmonic chirp and IF model in Equation 14.

    Parameters
    ----------
    lam : float
        Damping factor.
    b : float
        Chirp dispersion constant.
    ell : float
        Fundamental IF prior length scale.
    sigma : float
        Fundamental IF prior magnitude scale.
    delta : float
        Chirp initial covariances :math:`P_0^X = \delta \, eye(2)`.
    num_harmonics : int, default=1
        Number of harmonics (including the fundamental frequency itself).
    freq_scale : float, default=1.
        Scale the frequency for numerical stability. Freq = scale * g(V).

    Returns
    -------
    Callable, Callable, jnp.ndarray, jnp.ndarray, jnp.ndarray
        Drift function, dispersion, initial mean, initial covariance, and measurement H.

    Notes
    -----
    For simplicity, we use the same lam, b, and delta parameters for all the harmonic components. You can for sure
    use different parameters for the harmonics by modifying this function.
    """
    _gamma = math.sqrt(3) / ell
    _m32_drift = jnp.array([[0., 1.],
                            [-(_gamma ** 2), -2 * _gamma]])

    def drift(u: jndarray) -> jndarray:
        w = 2 * math.pi * g(u[-2]) * freq_scale
        ls = [jnp.array([[-lam, -w * k],
                         [w * k, -lam]]) for k in range(1, num_harmonics + 1)]
        return jax.scipy.linalg.block_diag(*ls, _m32_drift) @ u

    def dispersion(_) -> jndarray:
        return jnp.diag(jnp.array([b, b] * num_harmonics + [0., 2 * sigma * (math.sqrt(3) / ell) ** 1.5]))

    m0 = jnp.array([0., 1.] * num_harmonics + [0., 0.])
    _deltas = [delta, delta] * num_harmonics
    P0 = jax.scipy.linalg.block_diag(*_deltas, _stationary_cov_m32(ell, sigma))

    H = jnp.array([0., 1.] * num_harmonics + [0., 0.])
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


def disc_harmonic_chirp_lcd(lam: Union[float, jndarray],
                            b: float,
                            ell: float,
                            sigma: float,
                            num_harmonics: int = 1,
                            freq_scale: float = 1.) -> Callable[[jndarray, float], Tuple[jndarray, jndarray]]:
    """Locally conditional discretisation of the harmonic chirp model.

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
    num_harmonics : int, default=1
        Number of harmonics (including the fundamental frequency itself).
    freq_scale : float, default=1.
        Scale the frequency for numerical stability. Freq = scale * g(V).

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
        w = 2 * math.pi * g(u[-2]) * freq_scale
        blk_harmonics = [jnp.array([[jnp.cos(dt * k * w), -jnp.sin(dt * k * w)],
                                    [jnp.sin(dt * k * w), jnp.cos(dt * k * w)]]) * jnp.exp(-lam * dt) for k in
                         range(1, num_harmonics + 1)]
        blk_m32_m, blk_m32_Sigma = _m32_solution(ell, sigma, dt)

        cond_m = jax.scipy.linalg.block_diag(*blk_harmonics, blk_m32_m) @ u

        _d1 = [b ** 2 * dt] * (2 * num_harmonics)
        _d2 = [b ** 2 / (2 * lam) * (1 - jnp.exp(-2 * lam * dt))] * (2 * num_harmonics)
        Sigma = jax.lax.cond(lam == 0.,
                             lambda _: jax.scipy.linalg.block_diag(*_d1, blk_m32_Sigma),
                             lambda _: jax.scipy.linalg.block_diag(*_d2, blk_m32_Sigma),
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
        return tme.mean_and_cov(u, dt, drift, dispersion, order)

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


def build_harmonic_chirp_model(params: jndarray,
                               num_harmonics: int = 1,
                               freq_scale: float = 1.) -> Tuple[Callable[[jndarray], jndarray],
                                                                Callable[[jndarray], jndarray],
                                                                Callable[[jndarray, float], Tuple[jndarray, jndarray]],
                                                                jndarray, jndarray, jndarray]:
    """Build harmonic chirp model from given parameters for optimisation.

    Parameters
    ----------
    params : jnp.ndarray
        Parameters. From left to right, they are, lam, b, delta, ell, sigma, m0_1.
    num_harmonics : int, default=1
        Number of harmonics.
    freq_scale : float, default=1.
        Scale the frequency for numerical stability. Freq = scale * g(V).

    Returns
    -------
    Callable, Callable, Callable, jnp.ndarray, jnp.ndarray, jnp.ndarray
        Drift, dispersion, conditional mean and cov, m0, P0, and H. All that are essential to run filter and smoother.
    """
    lam, b, delta, ell, sigma, m0_v = params

    drift, dispersion, _, P0, H = model_harmonic_chirp(lam, b, ell, sigma, delta,
                                                       num_harmonics=num_harmonics,
                                                       freq_scale=freq_scale)
    m0 = jnp.array([0., 1.] * num_harmonics + [m0_v, 0.])

    m_and_cov = disc_harmonic_chirp_lcd(lam, b, ell, sigma,
                                        num_harmonics=num_harmonics,
                                        freq_scale=freq_scale)
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


def build_kpt_chirp_model(params: Sequence[float],
                          fs: float,
                          num_harmonics: int = 1) -> Tuple[jndarray, jndarray, jndarray, jndarray,
                                                           Callable[[jndarray], jndarray]]:
    """Build the state-space model in Liming et al., 2017.

    Parameters
    ----------
    params : jnp.ndarray
        From left to right, they are, q1, q2, p0, f0, and a0.
        The parameters q1 and q2 are the variance for the frequency and amplitudes. f0 and a0 are the initial values
        of frequency and amplitude.
    fs : float
        Sampling frequency.
    num_harmonics : int, default=1
        Number of harmonics (including the fundamental frequency).

    Returns
    -------
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Callable
        F, Sigma, m0, P0, h

    References
    ----------
    Liming Shi, Jesper K. Nielsen, Jesper R. Jensen, Max A. Little, and Mads G. Christensen. A Kalman-based fundamental
    frequency estimation algorithm. 2017.

    https://github.com/LimingShi/Kalman-Pitch-tracking.

    Notes
    -----
    The model seems to be very sensitive to the state-space parameters. With the maximum likelihood estimation,
    sometimes the estimated parameters do not make sense, and are worse than the hand-engineered ones.

    To avoid the negative frequency, we added a positive bijection g. This gives a fair comparison to other methods
    in the experiment.
    """
    q1, q2, p0, f0, a0 = params

    dim_x = num_harmonics + 2

    P0 = p0 * jnp.eye(dim_x)
    m0 = jnp.array([2 * math.pi * f0 / fs] + [a0 for _ in range(num_harmonics)] + [0.])

    F = np.eye(dim_x)
    F[-1, 0] = 1.

    Gamma = jnp.eye(dim_x)[:, :-1]
    _Sigma = jnp.diag(jnp.array([(2 * math.pi * q1 / fs) ** 2] + [q2 for _ in range(num_harmonics)]))
    Sigma = Gamma @ _Sigma @ Gamma.T

    G = jnp.eye(dim_x)[1:-1, :]

    def h(x):
        return jnp.dot(G @ x,
                       jnp.sin(g(x[0] + x[-1]) * jnp.arange(1, num_harmonics + 1))
                       )

    return F, Sigma, m0, P0, h


def posterior_cramer_rao(xss: jndarray, yss: jndarray, j0: jndarray,
                         logpdf_transition: Callable[[jndarray, jndarray], float],
                         logpdf_likelihood: Callable[[float, jndarray], float]) -> jndarray:
    """Compute the posterior Cramér--Rao lower bound at the given times by Monte Carlo.

    Ad-hoc implementation for 1D measurement.

    Parameters
    ----------
    xss : jnp.ndarray (T + 1, N, d)
        Trajectories of the state. T, N, and d are the number of times, number of MC samples, and state dimension,
        respectively. Note that one should have the initial samples in the beginning.
    yss : jnp.ndarray (T, N)
        Measurements.
    j0 : jnp.ndarray (d, d)
        -E[H_X log p(x0)].
    logpdf_transition : (d, ), (d, ) -> float
        Log p(x_k | x_{k-1})
    logpdf_likelihood : float, (d, ) -> float
        Log p(y_k | x_k)

    Returns
    -------
    jnp.ndarray (T, d, d)
        J, the inverse of the PCRLB lower bound matrices.

    Notes
    -----
    There are multiple posterior Cramér--Rao lower bounds in the literature (see, Frutsche et al., 2014). We here use
    the one by Tichavsky et al., 1998, see, also, Challa et al., 2011. Although this one is not the tightest, it is
    to our knowledge the only one that can be exactly computed (with Monte Carlo approximation). Other bounds need
    the filtering distributions to integrate out the latent variables which are not exactly tractable.

    References
    ----------
    Subhash Challa, Mark Morelande, Darko Musicki, and Robin Evans. Fundamentals of object tracking. Cambridge
    University Press, 2011, pp. 53.

    Peter Tichavsky, Carlos Muravchik, and Arye Nehorai. Posterior Cramér--Rao bounds for discrete-time nonlinear
    filtering. IEEE Transactions on Signal Processing, 1998.

    Carsteb Frutsche, Emre Ozkan, Lennart Svensson, and Fredrik Gustafsson. A fresh look at Bayesian Cramér--Rao
    bounds for discrete-time nonlinear filtering. In 17th International Conference on Information Fusion, 2014.
    """
    htt_logpdf_transition = jax.vmap(jax.hessian(logpdf_transition, argnums=0), in_axes=[0, 0])
    hts_logpdf_transition = jax.vmap(jacfwd(jacrev(logpdf_transition, argnums=1), argnums=0), in_axes=[0, 0])
    hss_logpdf_transition = jax.vmap(jax.hessian(logpdf_transition, argnums=1), in_axes=[0, 0])
    htt_logpdf_likelihood = jax.vmap(jax.hessian(logpdf_likelihood, argnums=1), in_axes=[0, 0])

    def scan_body(carry, elem):
        j = carry
        yt, xt, xs = elem

        d11 = -jnp.mean(hss_logpdf_transition(xt, xs), axis=0)
        d12 = -jnp.mean(hts_logpdf_transition(xt, xs), axis=0)
        d22 = -jnp.mean(htt_logpdf_transition(xt, xs) + htt_logpdf_likelihood(yt, xt), axis=0)

        j = d22 - d12.T @ jnp.linalg.solve(j + d11, d12)
        return j, j

    _, js = jax.lax.scan(scan_body, j0, (yss, xss[1:], xss[:-1]))
    return js
