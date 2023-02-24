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
A few implementations of commonly-used stochastic filters and smoothers in discrete time and continuous-discrete time.
"""
import jax
import jax.numpy as jnp
import jax.scipy
from chirpgp.quadratures import SigmaPoints, rk4_m_cov_backward, rk4_m_cov
from typing import Callable, Tuple
from functools import partial

__all__ = ['kf',
           'rts',
           'ekf',
           'ekf_for_kpt',
           'eks',
           'cd_ekf',
           'cd_eks',
           'sgp_filter',
           'sgp_smoother',
           'cd_sgp_filter',
           'cd_sgp_smoother']


@partial(jax.vmap, in_axes=[0, 0])
def _vectorised_outer(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.outer(x, y)


def _log_normal_pdf(x: float, mu: float, variance: float) -> float:
    return jax.scipy.stats.norm.logpdf(x, mu, jnp.sqrt(variance))


def _linear_predict(F: jnp.ndarray, Sigma: jnp.ndarray,
                    m: jnp.ndarray, P: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Mean and covariance of X_k from X_k = F X_{k-1} + Q_{k-1}.
    """
    return F @ m, F @ P @ F.T + Sigma


def _linear_update(mp: jnp.ndarray, Pp: jnp.ndarray,
                   H: jnp.ndarray, Xi: float, y: float) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Update for linear Gaussian measurement models
    (note that here the dim of measurement variable is assumed to be 1).

    Returns
    -------
    jnp.ndarray, jnp.ndarray, float
        Mean, variance, and negative log likelihood.
    """
    S = H @ Pp @ H.T + Xi
    K = Pp @ H.T / S
    pred = H @ mp
    return mp + K * (y - pred), Pp - jnp.outer(K, K) * S, -_log_normal_pdf(y, pred, S)


def _gaussian_smoother_common(DT: jnp.ndarray,
                              mf: jnp.ndarray, Pf: jnp.ndarray,
                              mp: jnp.ndarray, Pp: jnp.ndarray,
                              ms: jnp.ndarray, Ps: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Common procedure for Gaussian smoothers as per Equation 2.25 in Zhao 2021.

    Notes
    -----
    DT is the transpose of D.
    """
    c, low = jax.scipy.linalg.cho_factor(Pp)
    G = jax.scipy.linalg.cho_solve((c, low), DT).T
    ms = mf + G @ (ms - mp)
    Ps = Pf + G @ (Ps - Pp) @ G.T
    return ms, Ps


def _sgp_prediction(sgps: SigmaPoints,
                    vectorised_cond_m_cov: Callable[[jnp.ndarray, float], Tuple[jnp.ndarray, jnp.ndarray]],
                    dt: float,
                    mf: jnp.ndarray, Pf: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    r"""Sigma-point prediction of state-space model.

    Return the sigma-point approximated mean and covariance of :math:`X_k` from a state-space defined by the
    conditional mean and covariance :code:`cond_m_cov` starting from :math:`X_{k-1} \sim N(mf, Pf)`.

    Parameters
    ----------
    sgps : SigmaPoints
        Sigma-points object.
    vectorised_cond_m_cov : Callable
        A vectorised function (in particular, vmap-ed) that returns the conditional mean and covariance.
    dt : float
        Time interval between t_k and t_{k-1}.
    mf : jnp.ndarray
        Initial mean of this prediction.
    Pf : jnp.ndarray
        Initial covariance of this prediction.

    Returns
    -------
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
        Sigma-points predicted mean and covariance. The last two returns are used by the sgp smoother.
    """
    chol_Pf = jax.scipy.linalg.cholesky(Pf, lower=True)
    chi = sgps.gen_sigma_points(mf, chol_Pf)

    evals_of_m, evals_of_cov = vectorised_cond_m_cov(chi, dt)
    mp = sgps.expectation(evals_of_m)
    Pp = sgps.expectation(_vectorised_outer(evals_of_m, evals_of_m) + evals_of_cov) - jnp.outer(mp, mp)
    return mp, Pp, chi, evals_of_m


def _cd_sgp_common(sgps: SigmaPoints,
                   vectorised_drift: Callable[[jnp.ndarray], jnp.ndarray],
                   dispersion_const: jnp.ndarray,
                   m: jnp.ndarray, P: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""Sigma-point prediction of SDE model.
    """
    chol_P = jax.scipy.linalg.cholesky(P, lower=True)
    chi = sgps.gen_sigma_points(m, chol_P)

    evals_of_drift = vectorised_drift(chi)
    mp = sgps.expectation(evals_of_drift)
    _Pp = sgps.expectation(_vectorised_outer(chi - m, evals_of_drift))
    Pp = _Pp + _Pp.T + dispersion_const @ dispersion_const.T
    return mp, Pp


def _stack_smoothing_results(mfs: jnp.ndarray, Pfs: jnp.ndarray,
                             mss: jnp.ndarray, Pss: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.vstack([mss, mfs[-1]]), jnp.vstack([Pss, Pfs[-1, None]])


def kf(F: jnp.ndarray, Sigma: jnp.ndarray,
       H: jnp.ndarray, Xi: float,
       m0: jnp.ndarray, P0: jnp.ndarray,
       ys: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Kalman filter for 1d measurement.

    Parameters
    ----------
    F : jnp.ndarray (d, d)
        State transition mean matrix.
    Sigma : jnp.ndarray (d, d)
        State transition covariance.
    H : jnp.ndarray (d, )
        Measurement vector (for 1d measurement).
    Xi : float
        Measurement variance.
    m0 : jnp.ndarray (d, )
        Initial mean.
    P0 : jnp.ndarray (d, d)
        Initial covariance.
    ys : jnp.ndarray (T, )
        Measurements.

    Returns
    -------
    jnp.ndarray, jnp.ndarray, jnp.ndarray
        Filtering posterior means and covariances, and negative log likelihoods.
    """

    def scan_body(carry, elem):
        mf, Pf, n_ell = carry
        y = elem

        mp, Pp = _linear_predict(F, Sigma, mf, Pf)
        mf, Pf, n_ell_inc = _linear_update(mp, Pp, H, Xi, y)
        n_ell = n_ell + n_ell_inc
        return (mf, Pf, n_ell), (mf, Pf, n_ell)

    _, (mfs, Pfs, n_ell) = jax.lax.scan(scan_body, (m0, P0, 0.), ys)
    return mfs, Pfs, n_ell


def rts(F: jnp.ndarray, Sigma: jnp.ndarray,
        mfs: jnp.ndarray, Pfs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """RTS smoother for linear models.

    Parameters
    ----------
    F : jnp.ndarray (d, d)
        State transition mean matrix.
    Sigma : jnp.ndarray (d, d)
        State transition covariance.
    mfs : jnp.ndarray (T, d)
        Filtering posterior means.
    Pfs : jnp.ndarray (T, d, d)
        Filtering posterior covariances.

    Returns
    -------
    jnp.ndarray, jnp.ndarray
        Means and covariances of the smoothing estimates.
    """

    def scan_body(carry, elem):
        ms, Ps = carry
        mf, Pf = elem

        ms, Ps = _gaussian_smoother_common(F @ Pf,
                                           mf, Pf,
                                           F @ mf, F @ Pf @ F.T + Sigma,
                                           ms, Ps)
        return (ms, Ps), (ms, Ps)

    _, (mss, Pss) = jax.lax.scan(scan_body, (mfs[-1], Pfs[-1]), (mfs[:-1], Pfs[:-1]), reverse=True)
    return _stack_smoothing_results(mfs, Pfs, mss, Pss)


def ekf(cond_m_cov: Callable[[jnp.ndarray, float], Tuple[jnp.ndarray, jnp.ndarray]],
        H: jnp.ndarray, Xi: float,
        m0: jnp.ndarray, P0: jnp.ndarray,
        dt: float, ys: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Extended Kalman filter for non-linear models.

    Parameters
    ----------
    cond_m_cov : Callable ((d, ), float) -> (d, ), (d, d)
        A function that returns the conditional mean and covariance of SDE.
    H : jnp.ndarray (d, )
        Measurement matrix (for 1d measurement).
    Xi : float
        Measurement variance.
    m0 : jnp.ndarray (d, )
        Initial mean.
    P0 : jnp.ndarray (d, d)
        Initial covariance.
    dt : float
        Time interval.
    ys : jnp.ndarray (T, )
        Measurements.

    Returns
    -------
    jnp.ndarray, jnp.ndarray, jnp.ndarray
        Filtering posterior means and covariances, and negative log likelihoods.
    """

    def scan_body(carry, elem):
        mf, Pf, n_ell = carry
        y = elem

        jac_F = jax.jacfwd(lambda u: cond_m_cov(u, dt)[0], argnums=0)(mf)
        mp, Sigma = cond_m_cov(mf, dt)
        Pp = jac_F @ Pf @ jac_F.T + Sigma

        mf, Pf, n_ell_inc = _linear_update(mp, Pp, H, Xi, y)
        n_ell = n_ell + n_ell_inc
        return (mf, Pf, n_ell), (mf, Pf, n_ell)

    _, (mfs, Pfs, n_ell) = jax.lax.scan(scan_body, (m0, P0, 0.), ys)
    return mfs, Pfs, n_ell


def ekf_for_kpt(F: jnp.ndarray, Sigma: jnp.ndarray,
                h: Callable[[jnp.ndarray], jnp.ndarray], Xi: float,
                m0: jnp.ndarray, P0: jnp.ndarray,
                dt: float, ys: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Ad-hoc extended Kalman filter for the KPT model.

    Parameters
    ----------
    F : jnp.ndarray (d, d)
        State transition mean matrix.
    Sigma : jnp.ndarray (d, d)
        State transition covariance.
    h : Callable jnp.ndarray (d, ) -> jnp.ndarray ()
        Measurement function (for 1d measurement).
    Xi : float
        Measurement variance.
    m0 : jnp.ndarray (d, )
        Initial mean.
    P0 : jnp.ndarray (d, d)
        Initial covariance.
    dt : float
        Time interval.
    ys : jnp.ndarray (T, )
        Measurements.

    Returns
    -------
    jnp.ndarray, jnp.ndarray, jnp.ndarray
        Filtering posterior means and covariances, and negative log likelihoods.
    """

    def scan_body(carry, elem):
        mf, Pf, n_ell = carry
        y = elem

        mp, Pp = _linear_predict(F, Sigma, mf, Pf)

        H = jax.jacfwd(h)(mp)
        S = H @ Pp @ H.T + Xi
        K = Pp @ H.T / S
        pred = h(mp)
        mf = mp + K * (y - pred)
        Pf = Pp - jnp.outer(K, K) * S
        n_ell = n_ell -_log_normal_pdf(y, pred, S)
        return (mf, Pf, n_ell), (mf, Pf, n_ell)

    _, (mfs, Pfs, n_ell) = jax.lax.scan(scan_body, (m0, P0, 0.), ys)
    return mfs, Pfs, n_ell


def eks(cond_m_cov: Callable[[jnp.ndarray, float], Tuple[jnp.ndarray, jnp.ndarray]],
        mfs: jnp.ndarray, Pfs: jnp.ndarray, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Extended Kalman smoother for non-linear dynamical models.

    Parameters
    ----------
    cond_m_cov : Callable ((d, ), float) -> (d, ), (d, d)
        A function that returns the conditional mean and covariance of SDE.
    mfs : jnp.ndarray (T, d)
        Filtering posterior means.
    Pfs : jnp.ndarray (T, d, d)
        Filtering posterior covariances.
    dt : float
        Time interval.

    Returns
    -------
    jnp.ndarray, jnp.ndarray
        Means and covariances of the smoothing estimates.
    """

    def scan_body(carry, elem):
        ms, Ps = carry
        mf, Pf = elem

        jac_F = jax.jacfwd(lambda u: cond_m_cov(u, dt)[0], argnums=0)(mf)
        mp, Sigma = cond_m_cov(mf, dt)
        Pp = jac_F @ Pf @ jac_F.T + Sigma
        ms, Ps = _gaussian_smoother_common(jac_F @ Pf, mf, Pf, mp, Pp, ms, Ps)
        return (ms, Ps), (ms, Ps)

    _, (mss, Pss) = jax.lax.scan(scan_body, (mfs[-1], Pfs[-1]), (mfs[:-1], Pfs[:-1]), reverse=True)
    return _stack_smoothing_results(mfs, Pfs, mss, Pss)


def cd_ekf(a: Callable, b: Callable,
           H: jnp.ndarray, Xi: float,
           m0: jnp.ndarray, P0: jnp.ndarray,
           dt: float, ys: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Continuous-discrete extended Kalman filter with 4th order Runge--Kutta integration.

    Parameters
    ----------
    a : Callable (d, ) -> (d, )
        SDE drift function.
    b : Callable (d, w) -> (d, w)
        SDE dispersion function.
    H : jnp.ndarray (d, )
        Measurement matrix (for 1d measurement).
    Xi : float
        Measurement noise variance.
    m0 : jnp.ndarray (d, )
        Initial mean.
    P0 : jnp.ndarray (d, d)
        Initial covariance.
    dt : float
        Time interval
    ys : jnp.ndarray (T, )
        Measurements.

    Returns
    -------
    jnp.ndarray, jnp.ndarray, jnp.ndarray
        Filtering posterior means and covariances, and negative log likelihoods.
    """
    jac_of_a = jax.jacfwd(a)

    def odes(m, P):
        return a(m), P @ jac_of_a(m).T + jac_of_a(m) @ P + b(m) @ b(m).T

    def scan_body(carry, elem):
        mf, Pf, n_ell = carry
        y = elem

        mp, Pp = rk4_m_cov(odes, mf, Pf, dt)
        mf, Pf, n_ell_inc = _linear_update(mp, Pp, H, Xi, y)
        n_ell = n_ell + n_ell_inc
        return (mf, Pf, n_ell), (mf, Pf, n_ell)

    _, filtering_results = jax.lax.scan(scan_body, (m0, P0, 0.), ys)
    return filtering_results


def cd_eks(a: Callable, b: Callable,
           mfs: jnp.ndarray, Pfs: jnp.ndarray,
           dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Continuous-discrete extended Kalman smoother with 4th order Runge--Kutta integration.

    Parameters
    ----------
    a : Callable (d, ) -> (d, )
        SDE drift function.
    b : Callable (d, w) -> (d, w)
        SDE dispersion function.
    mfs : jnp.ndarray (T, d)
        Filtering means.
    Pfs : jnp.ndarray (T, d, d)
        Filtering covariances.
    dt : float
        Time interval

    Returns
    -------
    jnp.ndarray, jnp.ndarray
        Mean and covariance of the smoothing estimates.
    """
    dt = -dt

    jac_of_a = jax.jacfwd(a)

    def odes(m, P, mf, Pf):
        gamma = b(m) @ b(m).T
        c, low = jax.scipy.linalg.cho_factor(Pf)
        jac_and_gamma_and_chol = jac_of_a(m) + jax.scipy.linalg.cho_solve((c, low), gamma.T).T
        return a(m) + gamma @ jax.scipy.linalg.cho_solve((c, low), m - mf), \
               jac_and_gamma_and_chol @ P + P @ jac_and_gamma_and_chol.T - gamma

    def scan_body(carry, elem):
        ms, Ps = carry
        mf, Pf = elem

        ms, Ps = rk4_m_cov_backward(odes, ms, Ps, mf, Pf, dt)

        return (ms, Ps), (ms, Ps)

    _, (mss, Pss) = jax.lax.scan(scan_body, (mfs[-1], Pfs[-1]), (mfs[:-1], Pfs[:-1]), reverse=True)
    return _stack_smoothing_results(mfs, Pfs, mss, Pss)


def sgp_filter(cond_m_cov: Callable[[jnp.ndarray, float], Tuple[jnp.ndarray, jnp.ndarray]],
               sgps: SigmaPoints,
               H: jnp.ndarray, Xi: float,
               m0: jnp.ndarray, P0: jnp.ndarray,
               dt: float, ys: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Continuous-discrete sigma-point filter by discretising the SDE.

    Parameters
    ----------
    cond_m_cov : Callable ((d, ), float) -> (d, ), (d, d)
        A function that returns the conditional mean and covariance of SDE.
    sgps : SigmaPoints
        Instance of :code:`SigmaPoints`.
    H : jnp.ndarray (d, )
        Measurement matrix (for 1d measurement).
    Xi : float
        Measurement noise variance.
    m0 : jnp.ndarray (d, )
        Initial mean.
    P0 : jnp.ndarray (d, d)
        Initial covariance.
    dt : float
        Time interval.
    ys : jnp.ndarray (T, )
        Measurements.

    Returns
    -------
    jnp.ndarray, jnp.ndarray, jnp.ndarray
        Filtering posterior means and covariances, and negative log likelihoods.
    """

    vectorised_cond_m_cov = jax.vmap(cond_m_cov, in_axes=[0, None])

    def scan_sgp_filter(carry, elem):
        mf, Pf, n_ell = carry
        y = elem

        mp, Pp, _, _ = _sgp_prediction(sgps, vectorised_cond_m_cov, dt, mf, Pf)
        mf, Pf, n_ell_inc = _linear_update(mp, Pp, H, Xi, y)
        n_ell = n_ell + n_ell_inc
        return (mf, Pf, n_ell), (mf, Pf, n_ell)

    _, (mfs, Pfs, n_ell) = jax.lax.scan(scan_sgp_filter, (m0, P0, 0.), ys)
    return mfs, Pfs, n_ell


def sgp_smoother(cond_m_cov: Callable[[jnp.ndarray, float], Tuple[jnp.ndarray, jnp.ndarray]],
                 sgps: SigmaPoints,
                 mfs: jnp.ndarray, Pfs: jnp.ndarray,
                 dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Continuous-discrete sigma-point smoother by discretising the SDE.

    Parameters
    ----------
    cond_m_cov : Callable ((d, ), float) -> (d, ), (d, d)
        A function that returns the conditional mean and covariance of SDE.
    sgps : SigmaPoints
        Instance of :code:`SigmaPoints`.
    mfs : jnp.ndarray (T, d)
        Filtering means.
    Pfs : jnp.ndarray (T, d, d)
        Filtering covariances.
    dt : float
        Time interval.

    Returns
    -------
    jnp.ndarray, jnp.ndarray
        Means and covariances of the smoothing estimates.
    """

    vectorised_cond_m_cov = jax.vmap(cond_m_cov, in_axes=[0, None])

    def scan_sgp_smoother(carry, elem):
        ms, Ps = carry
        mf, Pf = elem

        mp, Pp, chi, evals_of_m = _sgp_prediction(sgps, vectorised_cond_m_cov, dt, mf, Pf)
        D = sgps.expectation(_vectorised_outer(chi, evals_of_m)) - jnp.outer(mf, mp)

        ms, Ps = _gaussian_smoother_common(D.T, mf, Pf, mp, Pp, ms, Ps)
        return (ms, Ps), (ms, Ps)

    _, (mss, Pss) = jax.lax.scan(scan_sgp_smoother, (mfs[-1], Pfs[-1]), (mfs[:-1], Pfs[:-1]), reverse=True)
    return _stack_smoothing_results(mfs, Pfs, mss, Pss)


def cd_sgp_filter(a: Callable, b: jnp.ndarray,
                  sgps: SigmaPoints,
                  H: jnp.ndarray, Xi: float,
                  m0: jnp.ndarray, P0: jnp.ndarray,
                  dt: float, ys: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Continuous-discrete sigma-points Kalman filter with 4th order Runge--Kutta integration.

    Parameters
    ----------
    a : Callable (d, ) -> (d, )
        SDE drift function.
    b : jnp.ndarray (d, dw)
        SDE dispersion matrix.
    sgps : SigmaPoints
        Instance of :code:`SigmaPoints`.
    H : jnp.ndarray (d, )
        Measurement matrix (for 1d measurement).
    Xi : float
        Measurement noise variance.
    m0 : jnp.ndarray (d, )
        Initial mean.
    P0 : jnp.ndarray (d, d)
        Initial covariance.
    dt : float
        Time interval
    ys : jnp.ndarray (T, )
        Measurements.

    Returns
    -------
    jnp.ndarray, jnp.ndarray, jnp.ndarray
        Filtering posterior means and covariances, and negative log likelihoods.
    """
    vectorised_drift = jax.vmap(a, in_axes=[0])

    def odes(m, P):
        return _cd_sgp_common(sgps, vectorised_drift, b, m, P)

    def scan_body(carry, elem):
        mf, Pf, n_ell = carry
        y = elem

        mp, Pp = rk4_m_cov(odes, mf, Pf, dt)
        mf, Pf, n_ell_inc = _linear_update(mp, Pp, H, Xi, y)
        n_ell = n_ell + n_ell_inc
        return (mf, Pf, n_ell), (mf, Pf, n_ell)

    _, filtering_results = jax.lax.scan(scan_body, (m0, P0, 0.), ys)
    return filtering_results


def cd_sgp_smoother(a: Callable, b: jnp.ndarray,
                    sgps: SigmaPoints,
                    mfs: jnp.ndarray, Pfs: jnp.ndarray,
                    dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Continuous-discrete sigma-points Kalman smoother with 4th order Runge--Kutta integration.

    Parameters
    ----------
    a : Callable (d, ) -> (d, )
        SDE drift function.
    b : jnp.ndarray (d, dw)
        SDE dispersion matrix.
    sgps : SigmaPoints
        Instance of :code:`SigmaPoints`.
    mfs : jnp.ndarray (T, d)
        Filtering means.
    Pfs : jnp.ndarray (T, d, d)
        Filtering covariances.
    dt : float
        Time interval

    Returns
    -------
    jnp.ndarray, jnp.ndarray
        Mean and covariance of the smoothing estimates.
    """
    dt = -dt

    vectorised_drift = jax.vmap(a, in_axes=[0])

    def odes(m, P, mf, Pf):
        gamma = b @ b.T
        c, low = jax.scipy.linalg.cho_factor(Pf)
        G = jax.scipy.linalg.cho_solve((c, low), gamma)

        _m, _P = _cd_sgp_common(sgps, vectorised_drift, b, m, P)
        return _m + G.T @ (m - mf), _P + G.T @ P + P @ G - 2 * gamma

    def scan_body(carry, elem):
        ms, Ps = carry
        mf, Pf = elem

        ms, Ps = rk4_m_cov_backward(odes, ms, Ps, mf, Pf, dt)

        return (ms, Ps), (ms, Ps)

    _, (mss, Pss) = jax.lax.scan(scan_body, (mfs[-1], Pfs[-1]), (mfs[:-1], Pfs[:-1]), reverse=True)
    return _stack_smoothing_results(mfs, Pfs, mss, Pss)
