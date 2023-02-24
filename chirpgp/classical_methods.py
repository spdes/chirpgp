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
This module implements a few classical methods for estimating instantaneous frequency. These include:

1. First moment from power spectrum.
2. Hilbert transform -> analytical signa -> derivative of phase.
3. MLE polynomial regression.
4. Adaptive notch filter (by Niedźwiecki and Meller, 2011).

Notes
-----
Most of the scipy.signal functions are not supported by jax, so many here use numpy implementations.
"""
import math
import jax
import jax.scipy.optimize
import jax.numpy as jnp
import jaxopt
import numpy as np
import scipy.signal
from chirpgp.toymodels import gen_chirp
from chirpgp.gauss_newton import gauss_newton, levenberg_marquardt
from functools import partial
from typing import Tuple

__all__ = ['hilbert_method',
           'mean_power_spectrum',
           'mle_polynomial',
           'adaptive_notch_filter']

jndarray = jnp.ndarray


def hilbert_method(ts: np.ndarray, ys: np.ndarray, *args, **kwargs) -> np.ndarray:
    r"""Hilbert transform method.

    An analytical representation of a signal :math:`y(t)` is defined as

    .. math::

        y_a(t) = y(t) + H(y)(t) i,

    where :math:`H(y)` stands for the Hilbert transform of :math:`y(t)`. So the instantaneous frequency
    of y(t) is then the derivative of :math:`H(y)(t)` (divided by 2 pi).

    Parameters
    ----------
    ts : ndarray (T, )
        Times. Must be uniform.
    ys : ndarray (T, )
        Signal.
    *args, **kwargs
        Parameters passed to scipy.signal.hilbert.

    Returns
    -------
    ndarray
        Instantaneous frequency.

    See Also
    --------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html.

    https://se.mathworks.com/help/signal/ug/hilbert-transform-and-instantaneous-frequency.html.

    Notes
    -----
    This method hardly applies to noisy measurements. You should filter out the noises first.
    """
    fs = 1 / np.diff(ts)[0]
    analytic_y = scipy.signal.hilbert(ys, *args, **kwargs)
    return np.diff(np.unwrap(np.angle(analytic_y))) / (2 * math.pi) * fs


def mean_power_spectrum(ts: np.ndarray, ys: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    r"""Estimate instantaneous frequency using mean of power spectral density.

    Let S(y)(t, f) be the power spectral density of signal y parametrised by t and f.

    Then one can estimate IF by

    .. math::

        IF(t) \approx \frac{\int f S(t, f) df}{\int S(t, f) df}.

    Parameters
    ----------
    ts : ndarray (T, )
        Times. Implemented for uniform times only.
    ys : ndarray (T, )
        Signal.
    *args, **kwargs
        Parameters passed to scipy.signal.spectrogram.

    Returns
    -------
    ndarray, ndarray
        Segment times and estimated IF at these times.
    """
    fs = 1 / np.diff(ts)[0]
    freqs, new_ts, Sxx = scipy.signal.spectrogram(ys, fs, *args, **kwargs)
    return new_ts, jnp.sum(freqs[:, None] * Sxx, axis=0) / jnp.sum(Sxx, axis=0)


def mle_polynomial(ts: jnp.ndarray, ys: jnp.ndarray, Xi: float,
                   init_params: jnp.ndarray,
                   method: str = 'levenberg_marquardt',
                   *args, **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""Maximum likelihood estimation for polynomial-approximated chirp frequency.

    .. math::

        Y(t_k) = \alpha \, \sin(\int^{t_k}_0 2 \pi f^n(s) ds) + \xi_k,

    where :math:`f^n(t) := c_0 + c_1 \, t + c_2 \, t^2 + \cdots + c_n \, t^n`, and :math:`\xi_k \sim N(0, \Xi)`.
    Now suppose that at times :math:`t_1,t_2, \ldots, t_T` we have measurements :math:`y_1,y_2, \ldots, y_T`.
    This is a polynomial regression problem. Eventually the objective function reads

    .. math::

        nll(\alpha, c_0, c_1, \ldots, c_n)
        = \sum^T_{k=1} (y_k - \alpha \, \sin(2 \pi \zeta^n(t_k)))^2 / Xi,

    where :math:`\zeta^n(t) = c_0 \, t + 1/2 \, c_1 \, t^2 + \ldots + 1/(n+1) \, c_n \, t^{n+1}`.

    Parameters
    ----------
    ts : jnp.ndarray (T, )
        Times
    ys : jnp.ndarray (T, )
        Chirp measurements.
    Xi : float
        Measurement noise variance.
    init_params : jnp.ndarray (n + 2, )
        Packed initial values. From left to right, they are [\alpha, c_0, c_1, \ldots, c_n], that is,
        [magnitude, initial phase, first poly coeff, second poly coeff, ...].
    method : str
        'levenberg_marquardt' (default) or 'gauss_newton'.
    *args, **kwargs
        Passed to the optimisation method.

    Returns
    -------
    jnp.ndarray. jnp.ndarray
        Optimised parameters and objective values.
    """
    n = init_params.shape[0] - 2
    if n < 0:
        raise ValueError('The size of init_params should be greater than 1.')
    alien = jnp.array([1. / (j + 1) for j in range(0, n + 1)])

    @partial(jax.vmap, in_axes=[0, None])
    def zeta(t, cs):
        return jnp.polyval(jnp.flip(jnp.insert(alien * cs, 0, 0.)), t)

    def f(params):
        alpha = params[0]
        poly_coeffs = params[1:]
        # TODO: Double check phase unwrap
        return gen_chirp(ts,
                         lambda u: alpha,
                         lambda u: zeta(u, poly_coeffs),
                         0.)

    if method == 'gauss_newton':
        minimiser, obj_vals = gauss_newton(f, init_params, ys, Xi, *args, **kwargs)
    elif method == 'levenberg_marquardt':
        minimiser, obj_vals = levenberg_marquardt(f, init_params, ys, Xi, *args, **kwargs)
    elif method == 'L-BFGS-B':

        def obj_func(params):
            return jnp.sum((ys - f(params)) ** 2) / Xi

        opt_solver = jaxopt.ScipyMinimize(method='L-BFGS-B', jit=True, fun=obj_func)
        minimiser, opt_state = opt_solver.run(init_params)
        obj_vals = opt_state.fun_val
    else:
        raise ValueError(f'Method {method} does not exist.')
    return minimiser, obj_vals


def adaptive_notch_filter(ts: jnp.ndarray, ys: jnp.ndarray,
                          alpha0: float, w0: float, s0: complex,
                          mu: float, gamma_alpha: float, gamma_w: float) -> Tuple[jndarray, jndarray, jndarray]:
    """Adaptive notch filter for frequency estimation.

    This implementation is based on the pilot adaptive notch filter as per Table II by Niedźwiecki and Meller 2011.

    Parameters
    ----------
    ts : jnp.ndarray (T, )
        Times.
    ys : jnp.ndarray (complex 32/64) (T, )
        Chirp measurements.
    alpha0: float
        Initial frequency rate.
    w0: float
        Initial instantaneous frequency.
    s0: float (complex 32/64)
        Initial envelope value.
    mu: float
        Parameter.
    gamma_alpha: float
        Parameter related to alpha.
    gamma_w: float
        Parameter related to frequency.

    Returns
    -------
    jnp.ndarray, jnp.ndarray, jnp.ndarray
        Estimated instantaneous frequencies, change rates, and envelope values.

    References
    ----------
    Maciej Niedźwiecki and Michał Meller. New Algorithms for Adaptive Notch Smoothing. IEEE Transactions on Signal
    Processing, 59(5):2024--2037, 2011.

    Notes
    -----
    Do not normalise :code:`alpha0` and :code:`w0` by dt. They are to be normalised inside this function.

    Parameters should satisfy gamma_alpha << gamma_w << mu < 1 according to the authors.
    """
    dt = jnp.diff(ts)[0]

    def scan_body(carry, elem):
        w, alpha, s = carry
        y = elem

        eps = y - jnp.exp(2 * math.pi * (w + alpha) * 1j) * s
        delta = jnp.imag(eps * jnp.exp(2 * math.pi * (w + alpha) * -1j) * jnp.conjugate(s)) / jnp.absolute(s) ** 2

        s = jnp.exp(2 * math.pi * (w + alpha) * 1j) * s + mu * eps
        w = w + alpha + gamma_w * delta
        alpha = alpha + gamma_alpha * delta
        return (w, alpha, s), (w, alpha, s)

    _, (freq_est, alpha_est, magnitude_est) = jax.lax.scan(scan_body, (w0 * dt, alpha0 * dt, s0), ys)

    return freq_est / dt, alpha_est / dt, magnitude_est
