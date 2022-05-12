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
Toymodels for testing chirp and IF estimation.
"""
import math
import jax.numpy as jnp
from chirpgp.models import jndarray
from chirpgp.tools import simulate_sde
from typing import Callable, List, Union, Tuple

__all__ = ['gen_chirp',
           'gen_chirp_envelope',
           'constant_mag',
           'damped_exp_mag',
           'random_ou_mag',
           'affine_freq',
           'polynomial_freq',
           'meow_freq']


def gen_chirp(ts: jndarray,
              magnitude_func: Callable[[jndarray], jndarray],
              phase_func: Callable[[jndarray], jndarray],
              base_phase: float = 0.) -> jndarray:
    r"""Generate a chirp signal of the form

    .. math::

        y(t) = \alpha(t) \sin(\phi_0 + 2 \pi \phi(t)),

    where :math:`\alpha`, :math:`\phi`, and :math:`\phi_0` stand for the magnitude function, phase function, and
    base phase, respectively.

    Parameters
    ----------
    ts : jnp.ndarray (T, )
        Times.
    magnitude_func : Callable (T, ) -> (T, )
        The magnitude function :math:`\alpha`.
    phase_func : Callable (T, ) -> (T, )
        The phase function :math:`\phi`.
    base_phase : float
        The base phase :math:`\phi_0`.

    Returns
    -------
    jnp.ndarray
        The chirp signal on :code:`ts`.

    Notes
    -----
    Please note the position of :math:`2 pi` in the equation above!
    """
    return magnitude_func(ts) * jnp.sin(base_phase + 2 * math.pi * phase_func(ts))


def gen_chirp_envelope(ts: jndarray,
                       magnitude_func: Union[Callable[[jndarray], jndarray], Callable[[jndarray], float]],
                       phase_func: Callable[[jndarray], jndarray],
                       base_phase: float = 0.) -> jndarray:
    r"""Generate a complex chirp signal of the form

    .. math::

        y(t) = \alpha(t) \exp(i 2 \, \pi \, \phi(t))

    Please refer to `gen_chirp` for details of the parameters.
    """
    return magnitude_func(ts) * jnp.exp((base_phase + 2 * math.pi * phase_func(ts)) * 1.j)


def constant_mag(b: float) -> Callable[[jndarray], jndarray]:
    """Constant magnitude function.

    Parameters
    ----------
    b : float
        The constant magnitude.
    """
    return lambda ts: jnp.ones_like(ts) * b


def damped_exp_mag(damp_rate: float) -> Callable[[jndarray], jndarray]:
    """Exponentially damped magnitude function.

    Parameters
    ----------
    damp_rate : float
        The damp rate.
    """
    return lambda ts: jnp.exp(-damp_rate * ts)


def random_ou_mag(ell: float, sigma: float, key: jndarray) -> Callable[[jndarray], jndarray]:
    """A realisation from an Ornstein--Uhlenbeck process as the magnitude.

    Parameters
    ----------
    ell : float
        The length scale of the OU process.
    sigma : float
        The magnitude scale of the OU process.
    key : jnp.ndarray
        PRNkey.
    """

    def m_and_cov(x: jndarray, dt: float):
        return jnp.exp(-dt / ell) * x, jnp.array([[sigma ** 2 * (1 - jnp.exp(-2 * dt / ell))]])

    def generate_ou(ts: jndarray) -> jndarray:
        dt = jnp.diff(ts)[0]
        T = ts.size
        return simulate_sde(m_and_cov,
                            jnp.array([0.]), jnp.array([[sigma ** 2]]),
                            dt, T, key, const_diag_cov=True).squeeze()

    return generate_ou


def affine_freq(a: float, b: float) -> Tuple[Callable[[jndarray], jndarray], Callable[[jndarray], jndarray]]:
    r"""Affine frequency function and quadratic phase function

    .. math::

        f(t) = a \, t + b

        \phi(t) = 1/2 \, a \, t^2 + b \, t

    Parameters
    ----------
    a : float
        Slope.
    b : float
        Offset.

    Returns
    -------
    Callable, Callable
        Frequency function and its phase function.
    """
    return lambda ts: a * ts + b, lambda ts: 0.5 * a * ts ** 2 + b * ts


def polynomial_freq(coeffs: List[float]) -> Tuple[Callable[[Union[jndarray, float]], Union[jndarray, float]],
                                                  Callable[[Union[jndarray, float]], Union[jndarray, float]]]:
    """Polynomial frequency and phase functions.

    Ploynomial = coeffs[0] + coeffs[1] * x + ...

    Parameters
    ----------
    coeffs : List[float]
        Coefficients of polynomial from lowest to the highest order.

    Returns
    -------
    Callable, Callable
        Polynomial of frequency and its corresponding phase function.
    """

    def freq_func(ts: Union[jndarray, float]) -> Union[jndarray, float]:
        f = jnp.empty_like(ts)
        for k, coeff in enumerate(coeffs):
            f += coeff * (ts ** k)
        return f

    def phase_func(ts: Union[jndarray, float]) -> Union[jndarray, float]:
        p = jnp.empty_like(ts)
        for k, coeff in enumerate(coeffs):
            p += coeff / (k + 1) * (ts ** (k + 1))
        return p

    return freq_func, phase_func


def meow_freq(mag: float = 500, scale: float = 5, offset: float = 5.5) -> Tuple[Callable[[jndarray], jndarray],
                                                                                Callable[[jndarray], jndarray]]:
    r"""This freq function is tricky for polynomials to fit due to its flat tails. Usually need high polynomial order.
    Too lazy to give it a name, so "meow" it is.

    Phase func:

    .. math::

        a \, e^{-b / sin(x)} + c \, x.

    Frequency func:

    .. math::

        a \, b cot(x) csc(x) \, e^{-b csc(x)} + c.

    Parameters
    ----------
    mag : float, default=500
        a.
    scale : float, default=5
        b.
    offset : float, default=5.5
        c.

    Returns
    -------
    Callable, Callable
        Frequency and phase functions.

    Notes
    -----
    The frequency and phase functions are valid in :math:`(0, \pi)`.
    """

    def freq_func(ts: jndarray) -> jndarray:
        return mag * scale * jnp.cos(ts) / (jnp.sin(ts) ** 2) * jnp.exp(-scale / jnp.sin(ts)) + offset

    def phase_func(ts: jndarray) -> jndarray:
        return mag * jnp.exp(-scale / jnp.sin(ts)) + offset * ts

    return freq_func, phase_func
