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
Gauss--Newton and levenberg--Marquardt.

This implementation is a bit ad-hoc to this package, please refer to that of `jaxopt` for a better implementation.
"""
import jax
import jax.numpy as jnp
from chirpgp.models import jndarray
from typing import Tuple, Callable


def _common_iterative_loop(update_func: Callable[[jndarray, float], Tuple[jndarray, float]],
                           obj_func: Callable[[jndarray], float],
                           init_params: jndarray,
                           init_lr: float,
                           init_diff: float,
                           stop_tolerance: float) -> Tuple[jndarray, jndarray]:
    """The common iteration procedure between Gauss--Newton and Levenberg--Marquardt.
    """
    params = init_params
    lr = init_lr
    obj_diff = init_diff
    obj_vals = []

    # Init loop
    new_obj_val = obj_func(params)
    obj_vals.append(new_obj_val)

    # Loop
    while obj_diff > stop_tolerance:
        old_obj_val = new_obj_val
        params, lr = update_func(params, lr)
        new_obj_val = obj_func(params)
        obj_diff = jnp.abs(new_obj_val - old_obj_val)
        obj_vals.append(new_obj_val)

    return params, jnp.asarray(obj_vals)


def gauss_newton(f: Callable[[jndarray], jndarray],
                 init_params: jndarray,
                 ys: jnp.ndarray,
                 Xi: float,
                 lr: float = 1.,
                 stop_tolerance: float = 1e-10,
                 init_diff: float = 1.e2) -> Tuple[jndarray, jndarray]:
    r"""Gauss--Newton method.

    Applies to models of the form

    .. math::

        Y_k = f(X_k, \theta) + \xi_k.

    Measurement dimension: 1.
    Number of measurements: T.
    Param dimension: d.

    Parameters
    ----------
    f : Callable (d, ) -> (T, )
        Measurement function.
    init_params : jnp.ndarray (d, )
        Initial parameters.
    ys : jnp.ndarray (T, )
        Measurements.
    Xi : float
        Measurement variance.
    lr : float, default=1.
        Learning rate.
    stop_tolerance : float, default=1e-10
        Stop tolerance in terms of objective value difference.
    init_diff : float, default=1e2
        Initial objective value difference.

    Returns
    -------
    jnp.ndarray, jnp.ndarray
        Optimised parameters and objective values.
    """

    def residual_func(params: jndarray) -> jndarray:
        return ys - f(params)

    def obj_func(params: jndarray) -> float:
        return jnp.sum(residual_func(params) ** 2 / Xi)

    @jax.jit
    def update(params: jndarray, _lr: float) -> Tuple[jndarray, float]:
        """_lr is a dummy argument (i.e., not used)
        """
        jac = jax.jacfwd(f)(params)
        _, vjp_fun = jax.vjp(f, params)
        inc = jnp.linalg.solve(jac.T @ jac, vjp_fun(residual_func(params))[0])
        return params + lr * inc, lr

    return _common_iterative_loop(update, obj_func, init_params, lr, init_diff, stop_tolerance)


def levenberg_marquardt(f: Callable[[jndarray], jndarray], init_params: jndarray, ys: jnp.ndarray, Xi: float,
                        lr: float = 1., nu: float = 2.,
                        stop_tolerance: float = 1e-10,
                        init_diff: float = 1.e2) -> Tuple[jndarray, jndarray]:
    r"""Levenberg--Marquardt.

    See the docstring of :code:`gauss_newton`.

    Parameters
    ----------
    f : Callable (d, ) -> (T, )
        Measurement function.
    init_params : jnp.ndarray (d, )
        Initial parameters.
    ys : jnp.ndarray (T, )
        Measurements.
    Xi : float
        Measurement variance.
    lr : float
        Initial learning rate.
    nu : float
        Parameter magnifying the change of learning rate.
    stop_tolerance : float, default=1e-10
        Stop tolerance in terms of objective value difference.
    init_diff : float, default=1e2
        Initial objective value difference.

    Returns
    -------
    jnp.ndarray, jnp.ndarray
        Optimised parameters and objective values.
    """

    def residual_func(params: jndarray) -> jndarray:
        return ys - f(params)

    def obj_func(params: jndarray) -> float:
        return jnp.sum(residual_func(params) ** 2 / Xi)

    @jax.jit
    def update(params: jndarray, _lr: float) -> Tuple[jndarray, float]:
        jac = jax.jacfwd(f)(params)
        gain = jac.T @ jac
        _, vjp_fun = jax.vjp(f, params)
        inc = jnp.linalg.solve(gain + _lr * jnp.diagflat(jnp.diagonal(gain)), vjp_fun(residual_func(params))[0])
        updated_params = params + inc
        return jax.lax.cond(obj_func(updated_params) < obj_func(params),
                            lambda u: (updated_params, _lr / nu),
                            lambda u: (params, _lr * nu),
                            jnp.array(0.))

    return _common_iterative_loop(update, obj_func, init_params, lr, init_diff, stop_tolerance)
